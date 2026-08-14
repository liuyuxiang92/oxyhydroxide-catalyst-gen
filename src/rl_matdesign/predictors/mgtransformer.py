"""MGTransformerPredictor — a generic structure-scoring bridge to an external
MGTransformer checkpoint (https://github.com/... see ``../MGTransformer``).

MGTransformer's own repo has no in-process Python API usable from here: its
inference stack (torch-geometric, dgl, e3nn, transformers, jarvis-tools) is a
separate, heavier dependency set than this repo's ``deepmd`` environment, and
mixing them in one process risks a torch-version conflict. This predictor talks
to it instead via a **persistent subprocess** (``MGTransformer/serve.py``,
launched once and kept alive for this predictor's lifetime) over stdin/stdout
JSON lines — the featurizer and model live entirely in ``MGTransformer/``; see
that repo's ``graph_builder.py`` / ``predict.py`` / ``serve.py`` for what's
actually being called.

Fully config-driven, no perovskite/formation-energy assumption anywhere in this
file: point ``target``/``ckpt`` at any finetuned checkpoint under
``mgt_repo/ckpt/finetuned/`` (bandgap, bulk modulus, ehull, ...) to reuse this
class for a completely different ``structure_score`` scenario. It opts into the
**structure** objective protocol (see ``structure_score.py``'s dispatch) by
exposing ``predict_structures`` rather than ``predict`` — a property that
depends on 3D geometry, not just stoichiometry, needs the built/relaxed cells,
not the raw candidate composition.

Config keys
-----------
    mgt_repo (required):
        Path to the ``MGTransformer`` checkout.
    mgt_python (required):
        Python interpreter of MGTransformer's own conda env (its dependency set
        is incompatible with this repo's — see module docstring).
    target (required):
        Finetuned checkpoint name under ``mgt_repo/ckpt/finetuned/`` (e.g.
        ``formation_energy_peratom``).
    ckpt, config:
        Optional overrides forwarded to ``serve.py --ckpt`` / ``--config``.
    device:
        Forwarded to ``serve.py --device`` (default ``"cpu"``).
    cutoff, max_neighbors, atom_features, triplet_endpoint, triplet_pad_mode:
        Optional overrides for ``graph_builder.py``'s unverified featurizer
        hyperparameters (see that module's docstring); forwarded as the
        matching ``serve.py`` CLI flags when present in *cfg*.

Output units — see MGTransformer's own ``predict.py`` docstring: raw model
output is an **uncalibrated, relative score** for *target*, never a physical
unit (eV/atom etc.), because the training-split mean/std needed to un-normalize
it aren't recoverable from a checkpoint alone. Ranking/argmin across candidates
is unaffected by that fixed affine transform.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_GRAPH_KWARG_FLAGS = {
    "cutoff": "--cutoff",
    "max_neighbors": "--max-neighbors",
    "atom_features": "--atom-features",
    "triplet_endpoint": "--triplet-endpoint",
    "triplet_pad_mode": "--triplet-pad-mode",
}


class MGTransformerPredictor:
    def __init__(self, cfg: Dict[str, Any], *, seed: Optional[int] = None) -> None:
        self.cfg = cfg
        self._seed = seed

        mgt_repo = cfg.get("mgt_repo")
        if not mgt_repo:
            raise ValueError(
                "MGTransformerPredictor needs 'mgt_repo' (path to the MGTransformer checkout)."
            )
        mgt_python = cfg.get("mgt_python")
        if not mgt_python:
            raise ValueError(
                "MGTransformerPredictor needs 'mgt_python' (the interpreter of "
                "MGTransformer's own conda env -- its deps are incompatible with "
                "this repo's, see module docstring)."
            )
        target = cfg.get("target")
        if not target:
            raise ValueError(
                "MGTransformerPredictor needs 'target' (a finetuned checkpoint "
                "name under mgt_repo/ckpt/finetuned/, e.g. 'formation_energy_peratom')."
            )

        self.mgt_repo = os.path.abspath(str(mgt_repo))
        serve_script = os.path.join(self.mgt_repo, "serve.py")
        if not os.path.exists(serve_script):
            raise FileNotFoundError(
                f"{serve_script!r} not found. Add MGTransformer's graph_builder.py / "
                "predict.py / serve.py (this repo does not ship them) before using "
                "MGTransformerPredictor."
            )

        cmd: List[str] = [str(mgt_python), "-u", serve_script, "--target", str(target)]
        if cfg.get("ckpt"):
            cmd += ["--ckpt", str(cfg["ckpt"])]
        if cfg.get("config"):
            cmd += ["--config", str(cfg["config"])]
        cmd += ["--device", str(cfg.get("device", "cpu"))]
        for key, flag in _GRAPH_KWARG_FLAGS.items():
            if key in cfg:
                cmd += [flag, str(cfg[key])]

        self._proc = subprocess.Popen(
            cmd, cwd=self.mgt_repo,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, bufsize=1,
        )
        self._await_ready()

        # One reused scratch POSCAR path per predictor instance rather than a
        # fresh temp file per structure — this predictor's calls are sequential
        # (one structure at a time over the pipe), so there's no concurrency
        # hazard, and it avoids filesystem churn across a long sweep.
        self._tmp_dir = tempfile.mkdtemp(prefix="mgt_bridge_")
        self._tmp_poscar = os.path.join(self._tmp_dir, "candidate.vasp")
        self._closed = False

    # ------------------------------------------------------------------

    def _await_ready(self) -> None:
        line = self._proc.stdout.readline()
        if not line:
            # stdout closed with nothing read -> the process is exiting/exited.
            # wait() (not poll()) so returncode is actually populated instead
            # of racing the child's teardown and reporting None.
            try:
                code = self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                code = None
            raise RuntimeError(
                f"MGTransformer serve.py exited during startup (returncode={code}) "
                "with no output on its own stdout -- its stderr (inherited to "
                "this process's stderr) printed above this traceback and has "
                "the real error (e.g. a missing Python package in mgt_python's env)."
            )
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"MGTransformer serve.py sent a non-JSON startup line: {line!r}") from exc
        if msg.get("status") != "ready":
            raise RuntimeError(f"MGTransformer serve.py did not report ready: {msg!r}")

    def _score_one(self, atoms: "ase.Atoms") -> float:
        from ase.io import write as ase_write

        if self._proc.poll() is not None:
            raise RuntimeError(
                f"MGTransformer serve.py subprocess has exited (returncode="
                f"{self._proc.returncode}); cannot score more structures."
            )
        ase_write(self._tmp_poscar, atoms, format="vasp")
        self._proc.stdin.write(self._tmp_poscar + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            code = self._proc.poll()
            raise RuntimeError(
                f"MGTransformer serve.py closed its output unexpectedly while "
                f"scoring {self._tmp_poscar!r} (returncode={code}); its stderr "
                "(inherited to this process's stderr) printed above and has "
                "the real error."
            )
        try:
            resp = json.loads(line)
        except json.JSONDecodeError as exc:
            code = self._proc.poll()
            raise RuntimeError(
                f"MGTransformer serve.py sent a non-JSON response line while "
                f"scoring {self._tmp_poscar!r} (subprocess returncode={code}, "
                f"None means still alive): {line!r}. Its stderr (inherited to "
                "this process's stderr) has the real error."
            ) from exc
        if "error" in resp:
            raise RuntimeError(f"MGTransformer serve.py failed on {self._tmp_poscar!r}: {resp['error']}")
        return float(resp["score"])

    # ------------------------------------------------------------------
    # Structure-objective protocol (see structure_score.py's dispatch)
    # ------------------------------------------------------------------

    def predict_structures(self, atoms_list: List["ase.Atoms"]) -> Tuple[float, float]:
        scores = [self._score_one(atoms) for atoms in atoms_list]
        arr = np.asarray(scores, dtype=float)
        return float(arr.mean()), float(arr.std())

    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 - best-effort shutdown
            self._proc.kill()
        finally:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - never raise from __del__
            pass
