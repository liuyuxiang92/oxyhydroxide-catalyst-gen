"""MGTransformerPredictor tests — the subprocess bridge to ../MGTransformer.

The real MGTransformer/serve.py needs torch/torch-geometric/jarvis-tools (a
separate conda env, per the plan) and isn't available in this repo's test env.
These tests instead point the bridge at a small FAKE serve.py (same JSON-lines
protocol, no heavy deps) so the bridge's own logic — process launch, readiness
handshake, request/response round trip, error propagation, shutdown — is
covered without needing the real model.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_matdesign.predictors.mgtransformer import MGTransformerPredictor  # noqa: E402


_FAKE_SERVE_SRC = '''
import argparse, json, sys

p = argparse.ArgumentParser()
p.add_argument("--target", default=None)
p.add_argument("--ckpt", default=None)
p.add_argument("--config", default=None)
p.add_argument("--device", default="cpu")
p.add_argument("--cutoff", default=None)
p.add_argument("--max-neighbors", default=None)
p.add_argument("--atom-features", default=None)
p.add_argument("--triplet-endpoint", default=None)
p.add_argument("--triplet-pad-mode", default=None)
args = p.parse_args()

print(json.dumps({"status": "ready", "ckpt": args.ckpt or "dummy", "calibrated": True}))
sys.stdout.flush()

for line in sys.stdin:
    path = line.strip()
    if not path:
        continue
    if (args.ckpt or "").endswith("will_error.pt"):
        print(json.dumps({"error": "boom"}))
        sys.stdout.flush()
        continue
    if (args.ckpt or "").endswith("echo_cutoff.pt"):
        # Ignores the structure entirely -- proves the CLI flag round-tripped.
        print(json.dumps({"score": float(args.cutoff)}))
        sys.stdout.flush()
        continue
    # Fake "score" = total atom count, parsed straight out of the VASP POSCAR
    # (line index 6 holds per-species counts) -- deterministic and directly
    # checkable against the ase.Atoms objects the test builds.
    with open(path) as f:
        lines = f.readlines()
    n_atoms = sum(int(x) for x in lines[6].split())
    print(json.dumps({"score": float(n_atoms)}))
    sys.stdout.flush()
'''


_FAKE_SERVE_CRASH_SRC = '''
import sys
# Simulates the real failure this test pins: serve.py dying on an import
# (e.g. ModuleNotFoundError: torch_geometric) before it ever prints the
# ready line -- stdout is empty, only stderr (a real traceback) has content.
print("Traceback (most recent call last):", file=sys.stderr)
print("ModuleNotFoundError: No module named 'torch_geometric'", file=sys.stderr)
sys.exit(1)
'''


def _fake_mgt_repo(tmp_path, src=_FAKE_SERVE_SRC):
    (tmp_path / "serve.py").write_text(src)
    return str(tmp_path)


def _base_cfg(tmp_path, **over):
    cfg = {
        "mgt_repo": _fake_mgt_repo(tmp_path),
        "mgt_python": sys.executable,
        "model": "ckpt/finetuned/formation_energy_peratom/formation_energy_peratom_checkpoint_best.pt",
    }
    cfg.update(over)
    return cfg


def test_missing_mgt_repo_raises():
    with pytest.raises(ValueError) as info:
        MGTransformerPredictor({"mgt_python": sys.executable, "model": "m.pt"})
    assert "mgt_repo" in str(info.value)


def test_missing_mgt_python_raises(tmp_path):
    with pytest.raises(ValueError) as info:
        MGTransformerPredictor({"mgt_repo": str(tmp_path), "model": "m.pt"})
    assert "mgt_python" in str(info.value)


def test_missing_model_raises(tmp_path):
    with pytest.raises(ValueError) as info:
        MGTransformerPredictor({"mgt_repo": str(tmp_path), "mgt_python": sys.executable})
    assert "model" in str(info.value)


def test_model_list_is_rejected_at_the_leaf(tmp_path):
    # A leaf is ONE model. An ensemble is expressed at the properties[] level, where
    # the engine builds one instance per path -- see structure_score's contract.
    with pytest.raises(ValueError) as info:
        MGTransformerPredictor({
            "mgt_repo": str(tmp_path), "mgt_python": sys.executable,
            "model": ["a.pt", "b.pt"],
        })
    assert "single 'model'" in str(info.value)


def test_missing_serve_script_raises(tmp_path):
    # mgt_repo exists but has no serve.py in it.
    with pytest.raises(FileNotFoundError) as info:
        MGTransformerPredictor({
            "mgt_repo": str(tmp_path), "mgt_python": sys.executable, "model": "m.pt",
        })
    assert "serve.py" in str(info.value)


def test_crashed_subprocess_reports_real_returncode_not_none(tmp_path):
    # Regression test: a serve.py that dies before printing the ready line
    # (e.g. a missing dependency in mgt_python's env, ModuleNotFoundError)
    # used to report "returncode=None" because poll() raced the child's
    # teardown -- wait() must be used so the real exit code is reported.
    cfg = {
        "mgt_repo": _fake_mgt_repo(tmp_path, src=_FAKE_SERVE_CRASH_SRC),
        "mgt_python": sys.executable,
        "model": "ckpt/finetuned/formation_energy_peratom/formation_energy_peratom_checkpoint_best.pt",
    }
    with pytest.raises(RuntimeError) as info:
        MGTransformerPredictor(cfg)
    msg = str(info.value)
    assert "returncode=None" not in msg
    assert "returncode=1" in msg


def test_score_structures_round_trip(tmp_path):
    from ase import Atoms

    predictor = MGTransformerPredictor(_base_cfg(tmp_path))
    try:
        a3 = Atoms("H3", positions=[(0, 0, 0), (1, 0, 0), (2, 0, 0)], cell=[10, 10, 10], pbc=True)
        a5 = Atoms("H5", positions=[(i, 0, 0) for i in range(5)], cell=[10, 10, 10], pbc=True)
        scores = predictor.score_structures([a3, a5])
        # ONE value per structure, in order, with no folding: the engine owns the
        # mean/std (a leaf is one model, so it must not report an uncertainty).
        assert list(scores) == [pytest.approx(3.0), pytest.approx(5.0)]
    finally:
        predictor.close()


def test_score_structures_single_structure(tmp_path):
    from ase import Atoms

    predictor = MGTransformerPredictor(_base_cfg(tmp_path))
    try:
        a4 = Atoms("H4", positions=[(i, 0, 0) for i in range(4)], cell=[10, 10, 10], pbc=True)
        scores = predictor.score_structures([a4])
        assert list(scores) == [pytest.approx(4.0)]
    finally:
        predictor.close()


def test_score_structures_error_propagates(tmp_path):
    from ase import Atoms

    predictor = MGTransformerPredictor(_base_cfg(tmp_path, model="will_error.pt"))
    try:
        a1 = Atoms("H", positions=[(0, 0, 0)], cell=[10, 10, 10], pbc=True)
        with pytest.raises(RuntimeError) as info:
            predictor.score_structures([a1])
        assert "boom" in str(info.value)
    finally:
        predictor.close()


def test_close_is_idempotent_and_cleans_tmp_dir(tmp_path):
    predictor = MGTransformerPredictor(_base_cfg(tmp_path))
    scratch_dir = predictor._tmp_dir
    assert os.path.isdir(scratch_dir)
    predictor.close()
    assert not os.path.isdir(scratch_dir)
    predictor.close()  # must not raise on a second call


def test_forwards_graph_kwargs_to_serve_subprocess(tmp_path):
    from ase import Atoms

    # A checkpoint named echo_cutoff.pt makes the fake server ignore the structure
    # and return --cutoff verbatim -- this only matches if MGTransformerPredictor
    # actually forwarded cutoff=6.0 onto the serve.py command line.
    predictor = MGTransformerPredictor(_base_cfg(tmp_path, model="echo_cutoff.pt", cutoff=6.0))
    try:
        a2 = Atoms("H2", positions=[(0, 0, 0), (1, 0, 0)], cell=[10, 10, 10], pbc=True)
        scores = predictor.score_structures([a2])
        assert list(scores) == [pytest.approx(6.0)]
    finally:
        predictor.close()


def test_model_path_is_forwarded_as_ckpt(tmp_path):
    # The bridge points serve.py at a checkpoint PATH rather than naming a target;
    # MGTransformer derives the target (and its calibration entry) from that path.
    predictor = MGTransformerPredictor(_base_cfg(tmp_path, model="some/dir/thing.pt"))
    try:
        assert predictor.model_path == "some/dir/thing.pt"
        cmd = predictor._proc.args
        assert "--ckpt" in cmd
        assert cmd[cmd.index("--ckpt") + 1] == "some/dir/thing.pt"
        assert "--target" not in cmd
    finally:
        predictor.close()
