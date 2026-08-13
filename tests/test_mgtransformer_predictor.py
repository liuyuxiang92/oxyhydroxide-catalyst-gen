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
p.add_argument("--target", required=True)
p.add_argument("--ckpt", default=None)
p.add_argument("--config", default=None)
p.add_argument("--device", default="cpu")
p.add_argument("--cutoff", default=None)
p.add_argument("--max-neighbors", default=None)
p.add_argument("--atom-features", default=None)
p.add_argument("--triplet-endpoint", default=None)
p.add_argument("--triplet-pad-mode", default=None)
args = p.parse_args()

print(json.dumps({"status": "ready", "ckpt": args.ckpt or "dummy"}))
sys.stdout.flush()

for line in sys.stdin:
    path = line.strip()
    if not path:
        continue
    if args.target == "will_error":
        print(json.dumps({"error": "boom"}))
        sys.stdout.flush()
        continue
    if args.target == "echo_cutoff":
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


def _fake_mgt_repo(tmp_path):
    (tmp_path / "serve.py").write_text(_FAKE_SERVE_SRC)
    return str(tmp_path)


def _base_cfg(tmp_path, **over):
    cfg = {
        "mgt_repo": _fake_mgt_repo(tmp_path),
        "mgt_python": sys.executable,
        "target": "formation_energy_peratom",
    }
    cfg.update(over)
    return cfg


def test_missing_mgt_repo_raises():
    with pytest.raises(ValueError) as info:
        MGTransformerPredictor({"mgt_python": sys.executable, "target": "t"})
    assert "mgt_repo" in str(info.value)


def test_missing_mgt_python_raises(tmp_path):
    with pytest.raises(ValueError) as info:
        MGTransformerPredictor({"mgt_repo": str(tmp_path), "target": "t"})
    assert "mgt_python" in str(info.value)


def test_missing_target_raises(tmp_path):
    with pytest.raises(ValueError) as info:
        MGTransformerPredictor({"mgt_repo": str(tmp_path), "mgt_python": sys.executable})
    assert "target" in str(info.value)


def test_missing_serve_script_raises(tmp_path):
    # mgt_repo exists but has no serve.py in it.
    with pytest.raises(FileNotFoundError) as info:
        MGTransformerPredictor({
            "mgt_repo": str(tmp_path), "mgt_python": sys.executable, "target": "t",
        })
    assert "serve.py" in str(info.value)


def test_predict_structures_round_trip(tmp_path):
    from ase import Atoms

    predictor = MGTransformerPredictor(_base_cfg(tmp_path))
    try:
        a3 = Atoms("H3", positions=[(0, 0, 0), (1, 0, 0), (2, 0, 0)], cell=[10, 10, 10], pbc=True)
        a5 = Atoms("H5", positions=[(i, 0, 0) for i in range(5)], cell=[10, 10, 10], pbc=True)
        mean, std = predictor.predict_structures([a3, a5])
        assert mean == pytest.approx(4.0)          # (3 + 5) / 2
        assert std == pytest.approx(1.0)            # population std of [3, 5]
    finally:
        predictor.close()


def test_predict_structures_single_structure(tmp_path):
    from ase import Atoms

    predictor = MGTransformerPredictor(_base_cfg(tmp_path))
    try:
        a4 = Atoms("H4", positions=[(i, 0, 0) for i in range(4)], cell=[10, 10, 10], pbc=True)
        mean, std = predictor.predict_structures([a4])
        assert mean == pytest.approx(4.0)
        assert std == pytest.approx(0.0)
    finally:
        predictor.close()


def test_predict_structures_error_propagates(tmp_path):
    from ase import Atoms

    predictor = MGTransformerPredictor(_base_cfg(tmp_path, target="will_error"))
    try:
        a1 = Atoms("H", positions=[(0, 0, 0)], cell=[10, 10, 10], pbc=True)
        with pytest.raises(RuntimeError) as info:
            predictor.predict_structures([a1])
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

    # target="echo_cutoff" makes the fake server ignore the structure and
    # return --cutoff verbatim -- this only matches if MGTransformerPredictor
    # actually forwarded cutoff=6.0 onto the serve.py command line.
    predictor = MGTransformerPredictor(_base_cfg(tmp_path, target="echo_cutoff", cutoff=6.0))
    try:
        a2 = Atoms("H2", positions=[(0, 0, 0), (1, 0, 0)], cell=[10, 10, 10], pbc=True)
        mean, _std = predictor.predict_structures([a2])
        assert mean == pytest.approx(6.0)
    finally:
        predictor.close()
