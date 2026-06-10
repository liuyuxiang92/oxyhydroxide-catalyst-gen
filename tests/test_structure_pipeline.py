"""Orchestration test for StructurePipelinePredictor (no DeepMD).

Stubs the builder and per-property eval to verify the build-once / weighted-combine
logic. The DeepMD ensemble eval + geo-opt are GPU-gated and tested separately.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_matdesign.predictors.structure_pipeline import StructurePipelinePredictor  # noqa: E402


def _cfg():
    # A valid 'sse' builder cfg so __init__ succeeds; we stub .builder afterwards.
    return {
        "builder": "sse",
        "base_poscar": "/dev/null",
        "valences": {"Li": 1, "P": 5, "S": -2, "O": -2, "Cl": -1, "Br": -1,
                     "Mn": {"sulfide": 2, "oxide": 4}},
        # geo_opt omitted -> relaxation disabled (no DeepMD needed)
        "k": 1.0,
        "properties": [
            {"name": "conductivity", "models": ["m1.pt"], "head": "experiment",
             "direction": "max", "weight": 2.0, "scale": 1.0, "objective": "mean_minus_kstd"},
            {"name": "stability", "models": ["s1.pt"], "head": None,
             "direction": "max", "weight": 1.0, "scale": 5.0, "objective": "mean_minus_kstd"},
        ],
    }


def test_weighted_combine_of_two_properties():
    p = StructurePipelinePredictor(_cfg())
    assert p.geo_opt_enabled is False  # no geo_opt block -> relaxation off

    # Stub: builder returns a dummy "structure"; eval returns known (mean, std).
    p.builder = type("B", (), {"build": lambda self, cand, *, n_configs, rng: [object()]})()
    fixed = {"conductivity": (10.0, 1.0), "stability": (5.0, 0.5)}
    p._eval_property = lambda prop, structures: fixed[prop["name"]]

    reward, std = p.predict({"P_site": {"P": 0.95, "Mn": 0.05}})

    # direction=max, mean_minus_kstd, k=1: v = mean - std.
    #   conductivity: 2 * (10 - 1) / 1   = 18.0
    #   stability:    1 * (5  - 0.5) / 5 = 0.9
    assert abs(reward - 18.9) < 1e-9
    assert abs(std - (1.0 + 0.5) / 2) < 1e-9  # mean of per-property stds


def test_registry_resolves_pipeline_and_builder():
    from rl_matdesign.registry import resolve_predictor, resolve_builder, PREDICTORS, BUILDERS

    assert "structure_pipeline" in PREDICTORS
    assert "sse" in BUILDERS
    pred = resolve_predictor("structure_pipeline", _cfg(), seed=0)
    assert isinstance(pred, StructurePipelinePredictor)
    b = resolve_builder("sse", {"base_poscar": "/dev/null",
                                "valences": {"Li": 1, "P": 5, "S": -2}}, seed=0)
    assert hasattr(b, "build")
