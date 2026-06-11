"""Plug-in registry for predictors and constraint filters.

Two extension paths:

1. **Short name** — `predictor: ooh` or `constraint_filter: smact_charge` in
   YAML resolves to a built-in factory registered in :data:`PREDICTORS` /
   :data:`CONSTRAINTS`.

2. **Fully-qualified name** — `predictor: "my_pkg.my_mod:MyClass"` loads
   ``MyClass`` from ``my_pkg.my_mod`` via :mod:`importlib`. The class is
   instantiated as ``MyClass(cfg, **ctx)`` where ``ctx`` is keyword-only
   context (``seed=...`` for predictors, ``env=...`` for constraints).

User predictors only need ``__init__(self, cfg, *, seed=None)`` and
``predict(self, composition) -> (mean, std)``. Zero edits to
``scripts/run_experiment.py``.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Optional


# Factory signature: (cfg: dict, **ctx: Any) -> instance
Factory = Callable[..., Any]


# ---------------------------------------------------------------------------
# Built-in predictor factories
# ---------------------------------------------------------------------------

def _make_structure_score(cfg: dict, *, seed: Optional[int] = None, **_):
    from .predictors.structure_score import StructureScorePredictor
    return StructureScorePredictor(cfg, seed=seed)


def _make_sinter_calcine(cfg: dict, **_):
    from .predictors.sinter_calcine import SinterCalcineRFPredictor
    return SinterCalcineRFPredictor(
        rf_model_path=cfg["rf_model"],
        mode=cfg.get("mode", "sinter"),
    )


def _make_ooh(cfg: dict, *, seed: Optional[int] = None, **_):
    from .predictors.ooh import OOHCatalystPredictor
    return OOHCatalystPredictor(
        base_poscar=cfg["base_poscar"],
        dp_models=cfg["dp_models"],
        objective=cfg.get("objective", "mean_minus_kstd"),
        k=float(cfg.get("k", 1.0)),
        n_random_configs=int(cfg.get("n_random_configs", 10)),
        ads_height=float(cfg.get("ads_height", 1.9)),
        ads_dz=float(cfg.get("ads_dz", 1.0)),
        geo_opt=bool(cfg.get("geo_opt", False)),
        geo_opt_model=str(cfg.get("geo_opt_model", "./DPA-3.1-3M_1.pt")),
        rng_seed=seed if seed is not None else 123,
        uncertainty=cfg.get("uncertainty", "models"),
        output_index=int(cfg.get("output_index", 0)),
    )


def _make_dummy(cfg: dict, **_):
    import random as _random

    class _DummyPredictor:
        def predict(self, composition):
            return float(_random.gauss(0.5, 0.1)), float(abs(_random.gauss(0.05, 0.01)))

        def batch_predict(self, compositions):
            return [self.predict(c) for c in compositions]

    return _DummyPredictor()


PREDICTORS: Dict[str, Factory] = {
    "structure_score":    _make_structure_score,
    "sinter_calcine":     _make_sinter_calcine,
    "ooh":                _make_ooh,
    "dummy":              _make_dummy,
}


# ---------------------------------------------------------------------------
# Built-in structure-builder factories (composition/groups -> ASE Atoms)
# ---------------------------------------------------------------------------

def _make_substitute_builder(cfg: dict, *, seed: Optional[int] = None, **_):
    from .predictors.builders.substitute import SubstituteBuilder
    return SubstituteBuilder(cfg, seed=seed)


def _make_sse_builder(cfg: dict, *, seed: Optional[int] = None, **_):
    from .predictors.builders.sse import SSESupercellBuilder
    return SSESupercellBuilder(cfg, seed=seed)


BUILDERS: Dict[str, Factory] = {
    "substitute": _make_substitute_builder,
    "sse":        _make_sse_builder,
}


def resolve_builder(kind: str, cfg: dict, *, seed: Optional[int] = None):
    """Build a structure-builder from a short name or ``pkg.module:ClassName`` FQN."""
    if ":" in kind:
        cls = _load_fqn(kind)
        return cls(cfg, seed=seed)
    factory = BUILDERS.get(kind.lower())
    if factory is None:
        raise ValueError(
            f"Unknown builder {kind!r}. Built-ins: {sorted(BUILDERS)}. "
            f"For a custom builder, use the FQN form: 'pkg.module:ClassName'."
        )
    return factory(cfg, seed=seed)


# ---------------------------------------------------------------------------
# Built-in constraint factories
# ---------------------------------------------------------------------------

def _make_smact_charge(cfg: dict, **_):
    from .constraints.smact_filter import SMACTChargeFilter
    if "smact_anions" in cfg:
        anions = cfg["smact_anions"]
    else:
        anions = [{
            "symbol": cfg.get("smact_anion", "O"),
            "charge": int(cfg.get("smact_anion_charge", -2)),
            "stoich": float(cfg.get("smact_anion_stoich", 1.5)),
        }]
    return SMACTChargeFilter(anions=anions)


def _make_last_step_element(cfg: dict, **_):
    from .constraints.last_step_element import LastStepElementFilter
    return LastStepElementFilter(
        required_elements=cfg["required_elements"],
        nonzero_ratio=bool(cfg.get("nonzero_ratio_at_last", True)),
        reserve_for_last=bool(cfg.get("reserve_for_last", True)),
    )


def _make_phase_pattern(cfg: dict, **_):
    from .constraints.phase_pattern import PhasePatternFilter
    return PhasePatternFilter(patterns=cfg["phase_patterns"])


def _make_ooh_phase(cfg: dict, *, env=None, **_):
    if env is None:
        raise ValueError(
            "constraint_filter 'ooh_phase' requires env (needs _allowed_units "
            "and _possible_sums_by_k). Build the env without a filter first, "
            "then call resolve_constraint(..., env=env)."
        )
    from abcde_ooh.constraints.phase_sampler import PhaseActionFilter
    return PhaseActionFilter(
        target_phase=cfg.get("target_phases", ["any"]),
        allowed_units=env._allowed_units,
        possible_sums_by_k=env._possible_sums_by_k,
    )


def _make_chain(cfg: dict, *, env=None, **_):
    from .constraints.chain import ChainConstraintFilter
    return ChainConstraintFilter(cfg, env=env)


def _make_sse_doping(cfg: dict, *, env=None, **_):
    from .constraints.sse_doping import SSEDopingFilter
    return SSEDopingFilter(cfg, env=env)


def _make_host_complement(cfg: dict, *, env=None, **_):
    from .constraints.host_complement import HostComplementFilter
    return HostComplementFilter(cfg, env=env)


CONSTRAINTS: Dict[str, Factory] = {
    "smact_charge":      _make_smact_charge,
    "last_step_element": _make_last_step_element,
    "ooh_phase":         _make_ooh_phase,
    "phase_pattern":     _make_phase_pattern,
    "chain":             _make_chain,
    "sse_doping":        _make_sse_doping,
    "host_complement":   _make_host_complement,
}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _load_fqn(kind: str) -> type:
    """Resolve ``"pkg.module:ClassName"`` → class object."""
    try:
        mod_path, cls_name = kind.rsplit(":", 1)
    except ValueError:
        raise ValueError(
            f"Bad FQN {kind!r}: expected 'pkg.module:ClassName' "
            "(use ':' to separate module path from class)."
        )
    try:
        mod = importlib.import_module(mod_path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Cannot import module {mod_path!r} for plug-in {kind!r}. "
            "Make sure the package is installed and on PYTHONPATH."
        ) from e
    try:
        return getattr(mod, cls_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module {mod_path!r} has no attribute {cls_name!r} for plug-in {kind!r}."
        ) from e


def resolve_predictor(kind: str, cfg: dict, *, seed: Optional[int] = None):
    """Build a predictor instance from a short name or FQN."""
    if ":" in kind:
        cls = _load_fqn(kind)
        return cls(cfg, seed=seed)
    factory = PREDICTORS.get(kind.lower())
    if factory is None:
        raise ValueError(
            f"Unknown predictor {kind!r}. "
            f"Built-ins: {sorted(PREDICTORS)}. "
            f"For a custom predictor, use the FQN form: 'pkg.module:ClassName'."
        )
    return factory(cfg, seed=seed)


def resolve_constraint(kind: Optional[str], cfg: dict, *, env=None):
    """Build a constraint filter instance from a short name or FQN, or None."""
    if kind is None:
        return None
    if ":" in kind:
        cls = _load_fqn(kind)
        return cls(cfg, env=env)
    factory = CONSTRAINTS.get(kind)
    if factory is None:
        raise ValueError(
            f"Unknown constraint_filter {kind!r}. "
            f"Built-ins: {sorted(CONSTRAINTS)}. "
            f"For a custom filter, use the FQN form: 'pkg.module:ClassName'."
        )
    return factory(cfg, env=env)
