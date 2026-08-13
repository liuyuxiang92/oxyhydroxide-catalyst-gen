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


def _make_rf_magpie(cfg: dict, **_):
    from .predictors.rf_magpie import RFMagpiePredictor
    return RFMagpiePredictor(model_path=cfg["model"])


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
        # Absent key => None => all three intermediates (historical behaviour).
        # An explicit empty list means the bare parent slab, so `.get(...)` must
        # not collapse [] to the default.
        adsorbates=cfg.get("adsorbates", None),
    )


def _make_dummy(cfg: dict, **_):
    import random as _random

    class _DummyPredictor:
        def predict(self, composition):
            return float(_random.gauss(0.5, 0.1)), float(abs(_random.gauss(0.05, 0.01)))

        def batch_predict(self, compositions):
            return [self.predict(c) for c in compositions]

    return _DummyPredictor()


# Leaf predictors usable as a flat `predictor:` or as a `properties[].predictor:`
# inside the reward engine. `dp_energy`/`dp_property` are NOT here — they are
# structure-scoring branches handled internally by the engine (they need its
# build/relax machinery), not standalone composition leaves.
PREDICTORS: Dict[str, Factory] = {
    "structure_score":    _make_structure_score,   # the multi-objective reward engine
    "rf_magpie":          _make_rf_magpie,
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


def _make_site_pick_builder(cfg: dict, *, seed: Optional[int] = None, **_):
    from .predictors.builders.site_pick import SitePickBuilder
    return SitePickBuilder(cfg, seed=seed)


BUILDERS: Dict[str, Factory] = {
    "substitute": _make_substitute_builder,
    "sse":        _make_sse_builder,
    "site_pick":  _make_site_pick_builder,
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

def _resolve_scaffold(cfg: dict):
    """Resolve ``(scaffold_fixed, n_sites, anions)`` shared by the smact_charge and
    pauling_en factories (so the two never diverge).

    Scaffold = "the whole formula before substitution" (host + fixed anions that
    are NOT optimised by the agent). Three ways to supply it, in priority order:
      1. scaffold_poscar (+ scaffold_site_symbol) — parse a template (perovskite).
      2. scaffold_formula / anion_formula — an explicit per-f.u. fixed formula.
      3. none — the agent picks every element itself (oxides).
    """
    scaffold_fixed = None
    n_sites = 1
    if cfg.get("scaffold_poscar"):
        from .constraints.charge import template_scaffold

        site_symbol = cfg.get("scaffold_site_symbol", cfg.get("site_symbol", "X"))
        scaffold_fixed, n_sites = template_scaffold(cfg["scaffold_poscar"], site_symbol)
    elif cfg.get("scaffold_formula") or cfg.get("anion_formula"):
        from .constraints.charge import parse_formula

        scaffold_fixed = parse_formula(cfg.get("scaffold_formula") or cfg.get("anion_formula"))

    # Deprecated explicit anion list (old oxide configs) — passed through for
    # backward compatibility; the filter ignores anions the agent picks itself.
    anions = cfg.get("smact_anions")
    if anions is None and any(
        k in cfg for k in ("smact_anion", "smact_anion_charge", "smact_anion_stoich")
    ):
        anions = [{
            "symbol": cfg.get("smact_anion", "O"),
            "charge": int(cfg.get("smact_anion_charge", -2)),
            "stoich": float(cfg.get("smact_anion_stoich", 1.5)),
        }]
    return scaffold_fixed, n_sites, anions


def _make_smact_charge(cfg: dict, **_):
    from .constraints.smact_filter import SMACTChargeFilter

    scaffold_fixed, n_sites, anions = _resolve_scaffold(cfg)
    return SMACTChargeFilter(
        anions=anions,
        scaffold_fixed=scaffold_fixed,
        scaffold_n_sites=n_sites,
    )


def _make_pauling_en(cfg: dict, **_):
    """``pauling_en`` — charge neutrality AND Pauling electronegativity (smact)."""
    from .constraints.smact_filter import ElectronegativityFilter

    scaffold_fixed, n_sites, anions = _resolve_scaffold(cfg)
    return ElectronegativityFilter(
        anions=anions,
        scaffold_fixed=scaffold_fixed,
        scaffold_n_sites=n_sites,
    )


def _has_constraint(cfg: dict, name: str) -> bool:
    if cfg.get("constraint_filter") == name:
        return True
    return any(
        isinstance(f, dict) and f.get("constraint_filter") == name
        for f in cfg.get("filters") or []
    )


def smact_charge_enabled(cfg: dict) -> bool:
    """True when a ``smact_charge`` constraint is configured (flat or in a chain)."""
    return _has_constraint(cfg, "smact_charge")


def pauling_en_enabled(cfg: dict) -> bool:
    """True when a ``pauling_en`` constraint is configured (flat or in a chain)."""
    return _has_constraint(cfg, "pauling_en")


def charge_check_enabled(cfg: dict) -> bool:
    """Single switch for the post-episode generation drop: True if either a
    ``smact_charge`` or ``pauling_en`` constraint is configured. When False the
    post-episode validation in ``generate_candidates`` does nothing (no smact
    import, output unchanged)."""
    return smact_charge_enabled(cfg) or pauling_en_enabled(cfg)


def charge_use_pauling(cfg: dict) -> bool:
    """True when the post-episode drop should also enforce Pauling EN (``pauling_en``)."""
    return pauling_en_enabled(cfg)


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
        total_units=env._total_units,
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


def _make_sum_bound(cfg: dict, *, env=None, **_):
    from .constraints.sum_bound import IndependentSumBoundFilter
    return IndependentSumBoundFilter(cfg, env=env)


CONSTRAINTS: Dict[str, Factory] = {
    "smact_charge":      _make_smact_charge,
    "pauling_en":        _make_pauling_en,
    "last_step_element": _make_last_step_element,
    "ooh_phase":         _make_ooh_phase,
    "phase_pattern":     _make_phase_pattern,
    "chain":             _make_chain,
    "sse_doping":        _make_sse_doping,
    "host_complement":   _make_host_complement,
    "sum_bound":         _make_sum_bound,
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


def build_reward(cfg: dict, *, seed: Optional[int] = None):
    """Build the run's reward predictor from a config.

    Routing is by the **shape** of the config, so there is no engine name to
    learn and no single-vs-multi mode to set:

    * ``properties:`` is a non-empty list  -> the multi-objective reward engine
      (:class:`StructureScorePredictor`). It auto-detects single (one entry) vs
      multi (many) objectives and combines them.
    * otherwise -> a flat single predictor named by the legacy ``predictor:`` key
      (registry short name or FQN).
    """
    props = cfg.get("properties")
    if isinstance(props, list) and props:
        return _make_structure_score(cfg, seed=seed)
    kind = cfg.get("predictor")
    if not kind:
        raise ValueError(
            "Reward config needs either a non-empty 'properties' list (the "
            "reward engine) or a flat 'predictor' name."
        )
    return resolve_predictor(kind, cfg, seed=seed)


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


def build_constraints(cfg: dict, *, env=None):
    """Build the constraint filter for a config, auto-detecting chaining.

    Mirrors :func:`build_reward`: routing is by the *shape* of the config, so a
    user with several constraints never has to name ``chain`` explicitly.

    * ``filters:`` is a non-empty list -> wrap them in a
      :class:`ChainConstraintFilter`, applied in list order. A single-entry list
      is fine too (it just runs that one filter, with the chain's safety
      fallback). You do **not** write ``constraint_filter: chain``.
    * otherwise -> the flat ``constraint_filter:`` (short name or FQN), or
      ``None`` when neither key is present.
    """
    filters = cfg.get("filters")
    if isinstance(filters, list) and filters:
        return _make_chain(cfg, env=env)
    return resolve_constraint(cfg.get("constraint_filter"), cfg, env=env)
