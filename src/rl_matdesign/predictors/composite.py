"""CompositePredictor — multi-objective property optimization.

Wraps N child predictors (each resolved through the registry) and combines
their raw (mean, std) outputs into a single scalar reward via a
weighted-signed-scaled sum::

    reward = Σ_k  w_k * dir_k * v_k / scale_k

where ``v_k`` is the per-child robust value::

    v_k = m_k                  if objective == "mean"
    v_k = m_k - k_c * s_k      if objective == "mean_minus_kstd"
    v_k = m_k + k_c * s_k      if objective == "mean_plus_kstd"

Each child keeps its own model-ensemble std in native physical units; no joint
``dp_std`` is computed (mixing eV/atom with GPa would be meaningless).
``generated.csv`` instead receives per-child ``obj_<name>_mean`` /
``obj_<name>_std`` columns through :meth:`per_objective_stats`.

Configuration keys
------------------
Required:
    - ``objectives``: non-empty list of sub-objective cfg dicts. Each dict has:
        * ``predictor``: short name or FQN routed through the registry.
        * ``direction``: ``"min"`` / ``"max"`` (or ``"minimize"`` / ``"maximize"``).
        * Any keys the child predictor needs (e.g. ``dp_models``, ``dp_head``).
      Optional per-objective: ``name`` (default ``obj{i}``), ``weight``
      (default ``1.0``), ``scale`` (default ``1.0``).

Composite-level optional keys (inherited by sub-cfgs that don't override):
    - ``base_poscar``, ``site_symbol``, ``n_random_configs``.
    - ``objective`` (default ``"mean_minus_kstd"``).
    - ``k`` (default ``1.0``).

Per-child RNG seeding
---------------------
Child ``k`` is constructed with ``seed = composite_seed + k`` so each child
draws independent random-site assignments. Fully deterministic given the
composite seed and the YAML order of ``objectives``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


_SHARED_DEFAULT_KEYS = ("base_poscar", "site_symbol", "n_random_configs", "poscar", "poscar_template")
_DIRECTION_ALIASES = {
    "min": -1.0, "minimize": -1.0,
    "max": +1.0, "maximize": +1.0,
}


class CompositePredictor:
    """Weighted-sum multi-objective predictor."""

    def __init__(self, cfg: Dict[str, Any], *, seed: Optional[int] = None) -> None:
        from ..registry import resolve_predictor

        self.cfg = cfg
        self.objective: str = cfg.get("objective", "mean_minus_kstd")
        self.k: float = float(cfg.get("k", 1.0))

        objectives = cfg.get("objectives")
        if not objectives:
            raise ValueError(
                "composite predictor requires a non-empty 'objectives' list."
            )

        shared_defaults = {
            key: cfg[key] for key in _SHARED_DEFAULT_KEYS if key in cfg
        }

        self.names: List[str] = []
        self.children: List[Any] = []
        self.weights: List[float] = []
        self.dirs: List[float] = []
        self.scales: List[float] = []

        seen_names: set[str] = set()
        for i, sub in enumerate(objectives):
            if not isinstance(sub, dict):
                raise ValueError(
                    f"objectives[{i}] must be a dict, got {type(sub).__name__}."
                )
            sub_cfg = dict(sub)

            name = str(sub_cfg.pop("name", f"obj{i}"))
            if name in seen_names:
                raise ValueError(
                    f"objectives[{i}].name = {name!r} duplicates an earlier objective. "
                    "Each objective needs a unique name (used as CSV column prefix)."
                )
            seen_names.add(name)

            direction = str(sub_cfg.pop("direction", "min")).lower()
            if direction not in _DIRECTION_ALIASES:
                raise ValueError(
                    f"objectives[{i}].direction = {direction!r}; expected one of "
                    f"{sorted(_DIRECTION_ALIASES)}."
                )
            dir_sign = _DIRECTION_ALIASES[direction]

            weight = float(sub_cfg.pop("weight", 1.0))
            scale = float(sub_cfg.pop("scale", 1.0))
            if scale == 0.0:
                raise ValueError(
                    f"objectives[{i}].scale must be non-zero (got 0)."
                )

            sub_kind = sub_cfg.pop("predictor", None)
            if not sub_kind:
                raise ValueError(
                    f"objectives[{i}] is missing 'predictor' (short name or "
                    "'pkg.module:ClassName' FQN)."
                )

            # Inherit shared defaults from the composite cfg only when the
            # sub-cfg doesn't set them. Sub-cfg wins on conflict.
            for key, value in shared_defaults.items():
                sub_cfg.setdefault(key, value)

            child_seed = None if seed is None else int(seed) + i
            child = resolve_predictor(sub_kind, sub_cfg, seed=child_seed)

            if not hasattr(child, "raw_mean_std"):
                raise TypeError(
                    f"objectives[{i}] predictor {sub_kind!r} does not implement "
                    "raw_mean_std(composition) -> (mean, std). Composite needs "
                    "raw, pre-objective stats to combine across properties."
                )

            self.names.append(name)
            self.children.append(child)
            self.weights.append(weight)
            self.dirs.append(dir_sign)
            self.scales.append(scale)

        self._stats_cache: Dict[tuple, Dict[str, Tuple[float, float]]] = {}

    # ------------------------------------------------------------------
    # PropertyPredictor protocol
    # ------------------------------------------------------------------

    def predict(self, composition: Dict[str, float]) -> Tuple[float, float]:
        """Return ``(reward, 0.0)`` — composite has no scalar joint std.

        Direction is applied to the raw mean *before* the std term is folded
        in, matching :class:`DPPropertyPredictor` / :class:`DPStructurePredictor`
        convention. For ``mean_minus_kstd`` this gives ``dir*m - k*s`` per
        child, so the std term acts as an uncertainty penalty regardless of
        whether the child is minimizing or maximizing (exploit semantics in
        both directions). Switch to ``mean_plus_kstd`` to flip to exploration.
        """
        from ..training import objective_from_mean_std

        stats = self._stats(composition)
        reward = 0.0
        for name, w, d, sc in zip(self.names, self.weights, self.dirs, self.scales):
            m, s = stats[name]
            v = objective_from_mean_std(d * m, s, self.objective, self.k)
            reward += w * v / sc
        return float(reward), 0.0

    def predict_raw(self, composition: Dict[str, float]) -> Tuple[float, float]:
        """Same combined value but using per-child raw means (no std penalty)."""
        stats = self._stats(composition)
        raw_combined = 0.0
        for name, w, d, sc in zip(self.names, self.weights, self.dirs, self.scales):
            m, _s = stats[name]
            raw_combined += w * d * m / sc
        return float(raw_combined), 0.0

    def per_objective_stats(
        self, composition: Dict[str, float]
    ) -> Dict[str, Tuple[float, float]]:
        """Return ``{name: (mean, std)}`` for each child, for CSV logging."""
        return dict(self._stats(composition))

    def batch_predict(
        self, compositions: List[Dict[str, float]]
    ) -> List[Tuple[float, float]]:
        return [self.predict(c) for c in compositions]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stats(self, composition: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        key = self._comp_key(composition)
        cached = self._stats_cache.get(key)
        if cached is not None:
            return cached
        stats: Dict[str, Tuple[float, float]] = {}
        for name, child in zip(self.names, self.children):
            m, s = child.raw_mean_std(composition)
            stats[name] = (float(m), float(s))
        # Single-entry cache: only the most recent composition. Generation loop
        # already memoizes at the outer level by comp_key, so this is just
        # belt-and-braces against accidental double-evaluation within one call.
        self._stats_cache = {key: stats}
        return stats

    @staticmethod
    def _comp_key(composition: Dict[str, float]) -> tuple:
        return tuple(sorted((str(el), float(frac)) for el, frac in composition.items()))
