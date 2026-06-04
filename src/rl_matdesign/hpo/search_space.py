"""Search-space sampling and validation for the HPO driver.

A search-space spec is a mapping ``{param_name: {dist: ..., ...}}``. Supported
distribution shapes:

- ``{dist: uniform,    low: float, high: float}``
- ``{dist: loguniform, low: float, high: float}``
- ``{dist: int,        low: int,   high: int,   log: bool=False, step: int=1}``
- ``{dist: int_log,    low: int,   high: int}``  (alias for int with log=True, step=1)
- ``{dist: categorical, choices: [...]}``
- ``{dist: frac_of,    base: <other_param_name>, low: float, high: float}``
    Sampled as a uniform fraction in ``[low, high]`` and multiplied by the
    resolved value of ``base`` (a fixed override or a base-config value the
    driver passes in via ``base_values``). Result is ``int(round(frac * base))``
    when ``base`` is an integer-typed key, else ``frac * base``.

``sample_from_search_space`` returns ``{param_name: sampled_value}`` and is
agnostic to which RL method the params apply to — validation against the
method is the caller's job (e.g. via ``validate_search_space``).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Set


_INT_TYPED_KEYS: Set[str] = {
    # Episode / iteration counts (DQN)
    "dqn_warmup_eps",
    "dqn_num_train_eps",
    "dqn_buffer_size",
    "dqn_batch_size",
    "dqn_grad_steps_per_ep",
    "dqn_target_update_freq",
    "dqn_eps_anneal_eps",
    "dqn_hidden_dim",
    "dqn_augment_permutations",
    # PG
    "pg_warmup_eps",
    "pg_num_iters",
    "pg_batch_eps",
    # Generation
    "num_gen_eps",
    "exploration_gen_eps",
}


class SearchSpaceError(ValueError):
    """Raised when a search-space spec is malformed or self-inconsistent."""


def validate_search_space(
    spec: Mapping[str, Mapping[str, Any]],
    *,
    fixed_overrides: Mapping[str, Any] | None = None,
    base_values: Mapping[str, Any] | None = None,
) -> None:
    """Check the spec is well-formed before any sampling happens.

    Catches:
    - Unknown ``dist`` values.
    - Missing required fields (``low``/``high``/``choices``/``base``).
    - ``frac_of`` referring to a base that is neither in ``fixed_overrides``
      nor ``base_values`` (the driver passes the base scenario YAML's values
      in ``base_values``).
    """
    fixed_overrides = fixed_overrides or {}
    base_values = base_values or {}

    if not isinstance(spec, Mapping):
        raise SearchSpaceError(f"search_space must be a mapping, got {type(spec).__name__}")

    for name, entry in spec.items():
        if not isinstance(entry, Mapping):
            raise SearchSpaceError(f"search_space[{name!r}] must be a mapping, got {type(entry).__name__}")
        dist = entry.get("dist")
        if dist is None:
            raise SearchSpaceError(f"search_space[{name!r}] missing required 'dist' field")
        if dist in ("uniform", "loguniform"):
            for k in ("low", "high"):
                if k not in entry:
                    raise SearchSpaceError(f"search_space[{name!r}] (dist={dist}) missing {k!r}")
            if dist == "loguniform" and (entry["low"] <= 0 or entry["high"] <= 0):
                raise SearchSpaceError(f"search_space[{name!r}] loguniform requires low > 0 and high > 0")
            if entry["low"] >= entry["high"]:
                raise SearchSpaceError(f"search_space[{name!r}] requires low < high")
        elif dist in ("int", "int_log"):
            for k in ("low", "high"):
                if k not in entry:
                    raise SearchSpaceError(f"search_space[{name!r}] (dist={dist}) missing {k!r}")
            if int(entry["low"]) > int(entry["high"]):
                raise SearchSpaceError(f"search_space[{name!r}] requires low <= high")
            if dist == "int_log" and int(entry["low"]) < 1:
                raise SearchSpaceError(f"search_space[{name!r}] int_log requires low >= 1")
        elif dist == "categorical":
            choices = entry.get("choices")
            if not isinstance(choices, (list, tuple)) or len(choices) == 0:
                raise SearchSpaceError(f"search_space[{name!r}] categorical needs non-empty 'choices' list")
        elif dist == "frac_of":
            base = entry.get("base")
            if base is None:
                raise SearchSpaceError(f"search_space[{name!r}] frac_of missing 'base'")
            if base not in fixed_overrides and base not in base_values:
                raise SearchSpaceError(
                    f"search_space[{name!r}] frac_of base={base!r} not found in "
                    "fixed_overrides or base config — driver cannot resolve it"
                )
            for k in ("low", "high"):
                if k not in entry:
                    raise SearchSpaceError(f"search_space[{name!r}] frac_of missing {k!r}")
            if entry["low"] >= entry["high"]:
                raise SearchSpaceError(f"search_space[{name!r}] requires low < high")
        else:
            raise SearchSpaceError(f"search_space[{name!r}] unknown dist={dist!r}")


def sample_from_search_space(
    trial,
    spec: Mapping[str, Mapping[str, Any]],
    *,
    resolved_values: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Sample a single point from the search space using ``trial`` (Optuna trial).

    ``resolved_values`` provides values for ``frac_of`` base lookups (the driver
    passes the merged fixed_overrides + base config). The returned dict maps
    each ``spec`` key to its sampled scalar value with the appropriate Python type.
    """
    resolved_values = dict(resolved_values or {})
    out: Dict[str, Any] = {}
    for name, entry in spec.items():
        dist = entry["dist"]
        if dist == "uniform":
            out[name] = trial.suggest_float(name, float(entry["low"]), float(entry["high"]))
        elif dist == "loguniform":
            out[name] = trial.suggest_float(name, float(entry["low"]), float(entry["high"]), log=True)
        elif dist == "int":
            out[name] = trial.suggest_int(
                name,
                int(entry["low"]),
                int(entry["high"]),
                step=int(entry.get("step", 1)),
                log=bool(entry.get("log", False)),
            )
        elif dist == "int_log":
            out[name] = trial.suggest_int(name, int(entry["low"]), int(entry["high"]), log=True)
        elif dist == "categorical":
            out[name] = trial.suggest_categorical(name, list(entry["choices"]))
        elif dist == "frac_of":
            frac = trial.suggest_float(name, float(entry["low"]), float(entry["high"]))
            base_key = entry["base"]
            base_val = resolved_values.get(base_key)
            if base_val is None:
                raise SearchSpaceError(
                    f"frac_of base={base_key!r} not resolvable at sample time "
                    f"(resolved_values has keys: {sorted(resolved_values)})"
                )
            raw = frac * float(base_val)
            out[name] = int(round(raw)) if base_key in _INT_TYPED_KEYS else raw
        else:
            raise SearchSpaceError(f"unknown dist={dist!r} for {name!r}")
    return out
