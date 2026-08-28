"""Registry + FQN dispatch tests."""
from __future__ import annotations

import pytest


def test_builtin_predictor_names_present():
    from rl_matdesign.registry import PREDICTORS
    expected = {"rf_magpie", "ooh", "dummy"}
    assert expected.issubset(set(PREDICTORS))


def test_builtin_constraint_names_present():
    from rl_matdesign.registry import CONSTRAINTS
    expected = {"phase_pattern", "ooh_phase", "smact_charge", "last_step_element"}
    assert expected.issubset(set(CONSTRAINTS))


def test_resolve_dummy_predictor_runs():
    from rl_matdesign.registry import resolve_predictor
    p = resolve_predictor("dummy", {})
    m, s = p.predict({"Fe": 0.5, "Ni": 0.5})
    assert isinstance(m, float) and isinstance(s, float)
    assert s >= 0.0


def test_resolve_unknown_predictor_lists_builtins():
    from rl_matdesign.registry import resolve_predictor
    with pytest.raises(ValueError) as info:
        resolve_predictor("definitely_not_a_real_kind", {})
    msg = str(info.value)
    # Helpful error must surface the registered short names.
    assert "rf_magpie" in msg and "ooh" in msg


def test_fqn_dispatch_loads_class(tmp_path, monkeypatch):
    # Drop a tiny module on PYTHONPATH and load it via FQN.
    mod = tmp_path / "fake_pred.py"
    mod.write_text(
        "class FakePredictor:\n"
        "    def __init__(self, cfg, *, seed=None):\n"
        "        self.cfg, self.seed = cfg, seed\n"
        "    def predict(self, comp):\n"
        "        return 0.42, 0.01\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    from rl_matdesign.registry import resolve_predictor
    p = resolve_predictor("fake_pred:FakePredictor", {"foo": 1}, seed=7)
    assert p.seed == 7
    assert p.cfg == {"foo": 1}
    m, s = p.predict({"X": 1.0})
    assert m == 0.42 and s == 0.01


def test_fqn_missing_module_helpful_error():
    from rl_matdesign.registry import resolve_predictor
    with pytest.raises(ModuleNotFoundError) as info:
        resolve_predictor("definitely_not_a_real_module:Foo", {})
    assert "definitely_not_a_real_module" in str(info.value)


def test_fqn_bad_form_treated_as_short_name():
    # Without ':' it's a short name; should fail with the helpful built-ins list.
    from rl_matdesign.registry import resolve_predictor
    with pytest.raises(ValueError) as info:
        resolve_predictor("not_a_real_short_name_no_colon", {})
    assert "Built-ins" in str(info.value)


def test_resolve_constraint_none_returns_none():
    from rl_matdesign.registry import resolve_constraint
    assert resolve_constraint(None, {}) is None


def test_resolve_unknown_constraint_lists_builtins():
    from rl_matdesign.registry import resolve_constraint
    with pytest.raises(ValueError) as info:
        resolve_constraint("not_a_real_constraint", {})
    assert "phase_pattern" in str(info.value)


def test_mgtransformer_is_registered_under_a_short_name():
    # Config authors write `predictor: mgtransformer`, not the full FQN.
    from rl_matdesign.registry import PREDICTORS

    assert "mgtransformer" in PREDICTORS


def test_mgtransformer_resolves_and_reports_the_missing_model_key():
    # Resolution goes through the registry (not importlib), and the leaf's own
    # required-key validation is reached.
    import pytest

    from rl_matdesign.registry import resolve_predictor

    with pytest.raises(ValueError) as info:
        resolve_predictor("mgtransformer", {"mgt_repo": ".", "mgt_python": "python"})
    assert "model" in str(info.value)
