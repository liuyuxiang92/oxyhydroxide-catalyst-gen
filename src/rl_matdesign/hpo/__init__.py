"""Hyperparameter-optimization helpers for the rl-matdesign Optuna driver.

Public surface kept small so `scripts/hpo.py` reads like a thin orchestrator.
"""

from .search_space import sample_from_search_space, validate_search_space
from .metric import top_k_mean_from_csv
from .runner import run_trial_subprocess, TrialRunError

__all__ = [
    "sample_from_search_space",
    "validate_search_space",
    "top_k_mean_from_csv",
    "run_trial_subprocess",
    "TrialRunError",
]
