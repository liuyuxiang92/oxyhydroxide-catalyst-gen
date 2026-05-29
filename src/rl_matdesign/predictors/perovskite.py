"""PerovskitePropertyPredictor — DP-ensemble reward oracle for perovskite oxides.

A thin subclass of :class:`~rl_matdesign.predictors.dp_structure.DPStructurePredictor`
that defaults ``site_symbol`` to ``"Fe"`` (the B-site placeholder in a
``LaBO₃``-style template).

All behaviour, parameters, and reward sign convention live on the base class.
"""
from __future__ import annotations

from .dp_structure import DPStructurePredictor


class PerovskitePropertyPredictor(DPStructurePredictor):
    DEFAULT_SITE_SYMBOL = "Fe"
