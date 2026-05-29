"""HEAPropertyPredictor — DP-ensemble reward oracle for High-Entropy Alloys.

A thin subclass of :class:`~rl_matdesign.predictors.dp_structure.DPStructurePredictor`
that defaults ``site_symbol`` to ``"X"`` (the FCC/BCC placeholder).

All behaviour, parameters, and reward sign convention live on the base class.
"""
from __future__ import annotations

from .dp_structure import DPStructurePredictor


class HEAPropertyPredictor(DPStructurePredictor):
    DEFAULT_SITE_SYMBOL = "X"
