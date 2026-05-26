"""Structure substitution utilities for generating alloyed supercells.

Generates *n_configs* random solid-solution structures by randomly assigning
elements to sites according to composition fractions (Random Solid Solution
Approximation).  Fast and suitable for DPA evaluation during RL training.

The logic is extracted from the OOH predictor's
``_choose_counts_from_fractions`` and ``_build_dp_inputs_for_one_doped_slab``
methods so that HEA and perovskite predictors share a single implementation.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def substitute_sites(
    template_poscar: str,
    composition: Dict[str, float],
    site_symbol: str = "X",
    n_configs: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> List["ase.Atoms"]:
    """Generate ASE Atoms objects with template sites substituted per *composition*.

    Parameters
    ----------
    template_poscar:
        Path to a POSCAR/CONTCAR file.  The sites occupied by *site_symbol*
        will be replaced.  For HEA use a placeholder element (e.g. ``"X"`` or
        ``"Cu"``).  For perovskite B-site use the placeholder B-site element.
    composition:
        Dict ``{element: fraction}``.  Fractions must sum to 1.0.
    site_symbol:
        Element symbol in the template that marks sites to be substituted.
    n_configs:
        Number of random structures to generate.
    rng:
        NumPy random generator for reproducibility.  If ``None`` a new default
        generator is created.

    Returns
    -------
    List of *n_configs* ASE Atoms objects.
    """
    try:
        from ase.io import read as ase_read
    except ImportError as exc:
        raise ImportError("substitute_sites requires ASE: pip install ase") from exc

    if rng is None:
        rng = np.random.default_rng()

    template = ase_read(template_poscar)

    # Identify target sites.
    site_indices = [i for i, s in enumerate(template.get_chemical_symbols()) if s == site_symbol]
    if not site_indices:
        raise ValueError(
            f"No sites with symbol '{site_symbol}' found in {template_poscar}. "
            "Check that site_symbol matches the placeholder element in the template."
        )

    n_sites = len(site_indices)
    counts = _fractions_to_counts(composition, n_sites)
    return _random_configs(template, site_indices, counts, n_configs, rng)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fractions_to_counts(composition: Dict[str, float], n_sites: int) -> Dict[str, int]:
    """Allocate integer atom counts from fractional composition.

    Uses a largest-remainder method to guarantee sum == n_sites exactly.
    """
    elements = list(composition.keys())
    fracs = [composition[e] for e in elements]

    raw = [f * n_sites for f in fracs]
    floors = [int(x) for x in raw]
    remainders = [(raw[i] - floors[i], i) for i in range(len(floors))]

    deficit = n_sites - sum(floors)
    if deficit < 0 or deficit > len(elements):
        raise ValueError(f"Cannot allocate {n_sites} sites from fractions {composition}.")

    # Assign remaining sites to elements with largest remainders.
    remainders.sort(reverse=True)
    counts_list = floors[:]
    for k in range(deficit):
        counts_list[remainders[k][1]] += 1

    return {elements[i]: counts_list[i] for i in range(len(elements))}


def _random_configs(
    template: "ase.Atoms",
    site_indices: List[int],
    counts: Dict[str, int],
    n_configs: int,
    rng: np.random.Generator,
) -> List["ase.Atoms"]:
    """Generate *n_configs* random solid-solution structures."""
    # Build the element list in the order they will be assigned.
    elem_list: List[str] = []
    for elem, count in counts.items():
        elem_list.extend([elem] * count)

    configs = []
    for _ in range(n_configs):
        atoms = template.copy()
        shuffled = rng.permutation(elem_list).tolist()
        symbols = atoms.get_chemical_symbols()
        for idx, elem in zip(site_indices, shuffled):
            symbols[idx] = elem
        atoms.set_chemical_symbols(symbols)
        configs.append(atoms)
    return configs
