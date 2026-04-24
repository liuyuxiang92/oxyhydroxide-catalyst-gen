"""Structure substitution utilities for generating alloyed supercells.

Supports two modes:
- ``"random"``:  Generate *n_configs* random solid-solution structures by
  randomly assigning elements to sites according to composition fractions.
  Fast; suitable for DPA evaluation during RL training.
  This is the Random Solid Solution Approximation (RSA).

- ``"sqs"``:  Generate a single Special Quasi-random Structure (SQS) via
  ``sqsgenerator``.  SQS minimises the deviation between pair/triplet
  correlation functions of the supercell and those of a perfectly random
  infinite alloy.  More physically rigorous; recommended for DFT validation
  and paper figures.  Requires: ``pip install sqsgenerator``.

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
    mode: str = "random",
    n_configs: int = 5,
    sqs_iterations: int = 1000,
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
    mode:
        ``"random"`` (default) or ``"sqs"``.
    n_configs:
        Number of random structures to generate (only used when ``mode="random"``).
    sqs_iterations:
        Number of optimisation iterations for SQS generation
        (only used when ``mode="sqs"``).
    rng:
        NumPy random generator for reproducibility.  If ``None`` a new default
        generator is created.

    Returns
    -------
    List of ASE Atoms objects (length = *n_configs* for random mode, 1 for SQS).
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

    if mode == "random":
        return _random_configs(template, site_indices, counts, n_configs, rng)
    elif mode == "sqs":
        return _sqs_config(template, site_indices, counts, sqs_iterations)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'random' or 'sqs'.")


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


def _sqs_config(
    template: "ase.Atoms",
    site_indices: List[int],
    counts: Dict[str, int],
    sqs_iterations: int,
) -> List["ase.Atoms"]:
    """Generate a single SQS structure via sqsgenerator.

    Requires: ``pip install sqsgenerator``
    """
    try:
        import sqsgenerator
    except ImportError as exc:
        raise ImportError(
            "SQS mode requires sqsgenerator: pip install sqsgenerator"
        ) from exc

    # Extract the sub-lattice as a separate Atoms object for sqsgenerator.
    from ase import Atoms as AseAtoms

    sublattice = template[site_indices]

    # sqsgenerator expects composition as {symbol: count} dict.
    result = sqsgenerator.run_sqs_iterations(
        structure=sublattice,
        target_concentrations={k: v / len(site_indices) for k, v in counts.items()},
        iterations=sqs_iterations,
    )

    # Merge SQS sub-lattice back into the full template.
    best_sqs = result.get_best_structure()
    atoms = template.copy()
    symbols = atoms.get_chemical_symbols()
    sqs_symbols = best_sqs.get_chemical_symbols()
    for idx, sym in zip(site_indices, sqs_symbols):
        symbols[idx] = sym
    atoms.set_chemical_symbols(symbols)
    return [atoms]
