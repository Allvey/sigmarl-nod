"""Proximal Saturating Bifurcation MARL.

P0 contains only the Base-MAPPO passthrough and experiment contracts.  Later
stages add local pair perception, proximal state, and trainable PSB modules
without changing the P0 compatibility guarantee.
"""

from utilities.psb_marl.config import (
    PSBConfigError,
    PSBExperimentConfig,
    load_psb_experiment,
)

__all__ = [
    "PSBConfigError",
    "PSBExperimentConfig",
    "load_psb_experiment",
]
