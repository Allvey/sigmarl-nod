"""Independent Opinion Dynamics + MARL components.

M2 intentionally exposes configuration only. Mathematical and stateful modules
are added in later milestones and must not be imported by the Base path.
"""

from utilities.opinion.config import (
    LoadedOpinionExperiment,
    OpinionConfigError,
    OpinionExperimentConfig,
    load_opinion_experiment,
)

__all__ = [
    "LoadedOpinionExperiment",
    "OpinionConfigError",
    "OpinionExperimentConfig",
    "load_opinion_experiment",
]
