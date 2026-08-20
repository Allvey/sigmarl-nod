"""Opinion Dynamics + MARL components."""

from utilities.opinion.config import (
    DEFAULT_OPINION_CONFIG_PATH,
    OPINION_CONFIG_FIELDS,
    OPINION_STAGES,
    LoadedOpinionExperimentConfig,
    OpinionConfig,
    load_opinion_experiment_config,
)
from utilities.opinion.conflict_graph import (
    PAIR_FEATURE_NAMES,
    ConflictGraph,
    ConflictGraphOutput,
)
from utilities.opinion.dynamics import (
    OpinionDynamics,
    gather_candidate_opinions,
    scatter_candidate_opinions,
)
from utilities.opinion.evidence_net import EvidenceOutput, OpinionEvidenceNet, swap_roles
from utilities.opinion.policy import (
    BaseActorOutput,
    BaseGaussianActor,
    OpinionAugmentedPolicyCore,
    OpinionPolicyCoreOutput,
    OpinionPolicyOutput,
    OpinionTanhNormalPolicy,
    PairInteractionEncoder,
    PairInteractionFeatures,
)
from utilities.opinion.residual import OpinionResidual, ResidualOutput

__all__ = (
    "DEFAULT_OPINION_CONFIG_PATH",
    "OPINION_CONFIG_FIELDS",
    "OPINION_STAGES",
    "LoadedOpinionExperimentConfig",
    "OpinionConfig",
    "ConflictGraph",
    "ConflictGraphOutput",
    "BaseActorOutput",
    "BaseGaussianActor",
    "OpinionDynamics",
    "OpinionEvidenceNet",
    "OpinionAugmentedPolicyCore",
    "OpinionPolicyCoreOutput",
    "OpinionPolicyOutput",
    "OpinionResidual",
    "OpinionTanhNormalPolicy",
    "EvidenceOutput",
    "ResidualOutput",
    "PAIR_FEATURE_NAMES",
    "PairInteractionEncoder",
    "PairInteractionFeatures",
    "gather_candidate_opinions",
    "load_opinion_experiment_config",
    "scatter_candidate_opinions",
    "swap_roles",
)
