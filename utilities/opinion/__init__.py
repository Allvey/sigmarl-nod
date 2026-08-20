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
from utilities.opinion.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    load_opinion_checkpoint,
    save_opinion_checkpoint,
)
from utilities.opinion.collector import (
    CollectorStepOutput,
    OpinionStatefulCollector,
    apply_opinion_resets,
    decay_dense_opinions,
)
from utilities.opinion.diagnostics import OpinionDiagnostics
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
from utilities.opinion.ppo_loss import (
    OpinionCentralizedCritic,
    OpinionSequencePPOLoss,
    SequencePPOOutput,
)
from utilities.opinion.sequence_buffer import OpinionSequenceBuffer, SequenceChunk

__all__ = (
    "DEFAULT_OPINION_CONFIG_PATH",
    "OPINION_CONFIG_FIELDS",
    "OPINION_STAGES",
    "LoadedOpinionExperimentConfig",
    "OpinionConfig",
    "ConflictGraph",
    "ConflictGraphOutput",
    "CHECKPOINT_SCHEMA_VERSION",
    "CollectorStepOutput",
    "BaseActorOutput",
    "BaseGaussianActor",
    "OpinionDynamics",
    "OpinionEvidenceNet",
    "OpinionAugmentedPolicyCore",
    "OpinionCentralizedCritic",
    "OpinionDiagnostics",
    "OpinionPolicyCoreOutput",
    "OpinionPolicyOutput",
    "OpinionResidual",
    "OpinionSequenceBuffer",
    "OpinionSequencePPOLoss",
    "OpinionStatefulCollector",
    "OpinionTanhNormalPolicy",
    "EvidenceOutput",
    "ResidualOutput",
    "SequenceChunk",
    "SequencePPOOutput",
    "PAIR_FEATURE_NAMES",
    "PairInteractionEncoder",
    "PairInteractionFeatures",
    "apply_opinion_resets",
    "decay_dense_opinions",
    "gather_candidate_opinions",
    "load_opinion_experiment_config",
    "load_opinion_checkpoint",
    "save_opinion_checkpoint",
    "scatter_candidate_opinions",
    "swap_roles",
)
