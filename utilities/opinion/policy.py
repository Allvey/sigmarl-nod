"""Single-step Opinion-MARL policy composition.

This module is intentionally stateless with respect to temporal opinion state.
The caller supplies ``z_prev`` and receives ``z_next``; the M6 collector will be
the sole owner of the cross-step dense state.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import torch
from torch import Tensor, nn
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import TanhNormal

from utilities.opinion.config import OPINION_STAGES, OpinionConfig
from utilities.opinion.dynamics import OpinionDynamics
from utilities.opinion.evidence_net import OpinionEvidenceNet
from utilities.opinion.residual import OpinionResidual


class PairInteractionFeatures(NamedTuple):
    ego_features: Tensor
    neighbor_features: Tensor
    symmetric_context: Tensor
    antisymmetric_context: Tensor


class BaseActorOutput(NamedTuple):
    loc: Tensor
    scale: Tensor


class OpinionPolicyCoreOutput(NamedTuple):
    base_loc: Tensor
    final_loc: Tensor
    scale: Tensor
    raw_b: Tensor
    b: Tensor
    z_next: Tensor
    q: Tensor
    normalized_weights: Tensor
    aggregate: Tensor
    residual: Tensor


class OpinionPolicyOutput(NamedTuple):
    action: Tensor
    log_prob: Tensor
    core: OpinionPolicyCoreOutput


class PairInteractionEncoder(nn.Module):
    """Give every M4 pair feature an explicit role for EvidenceNet.

    The split preserves all 12 current-physics features. Relative ego-frame
    quantities form the directed context. Distance/CPA quantities are symmetric,
    while the two absolute speed magnitudes are assigned to the individual roles.
    """

    pair_feature_dim = 12
    individual_feature_dim = 1
    symmetric_context_dim = 5
    antisymmetric_context_dim = 5

    _SYMMETRIC_INDICES = (4, 5, 6, 7, 9)
    _ANTISYMMETRIC_INDICES = (0, 1, 2, 3, 8)
    _EGO_SPEED_INDEX = 10
    _NEIGHBOR_SPEED_INDEX = 11

    def forward(self, pair_features: Tensor) -> PairInteractionFeatures:
        if not torch.is_tensor(pair_features):
            raise TypeError("pair_features must be a tensor")
        if pair_features.ndim < 2 or pair_features.shape[-1] != self.pair_feature_dim:
            raise ValueError("pair_features must end with the M4 feature dimension 12")
        if not pair_features.is_floating_point():
            raise TypeError("pair_features must use a floating dtype")
        return PairInteractionFeatures(
            ego_features=pair_features[..., self._EGO_SPEED_INDEX : self._EGO_SPEED_INDEX + 1],
            neighbor_features=pair_features[
                ..., self._NEIGHBOR_SPEED_INDEX : self._NEIGHBOR_SPEED_INDEX + 1
            ],
            symmetric_context=pair_features[..., list(self._SYMMETRIC_INDICES)],
            antisymmetric_context=pair_features[
                ..., list(self._ANTISYMMETRIC_INDICES)
            ],
        )


class BaseGaussianActor(nn.Module):
    """Shared decentralized MLP producing loc/scale for each agent."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        depth: int = 2,
    ) -> None:
        super().__init__()
        for name, value in {
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
            "depth": depth,
        }.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int")
        if action_dim < 2:
            raise ValueError("action_dim must include speed and steering channels")

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        layers = []
        input_dim = observation_dim
        for _ in range(depth):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.Tanh()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 2 * action_dim))
        self.network = nn.Sequential(*layers)
        self.extractor = NormalParamExtractor()

    def forward(self, observation: Tensor) -> BaseActorOutput:
        if not torch.is_tensor(observation):
            raise TypeError("observation must be a tensor")
        if observation.ndim < 2 or observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"observation must end with dimension {self.observation_dim}"
            )
        if not observation.is_floating_point() or not torch.isfinite(observation).all():
            raise ValueError("observation must be a finite floating tensor")
        loc, scale = self.extractor(self.network(observation))
        return BaseActorOutput(loc=loc, scale=scale)


class OpinionAugmentedPolicyCore(nn.Module):
    """Compose the base actor and fixed opinion path before sampling."""

    speed_action_index = 0

    def __init__(
        self,
        *,
        base_actor: nn.Module,
        interaction_encoder: PairInteractionEncoder,
        evidence_net: nn.Module,
        dynamics: OpinionDynamics,
        residual_module: OpinionResidual,
        stage: str,
        dt: float,
    ) -> None:
        super().__init__()
        if stage not in OPINION_STAGES:
            raise ValueError(f"stage must be one of {OPINION_STAGES}, got {stage!r}")
        if isinstance(dt, bool) or not isinstance(dt, (int, float)):
            raise ValueError("dt must be numeric")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.base_actor = base_actor
        self.interaction_encoder = interaction_encoder
        self.evidence_net = evidence_net
        self.dynamics = dynamics
        self.residual_module = residual_module
        self.stage = stage
        self.dt = float(dt)

    @classmethod
    def from_config(
        cls,
        *,
        observation_dim: int,
        action_dim: int,
        config: OpinionConfig,
        dt: float,
        actor_hidden_dim: int = 256,
        actor_depth: int = 2,
    ) -> "OpinionAugmentedPolicyCore":
        if not isinstance(config, OpinionConfig):
            raise TypeError("config must be an OpinionConfig")
        interaction_encoder = PairInteractionEncoder()
        return cls(
            base_actor=BaseGaussianActor(
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dim=actor_hidden_dim,
                depth=actor_depth,
            ),
            interaction_encoder=interaction_encoder,
            evidence_net=OpinionEvidenceNet(
                individual_feature_dim=interaction_encoder.individual_feature_dim,
                symmetric_context_dim=interaction_encoder.symmetric_context_dim,
                antisymmetric_context_dim=interaction_encoder.antisymmetric_context_dim,
                hidden_dim=config.evidence_hidden_dim,
                num_layers=config.evidence_num_layers,
                b_max=config.b_max,
                b_temperature=config.b_temperature,
            ),
            dynamics=OpinionDynamics(
                kappa=config.kappa,
                nu=config.nu,
                alpha=config.alpha,
                eta=config.eta,
                z_clip=config.z_clip,
                n_substeps=config.n_substeps,
            ),
            residual_module=OpinionResidual(z0=config.z0),
            stage=config.stage,
            dt=dt,
        )

    def _validate_pair_contract(
        self,
        pair_features: Tensor,
        urgency: Tensor,
        confidence: Tensor,
        pair_mask: Tensor,
        z_prev: Tensor,
    ) -> None:
        if z_prev.ndim < 1:
            raise ValueError("z_prev must include a candidate dimension")
        expected_pair_shape = z_prev.shape + (self.interaction_encoder.pair_feature_dim,)
        if pair_features.shape != expected_pair_shape:
            raise ValueError(
                f"pair_features must have shape {expected_pair_shape}; "
                f"got {tuple(pair_features.shape)}"
            )
        for name, tensor in (("urgency", urgency), ("confidence", confidence)):
            if tensor.shape != z_prev.shape:
                raise ValueError(f"{name} must have shape {tuple(z_prev.shape)}")
        if pair_mask.shape != z_prev.shape or pair_mask.dtype is not torch.bool:
            raise ValueError("pair_mask must be bool and match z_prev")

    def forward(
        self,
        observation: Tensor,
        pair_features: Tensor,
        urgency: Tensor,
        confidence: Tensor,
        pair_mask: Tensor,
        z_prev: Tensor,
        *,
        residual_scale: float,
        direction: Optional[Tensor] = None,
    ) -> OpinionPolicyCoreOutput:
        if isinstance(residual_scale, bool) or not isinstance(
            residual_scale, (int, float)
        ):
            raise ValueError("residual_scale must be numeric")
        if not math.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError("residual_scale must be finite and non-negative")
        self._validate_pair_contract(
            pair_features, urgency, confidence, pair_mask, z_prev
        )
        base = self.base_actor(observation)
        if base.loc.shape != base.scale.shape or base.loc.shape[:-1] != z_prev.shape[:-1]:
            raise ValueError("base loc/scale must match the pair agent batch shape")
        if base.loc.shape[-1] < 2:
            raise ValueError("base actor output must include speed and steering")
        if not torch.isfinite(base.loc).all() or not torch.isfinite(base.scale).all():
            raise ValueError("base loc and scale must be finite")
        if (base.scale <= 0).any():
            raise ValueError("base scale must be positive")

        if self.stage == "base":
            raw_b = torch.zeros_like(z_prev)
            b = torch.zeros_like(z_prev)
            z_next = torch.zeros_like(z_prev)
            q = torch.zeros_like(z_prev)
            normalized_weights = torch.zeros_like(z_prev)
            aggregate = torch.zeros_like(z_prev[..., 0])
            residual = torch.zeros_like(aggregate)
        else:
            interactions = self.interaction_encoder(pair_features)
            evidence = self.evidence_net(
                interactions.ego_features,
                interactions.neighbor_features,
                interactions.symmetric_context,
                interactions.antisymmetric_context,
                urgency,
                confidence,
                pair_mask,
            )
            raw_b = torch.where(pair_mask, evidence.raw_b, torch.zeros_like(z_prev))
            b = torch.where(pair_mask, evidence.b, torch.zeros_like(z_prev))
            urgency_safe = torch.where(
                pair_mask, urgency, torch.zeros_like(urgency)
            )
            z_next = self.dynamics(
                z_prev,
                b,
                urgency_safe,
                pair_mask,
                dt=self.dt,
            )
            if direction is None:
                direction = torch.ones_like(z_prev)
            residual_output = self.residual_module(
                z_next,
                urgency_safe,
                direction,
                pair_mask,
                residual_scale=residual_scale,
            )
            q = residual_output.q
            normalized_weights = residual_output.normalized_weights
            aggregate = residual_output.aggregate
            residual = residual_output.residual

        final_loc = base.loc.clone()
        final_loc[..., self.speed_action_index] = (
            final_loc[..., self.speed_action_index] + residual
        )
        return OpinionPolicyCoreOutput(
            base_loc=base.loc,
            final_loc=final_loc,
            scale=base.scale,
            raw_b=raw_b,
            b=b,
            z_next=z_next,
            q=q,
            normalized_weights=normalized_weights,
            aggregate=aggregate,
            residual=residual,
        )


class OpinionTanhNormalPolicy(nn.Module):
    """Sample or re-evaluate actions from the final opinion-adjusted loc."""

    def __init__(
        self,
        *,
        core: OpinionAugmentedPolicyCore,
        action_low: Tensor,
        action_high: Tensor,
    ) -> None:
        super().__init__()
        if not torch.is_tensor(action_low) or not torch.is_tensor(action_high):
            raise TypeError("action bounds must be tensors")
        if action_low.ndim < 1 or action_low.shape != action_high.shape:
            raise ValueError("action bounds must have equal non-scalar shapes")
        if not action_low.is_floating_point() or action_low.dtype != action_high.dtype:
            raise TypeError("action bounds must share a floating dtype")
        if not torch.isfinite(action_low).all() or not torch.isfinite(action_high).all():
            raise ValueError("action bounds must be finite")
        if not (action_low < action_high).all():
            raise ValueError("every action_low must be below action_high")
        self.core = core
        self.register_buffer("action_low", action_low.detach().clone())
        self.register_buffer("action_high", action_high.detach().clone())

    def forward(
        self,
        observation: Tensor,
        pair_features: Tensor,
        urgency: Tensor,
        confidence: Tensor,
        pair_mask: Tensor,
        z_prev: Tensor,
        *,
        residual_scale: float,
        direction: Optional[Tensor] = None,
        action: Optional[Tensor] = None,
    ) -> OpinionPolicyOutput:
        core_output = self.core(
            observation,
            pair_features,
            urgency,
            confidence,
            pair_mask,
            z_prev,
            residual_scale=residual_scale,
            direction=direction,
        )
        if core_output.final_loc.shape[-1] != self.action_low.shape[-1]:
            raise ValueError("policy action dimension does not match action bounds")
        try:
            broadcast_shape = torch.broadcast_shapes(
                core_output.final_loc.shape, self.action_low.shape
            )
        except RuntimeError as error:
            raise ValueError(
                "action bounds must broadcast to the policy action shape"
            ) from error
        if broadcast_shape != core_output.final_loc.shape:
            raise ValueError("action bounds must not expand the policy action shape")
        distribution = TanhNormal(
            core_output.final_loc,
            core_output.scale,
            min=self.action_low,
            max=self.action_high,
        )
        if action is None:
            action = distribution.rsample()
        elif action.shape != core_output.final_loc.shape:
            raise ValueError("supplied action must match the policy action shape")
        if not torch.isfinite(action).all():
            raise ValueError("action must be finite")
        log_prob = distribution.log_prob(action)
        if not torch.isfinite(log_prob).all():
            raise RuntimeError("Opinion TanhNormal produced a non-finite log_prob")
        return OpinionPolicyOutput(
            action=action,
            log_prob=log_prob,
            core=core_output,
        )
