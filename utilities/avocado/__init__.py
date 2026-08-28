"""Strict A0-A2 AVOCADO reference implementation.

This package intentionally has no dependency on the Opinion-MARL modules.  It
implements the holonomic, disc-shaped, single-integrator formulation used by
AVOCADO before any SigmaRL bicycle-model or learning extensions are applied.
"""

from utilities.avocado.controller import AVOCADOController, fixed_orca_actions
from utilities.avocado.bicycle import (
    BicycleActionResult,
    BicycleAdapterParameters,
    constrain_velocity_to_path,
    path_velocity_cone_constraints,
    reference_path_preferred_velocity,
    stanley_path_preferred_velocity,
    vector_velocity_to_bicycle_action,
    wrap_to_pi,
)
from utilities.avocado.road_safety import (
    TTCSafetyShieldResult,
    apply_ttc_braking_shield,
    bicycle_action_velocity,
)
from utilities.avocado.core import (
    AVOCADOParameters,
    HalfPlane,
    ProjectionResult,
    VelocityObstacleCorrection,
    attention_euler_step,
    attention_reference_step,
    build_oca_half_plane,
    collision_time,
    finite_velocity_obstacle_correction,
    goal_preferred_velocity,
    opinion_euler_step,
    opinion_to_cooperation,
    projection_estimator,
    solve_closest_admissible_velocity,
)

__all__ = [
    "AVOCADOController",
    "AVOCADOParameters",
    "BicycleActionResult",
    "BicycleAdapterParameters",
    "HalfPlane",
    "ProjectionResult",
    "VelocityObstacleCorrection",
    "attention_euler_step",
    "attention_reference_step",
    "build_oca_half_plane",
    "collision_time",
    "constrain_velocity_to_path",
    "path_velocity_cone_constraints",
    "finite_velocity_obstacle_correction",
    "fixed_orca_actions",
    "goal_preferred_velocity",
    "opinion_euler_step",
    "opinion_to_cooperation",
    "projection_estimator",
    "reference_path_preferred_velocity",
    "stanley_path_preferred_velocity",
    "solve_closest_admissible_velocity",
    "vector_velocity_to_bicycle_action",
    "TTCSafetyShieldResult",
    "apply_ttc_braking_shield",
    "bicycle_action_velocity",
    "wrap_to_pi",
]
