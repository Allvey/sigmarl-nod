"""A1: VMAS environment matching AVOCADO's original robot model.

Unlike SigmaRL's road-traffic scenario, this environment uses disc agents and
accepts a two-dimensional velocity command.  The custom dynamics makes the
VMAS update exactly ``p[k+1] = p[k] + dt * v[k]`` for one simulation substep.
Physical collision response is disabled so a failed avoidance rollout remains
an unmodified single-integrator experiment; collisions are measured
geometrically by the A2 benchmark.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from vmas.simulator.core import Agent, Sphere, World
from vmas.simulator.dynamics.common import Dynamics
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class DirectVelocityDynamics(Dynamics):
    """Convert a desired velocity into the exact VMAS single-integrator step."""

    def __init__(self, world: World, maximum_speed: float) -> None:
        super().__init__()
        self.world = world
        self.maximum_speed = float(maximum_speed)

    @property
    def needed_action_size(self) -> int:
        return 2

    def process_action(self) -> None:
        desired = self.agent.action.u[:, :2]
        norm = torch.linalg.vector_norm(desired, dim=-1, keepdim=True)
        scale = torch.clamp(
            self.maximum_speed / norm.clamp_min(1e-8), max=1.0
        )
        desired = desired * scale
        self.agent.state.force = (
            self.agent.mass * (desired - self.agent.state.vel) / self.world.dt
        )


class Scenario(BaseScenario):
    """Independent holonomic-disc testbed for strict AVOCADO validation."""

    SUPPORTED_LAYOUTS = {
        "head_on_noncooperative",
        "head_on_cooperative",
        "circle_cooperative",
        "crossing_mixed",
    }

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.layout = str(kwargs.get("layout", "head_on_noncooperative"))
        if self.layout not in self.SUPPORTED_LAYOUTS:
            raise ValueError(
                f"Unknown AVOCADO layout {self.layout!r}; expected one of "
                f"{sorted(self.SUPPORTED_LAYOUTS)}."
            )
        self.n_agents = int(kwargs.get("n_agents", 2))
        self.controlled_agents = int(kwargs.get("controlled_agents", 1))
        self.dt = float(kwargs.get("dt", 0.05))
        self.robot_radius = float(kwargs.get("robot_radius", 0.2))
        self.agent_radius = float(kwargs.get("agent_radius", 0.2))
        self.avoidance_radius_scale = float(
            kwargs.get("avoidance_radius_scale", 1.1)
        )
        self.robot_max_speed = float(kwargs.get("robot_max_speed", 1.0))
        self.agent_max_speed = float(kwargs.get("agent_max_speed", 0.75))
        self.goal_tolerance = float(kwargs.get("goal_tolerance", 0.1))
        self.position_jitter = float(kwargs.get("position_jitter", 0.0))
        self.seed = int(kwargs.get("layout_seed", 0))
        if self.n_agents < 2:
            raise ValueError("The AVOCADO environment requires at least two agents.")
        if not 1 <= self.controlled_agents <= self.n_agents:
            raise ValueError("controlled_agents must be in [1, n_agents].")
        if self.layout in {"head_on_noncooperative", "head_on_cooperative"}:
            if self.n_agents != 2:
                raise ValueError("Head-on layouts require n_agents=2.")
        if self.layout == "head_on_noncooperative" and self.controlled_agents != 1:
            raise ValueError("head_on_noncooperative requires controlled_agents=1.")
        if self.layout in {"head_on_cooperative", "circle_cooperative"}:
            if self.controlled_agents != self.n_agents:
                raise ValueError(f"{self.layout} requires all agents to be controlled.")
        if (
            self.dt <= 0
            or self.robot_radius <= 0
            or self.agent_radius <= 0
            or self.avoidance_radius_scale < 1.0
        ):
            raise ValueError(
                "dt and physical radii must be positive, and "
                "avoidance_radius_scale must be at least one."
            )

        world = World(
            batch_dim,
            device,
            dt=self.dt,
            substeps=1,
            drag=0.0,
            collision_force=0.0,
            dim_c=0,
        )
        self.controlled_mask = torch.arange(
            self.n_agents, device=device
        ) < self.controlled_agents
        self.radii = torch.where(
            self.controlled_mask,
            torch.full((self.n_agents,), self.robot_radius, device=device),
            torch.full((self.n_agents,), self.agent_radius, device=device),
        )
        self.security_radii = self.radii * self.avoidance_radius_scale
        self.maximum_speeds = torch.where(
            self.controlled_mask,
            torch.full((self.n_agents,), self.robot_max_speed, device=device),
            torch.full((self.n_agents,), self.agent_max_speed, device=device),
        )

        for index in range(self.n_agents):
            controlled = bool(self.controlled_mask[index])
            maximum_speed = float(self.maximum_speeds[index])
            world.add_agent(
                Agent(
                    name=f"{'robot' if controlled else 'agent'}_{index}",
                    shape=Sphere(radius=float(self.radii[index])),
                    movable=True,
                    rotatable=False,
                    collide=False,
                    color=Color.BLUE if controlled else Color.RED,
                    max_speed=maximum_speed,
                    u_range=maximum_speed,
                    render_action=True,
                    dynamics=DirectVelocityDynamics(world, maximum_speed),
                )
            )
        self.goals = torch.zeros(
            batch_dim, self.n_agents, 2, device=device, dtype=torch.float32
        )
        self.initial_positions = torch.zeros_like(self.goals)
        self.viewer_size = (900, 900)
        self.viewer_zoom = 1.0
        return world

    def _layout_positions_and_goals(self) -> Tuple[Tensor, Tensor]:
        device = self.world.device
        dtype = torch.float32
        if self.layout in {"head_on_noncooperative", "head_on_cooperative"}:
            positions = torch.tensor(
                [[-1.25, 0.0], [1.25, 0.0]], device=device, dtype=dtype
            )
            goals = torch.flip(positions, dims=(0,))
            return positions, goals

        if self.layout == "circle_cooperative":
            radius = max(
                2.5,
                2.3 * self.n_agents * self.robot_radius / math.pi,
            )
            angles = torch.arange(self.n_agents, device=device, dtype=dtype)
            angles = 2.0 * math.pi * angles / self.n_agents
            positions = radius * torch.stack(
                (torch.cos(angles), torch.sin(angles)), dim=-1
            )
            return positions, -positions

        # Controlled robots move horizontally while non-cooperative agents form
        # a perpendicular flow, mirroring the paper's crossing setting.
        if self.n_agents < 4 or self.controlled_agents >= self.n_agents:
            raise ValueError(
                "crossing_mixed requires at least four entities and both roles."
            )
        positions = []
        goals = []
        robot_count = self.controlled_agents
        agent_count = self.n_agents - robot_count
        for index in range(robot_count):
            y = (index - (robot_count - 1) / 2.0) * 0.55
            side = -1.0 if index % 2 == 0 else 1.0
            positions.append([2.0 * side, y])
            goals.append([-2.0 * side, y])
        for offset in range(agent_count):
            x = (offset - (agent_count - 1) / 2.0) * 0.55
            side = -1.0 if offset % 2 == 0 else 1.0
            positions.append([x, 2.0 * side])
            goals.append([x, -2.0 * side])
        return (
            torch.tensor(positions, device=device, dtype=dtype),
            torch.tensor(goals, device=device, dtype=dtype),
        )

    def reset_world_at(self, env_index: Optional[int] = None) -> None:
        positions, goals = self._layout_positions_and_goals()
        target_indices = (
            range(self.world.batch_dim) if env_index is None else (env_index,)
        )
        generator = torch.Generator(device="cpu")
        for target in target_indices:
            generator.manual_seed(self.seed + int(target))
            if self.position_jitter > 0:
                jitter = (
                    torch.rand(
                        positions.shape,
                        generator=generator,
                        device="cpu",
                    )
                    * 2.0
                    - 1.0
                ).to(self.world.device) * self.position_jitter
            else:
                jitter = torch.zeros_like(positions)
            current_positions = positions + jitter
            current_goals = goals - jitter
            self.initial_positions[target] = current_positions
            self.goals[target] = current_goals
            for index, agent in enumerate(self.world.agents):
                agent.set_pos(current_positions[index], batch_index=target)
                agent.set_vel(
                    torch.zeros(2, device=self.world.device), batch_index=target
                )

    def reward(self, agent: Agent) -> Tensor:
        index = self.world.agents.index(agent)
        distance = torch.linalg.vector_norm(
            self.goals[:, index] - agent.state.pos, dim=-1
        )
        return -distance

    def observation(self, agent: Agent) -> Tensor:
        index = self.world.agents.index(agent)
        return torch.cat(
            (agent.state.pos, agent.state.vel, self.goals[:, index]), dim=-1
        )

    def done(self) -> Tensor:
        positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        distances = torch.linalg.vector_norm(self.goals - positions, dim=-1)
        return torch.all(distances <= self.goal_tolerance, dim=-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        index = self.world.agents.index(agent)
        distance_to_goal = torch.linalg.vector_norm(
            self.goals[:, index] - agent.state.pos, dim=-1
        )
        minimum_clearance = torch.full(
            (self.world.batch_dim,), torch.inf, device=self.world.device
        )
        for other_index, other in enumerate(self.world.agents):
            if other_index == index:
                continue
            center_distance = torch.linalg.vector_norm(
                other.state.pos - agent.state.pos, dim=-1
            )
            clearance = center_distance - self.radii[index] - self.radii[other_index]
            minimum_clearance = torch.minimum(minimum_clearance, clearance)
        return {
            "distance_to_goal": distance_to_goal,
            "minimum_clearance": minimum_clearance,
        }

    def extra_render(self, env_index: int = 0) -> "List[object]":
        from vmas.simulator import rendering

        geoms: List[object] = []
        for index in range(self.n_agents):
            goal = self.goals[env_index, index]
            marker = rendering.make_circle(self.goal_tolerance, filled=False)
            transform = rendering.Transform()
            transform.set_translation(float(goal[0]), float(goal[1]))
            marker.add_attr(transform)
            marker.set_color(*Color.GREEN.value)
            geoms.append(marker)
        return geoms
