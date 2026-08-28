# AVOCADO baseline source notice

This directory is a PyTorch/VMAS implementation written for this project. It
does not vendor or import the authors' `avocado` Python/C++ package.

The mathematical formulation and parameter mapping reference:

- D. Martínez-Baselga et al., “AVOCADO: Adaptive Optimal Collision Avoidance
  Driven by Opinion,” *IEEE Transactions on Robotics*, 2025;
- <https://github.com/dmartinezbaselga/AVOCADO>, particularly `src/Agent.cpp`
  and `actors.py`, consulted on 2026-08-27.

The finite velocity-obstacle construction and infeasible linear-programming
fallback follow the RVO2/ORCA algorithm. The official AVOCADO repository states
that its inherited RVO2 portions are available under Apache License 2.0 and its
AVOCADO modifications under AGPL-3.0-or-later. The repository's `LICENSE`,
`LICENSE-Apache-2.0`, `LICENSE-AGPL-3.0`, and source headers remain the
authoritative licensing records.

This notice records provenance; it is not a legal determination that the
current project's MIT license covers redistribution of every derived portion.
Review the upstream licenses before publishing or redistributing this module.
