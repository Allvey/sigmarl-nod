"""Independent Opinion Dynamics + MARL components.

M2 exposes configuration and M3 exposes pure mathematical modules. Stateful
components are added later. This package intentionally performs no eager
imports, so loading the typed configuration does not import mathematical
networks into the Base/no-op path. Import concrete components from their own
modules.
"""
