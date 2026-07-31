from __future__ import annotations

from collections.abc import Callable

from .algorithm import VPRAlgorithm

AlgorithmFactory = Callable[..., VPRAlgorithm]
ALGORITHM_REGISTRY: dict[str, AlgorithmFactory] = {}


def register_algorithm(name: str, factory: AlgorithmFactory) -> None:
    if not name:
        raise ValueError("Algorithm name cannot be empty.")
    ALGORITHM_REGISTRY[name] = factory


def create_algorithm(name: str, **kwargs: object) -> VPRAlgorithm:
    try:
        factory = ALGORITHM_REGISTRY[name]
    except KeyError as error:
        choices = ", ".join(sorted(ALGORITHM_REGISTRY)) or "none registered"
        raise ValueError(f"Unknown algorithm '{name}'. Available: {choices}") from error
    return factory(**kwargs)


def list_algorithms() -> list[str]:
    return sorted(ALGORITHM_REGISTRY)
