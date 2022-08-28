"""
Types of experts.
"""

from dataclasses import dataclass


@dataclass
class ExpertConfig:
    """
    Parameters for experts
    """

    # type of expert this config refers to
    name: str

    # probability of providing an advice
    availability: float

    # probability of providing a correct action
    accuracy: float


class ExpertPresets:
    OPTIMISTIC = ExpertConfig(
        name="optimistic", availability=1.0, accuracy=1.0
    )
    REALISTIC = ExpertConfig(
        name="realistic", availability=0.75, accuracy=0.75
    )
    PESSIMISTIC = ExpertConfig(
        name="pessimistic", availability=0.2, accuracy=0.1
    )
