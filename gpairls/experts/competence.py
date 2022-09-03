"""
Types of experts.
"""

from dataclasses import dataclass


@dataclass
class ExpertConfig:
    """
    Parameters for experts
    """

    # probability of providing an advice
    availability: float

    # probability of providing a correct action
    accuracy: float

    @property
    def name(self):
        return f"avail-{self.availability}_acc-{self.accuracy}"


class ExpertPresets:
    OPTIMISTIC = ExpertConfig(
        availability=1.0, accuracy=1.0
    )
    REALISTIC = ExpertConfig(
        availability=0.75, accuracy=0.75
    )
    PESSIMISTIC = ExpertConfig(
        availability=0.2, accuracy=0.1
    )
