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
    """
    Source: https://arxiv.org/pdf/2102.02441.pdf
    """

    OPTIMISTIC = ExpertConfig(
        name="optimistic", availability=1.0, accuracy=1.0
    )
    REALISTIC = ExpertConfig(
        name="realistic", availability=0.94870, accuracy=0.47316
    )
    PESSIMISTIC = ExpertConfig(
        name="pessimistic", availability=0.47435, accuracy=0.23658
    )
