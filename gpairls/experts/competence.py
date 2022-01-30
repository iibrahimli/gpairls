"""
Types of experts.
"""

from dataclasses import dataclass


@dataclass
class ExpertCompetenceConfig:
    """
    Parameters for experts of different competence
    """

    # type of expert this config refers to
    name: str

    # probability of providing an advice
    availability: float

    # probability of providing a correct action
    accuracy: float


class ExpertCompetencePreset:
    """
    Source: https://arxiv.org/pdf/2102.02441.pdf
    """

    OPTIMISTIC = ExpertCompetenceConfig(
        name="optimistic", availability=1.0, accuracy=1.0
    )
    REALISTIC = ExpertCompetenceConfig(
        name="realistic", availability=0.94870, accuracy=0.47316
    )
    PESSIMISTIC = ExpertCompetenceConfig(
        name="pessimistic", availability=0.47435, accuracy=0.23658
    )
