"""
Implementation of simulated users.
"""

from abc import ABC, abstractmethod

import numpy as np

from .competence import ExpertCompetenceConfig, ExpertCompetencePreset


class Expert(ABC):
    def __init__(
        self, env_name: str, competence: ExpertCompetenceConfig, name: str
    ) -> None:
        self.env_name = env_name
        self.competence = competence
        self.name = name
        super().__init__()

    @abstractmethod
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Returns the recommended action to take after the given observation.
        """
        pass


class MountainCarExpert(Expert):
    def __init__(self) -> None:
        super().__init__(
            "MountainCarContinuous-v0",
            ExpertCompetencePreset.REALISTIC,
            "Realistic-MountainCar",
        )

    def get_action(self, obs):
        """
        Returns the recommended action to take after the given observation.
        """

        if np.random.random() < self.competence.availability:
            if np.random.random() < self.competence.accuracy:
                # correct action
                x, v = list(obs)
                if v > 0:
                    return [1.0]
                else:
                    return [-1.0]
            else:
                # incorrect (random) action
                return np.random.uniform(-1., 1., 1)

        else:
            # unavailable
            return None