"""
Probabilistic Policy Reuse

Based on BPA code (https://github.com/mwizard1010/robot-control) with
refactoring, support for bisimulation-based retrieval, and type annotations.
"""

from typing import Any, Dict, Tuple
from dataclasses import dataclass

import numpy as np


@dataclass
class ActionNode:
    action: Any
    prob: float

    def __repr__(self) -> str:
        return f"AP(a={self.action}, p={self.prob:.2f})"


class PPR:
    """
    Probabilistic Policy Reuse class.

    Attributes:
        vals: A dictionary of key-value pairs where the key is an identifier
            and the value is an ActionNode object.
        init_prob: The initial probability of an action being chosen.
        decay_rate: The rate at which the probability is reduced on each step.
    """

    def __init__(self, init_prob: float = 0.8, decay_rate: float = 0.05):
        """
        Initialize a PPR object with given initial probability and decay rate.
        """
        self.vals: Dict[Any, ActionNode] = {}
        self.init_prob = init_prob
        self.decay_rate = decay_rate

    def __repr__(self) -> str:
        return f"PPR(init={self.init_prob}, decay={self.decay_rate}, vals={self.vals})"

    def step(self):
        """
        Reduce the probability of each action by the decay rate, without
        going below 0.
        """
        for key in self.vals:
            ap = self.vals[key]
            ap.prob = max(ap.prob - self.decay_rate, 0)

    def add(self, key: Any, action: Any):
        """
        Add a new action.
        """
        self.vals[tuple(key)] = ActionNode(
            action=action, prob=self.init_prob
        )

    def get(self, key: Any) -> Tuple[Any, float]:
        """
        Get the action and its probability for the given key, or (None, 0) if
        the key is not present.
        """
        key = tuple(key)
        if key in self.vals:
            return self.vals[key].action, self.vals[key].prob
        else:
            # select action with lowest Euclidean distance between embeddings
            min_dist = np.inf
            min_embed = None
            dist_sum = 0
            count = 0
            for emb in self.vals:
                dist = np.linalg.norm(np.array(emb) - np.array(key))
                dist_sum += dist
                count += 1
                if dist < min_dist:
                    min_dist = dist
                    min_embed = emb
            if count > 0 and min_dist > (dist_sum / count) / 2:
                return self.vals[min_embed].action, self.vals[min_embed].prob
            else:
                return None, 0