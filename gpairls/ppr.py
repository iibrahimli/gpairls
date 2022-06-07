"""
Probabilistic Policy Reuse

Based on BPA code (https://github.com/mwizard1010/robot-control) with
refactoring, support for bisimulation-based retrieval, and type annotations.
"""

from typing import Any, List, Tuple
from dataclasses import dataclass

import numpy as np


# TODO: maybe vary this over the course of training?
# 2 embeddings having distance less than this are considered the same
_EMB_DIST_THRESHOLD = 0.01


def _embedding_dist(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Euclidean distance between embeddings"""
    return np.linalg.norm(emb_a - emb_b)


@dataclass
class ActionNode:
    emb: np.ndarray
    action: Any
    prob: float

    def __repr__(self) -> str:
        return f"AP(emb={self.emb} a={self.action}, p={self.prob:.2f})"


class PPR:
    """
    Probabilistic Policy Reuse class. Stores embeddings and associated advice
    (actions), retrieves the advice based on embedding distance and

    Attributes:
        vals: A list of tuples of (embedding (np arrays), ActionNode).
        init_prob: The initial probability of an action being chosen.
        decay_rate: The rate at which the probability is reduced on each step.
    """

    def __init__(self, init_prob: float = 0.8, decay_rate: float = 0.05):
        """
        Initialize a PPR object with given initial probability and decay rate.
        """
        self.vals: List[ActionNode] = []
        self.init_prob = init_prob
        self.decay_rate = decay_rate

    def __repr__(self) -> str:
        return f"PPR(init={self.init_prob}, decay={self.decay_rate}, vals={self.vals})"

    def step(self):
        """
        Reduce the probability of each action by the decay rate and remove ones
        where probability reaches 0.
        """
        for i in range(len(self.vals)):
            self.vals[i].prob -= self.decay_rate
        self.vals = [v for v in self.vals if v.prob > 0]

    def add(self, emb: np.ndarray, action: Any):
        """
        Add a new action.
        """
        # reset probability and action if embedding is already present
        for i in range(len(self.vals)):
            if _embedding_dist(self.vals[i].emb, emb) < _EMB_DIST_THRESHOLD:
                self.vals[i].action = action
                self.vals[i].prob = self.init_prob
                return
        self.vals.append(ActionNode(emb, action, self.init_prob))

    def get(self, emb: np.ndarray) -> Tuple[Any, float]:
        """
        Get the action and its probability for the given emb, or (None, 0) if
        the emb is not present.
        """
        if len(self.vals) == 0:
            return None, 0.0

        distances = [_embedding_dist(emb, v.emb) for v in self.vals]
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        # no good embeddings found
        if min_dist > _EMB_DIST_THRESHOLD:
            return None, 0.0

        return self.vals[min_idx].action, self.vals[min_idx].prob