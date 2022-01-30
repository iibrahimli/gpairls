"""
A training run class.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class TrainingRun:
    """
    Encapsulates statistics of a training run
    """

    def __init__(
        self,
        run_name: str,
        metadata: Dict[Any, Any],
        tracked_stats: List[str],
    ):
        self.run_name = run_name
        self.metadata = metadata
        self.stats = {stat: [] for stat in tracked_stats}

    def __repr__(self) -> str:
        return f"TrainingRun(run_name={self.run_name}, stats={list(self.stats.keys())})"

    def add(self, **kwargs: float):
        """
        Add values to the training run.
        """
        for key, value in kwargs.items():
            if key in self.stats:
                self.stats[key].append(value)

    def save(self, dir_path: str):
        """
        Save the training run to given directory.
        """

        # create directory if it does not exist
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # save metadata
        metadata_path = dir_path / "metadata.json"
        with metadata_path.open("w") as f:
            json.dump(self.metadata, f, indent=4)

        # save stats
        csv_path = dir_path / "stats.csv"
        df = pd.DataFrame(self.stats)
        df.set_index("episode", inplace=True)
        df.to_csv(csv_path)
