from abc import ABC, abstractmethod
from datasets import Dataset


class ClusteringBase(ABC):
    score_type = "base"

    @abstractmethod
    def run(self, instructanswer_dataset: Dataset) -> Dataset:
        pass
