from abc import ABC, abstractmethod
from datasets import Dataset


class ScoreFilterBase(ABC):
    filter_type = "base"

    def __init__(self):
        self._clean_dataset = []
        self._removed_dataset = []

    @abstractmethod
    def is_pass(self, instructanswer: dict) -> None:
        pass

    def _get_dataset_statistics(self, instructanswer_data: Dataset) -> None:
        pass

    def process(self, instructanswer_dataset: Dataset) -> None:
        """removal not inplace yet"""
        self._get_dataset_statistics(instructanswer_dataset)
        self._clean_dataset = instructanswer_dataset.filter(self.is_pass, desc="Slicing clean data")
        self._removed_dataset = instructanswer_dataset.filter(lambda x: not self.is_pass(x), desc="Slicing removed data")

    def get_clean_dataset(self):
        return self._clean_dataset

    def get_removed_dataset(self):
        """ For inspection"""
        return self._removed_dataset
