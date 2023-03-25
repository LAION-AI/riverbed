from abc import ABC, abstractmethod
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset


class ScorerBase(ABC):
    @abstractmethod
    def score(self, instructanswer_dataset: Dataset) -> Dataset:
        pass


class HFPipelineScorerBase(ScorerBase):
    def __init__(self, score_id: str, model_id: str, task: str, batch_size: int, device: int, max_length: int, **kwargs):
        self._model = pipeline(task, model=model_id, device=device, **kwargs)
        self._model_id = self._model.model.name_or_path
        self._score_id = score_id
        self._batch_size = batch_size
        self._max_length = max_length

    @abstractmethod
    def input_preprocessing(self, ia: dict) -> dict:
        """Preprocessing InstructAnswer into text for scorer input"""
        pass

    @abstractmethod
    def score_processing(self, output: dict) -> float:
        """Convert classifier output into float to cater for different output in HF model hub"""
        pass

    def score(self, instructanswer_dataset: Dataset) -> Dataset:
        """
        Preprocessing uses the map function in Dataset, without batching to avoid overhead
        Model inferencing follows the best practice of https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching
        """
        instructanswer_dataset = instructanswer_dataset.map(self.input_preprocessing,
                                                            desc=f"{self.__class__.__name__}_preprocessing")
        output = []
        for out in tqdm(self._model(KeyDataset(instructanswer_dataset, "text"),
                                    batch_size=self._batch_size, max_length=self._max_length, truncation=True),
                        total=len(instructanswer_dataset), desc=self.__class__.__name__):
            output.append(self.score_processing(out))

        instructanswer_dataset = instructanswer_dataset.remove_columns("text")
        instructanswer_dataset = instructanswer_dataset.add_column(f"{self._score_id}_score", output)
        instructanswer_dataset = instructanswer_dataset.add_column(f"{self._score_id}_model_id", [self._model_id] * len(output))
        return instructanswer_dataset
