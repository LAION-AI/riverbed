from typing import Iterable, List, Tuple, Optional
from tqdm import tqdm
from evaluate import load
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers import ElectraForPreTraining, ElectraTokenizerFast
from flair.data import Sentence
from flair.models import SequenceTagger
import torch
import numpy as np
from llmdq.scorer.base import ScorerBase, HFPipelineScorerBase
import logging


logging.disable(logging.WARNING)


class RewardModelScorer(HFPipelineScorerBase):
    def __init__(self, score_id, model_id, task="text-classification", batch_size=8, max_length=1024, top_k=1, device=-1):
        super().__init__(score_id, model_id, task, batch_size, device, max_length, top_k=top_k)

    def input_preprocessing(self, ia: dict) -> dict:
        return {"text": ia["instruct"] + '\n' + ia["answer"]}

    def score_processing(self, output: dict) -> float:
        return output[0]['score']


class PerplexityScorer(ScorerBase):
    def __init__(self, score_id, model_id, max_length=1024, batch_size=8, device=-1):
        self._perplexity = load("perplexity", module_type="measurement", device=device)
        self._model_id = model_id
        self._batch_size = batch_size
        self._max_length = max_length
        self._score_id = score_id

        # preload the model
        self._perplexity.compute(data=["prewarm model"], model_id=self._model_id)

    def input_preprocessing(self, ia: dict) -> dict:
        return {"text": ia["instruct"] + '\n' + ia["answer"]}

    def score(self, instructanswer_dataset: Dataset) -> Dataset:
        instructanswer_dataset = instructanswer_dataset.map(self.input_preprocessing,
                                                            desc=f"{self.__class__.__name__}_preprocessing")
        output = self._perplexity.compute(data=instructanswer_dataset['text'], model_id=self._model_id,
                        batch_size=self._batch_size, max_length=self._max_length)
        instructanswer_dataset = instructanswer_dataset.remove_columns("text")
        instructanswer_dataset = instructanswer_dataset.add_column(f"{self._score_id}_score", output['perplexities'])
        instructanswer_dataset = instructanswer_dataset.add_column(f"{self._score_id}_model_id", [self._model_id] * len(output['perplexities']))
        return instructanswer_dataset


class ToxicityScorer(HFPipelineScorerBase):
    def __init__(self, score_id, model_id, task="text-classification", batch_size=8, max_length=1024, top_k=None, device=-1):
        super().__init__(score_id, model_id, task, batch_size, device, max_length, top_k=top_k)

    def input_preprocessing(self, ia: dict) -> dict:
        return {"text": ia["instruct"] + '\n' + ia["answer"]}

    def score_processing(self, output: dict) -> float:
        return max([s["score"] for s in output])


class GibberishScorer(HFPipelineScorerBase):
    def __init__(self, score_id, model_id, task="text-classification", batch_size=8, max_length=1024, top_k=None, device=-1):
        super().__init__(score_id, model_id, task, batch_size, device, max_length, top_k=top_k)

    def input_preprocessing(self, ia: dict) -> dict:
        return {"text": ia["instruct"] + '\n' + ia["answer"]}

    def score_processing(self, output: dict) -> float:
        return 1.0 - [l for l in output if l['label'] == 'clean'][0]['score']


class ContradictionScorer(HFPipelineScorerBase):
    def __init__(self, score_id, model_id, task="zero-shot-classification", batch_size=8, max_length=1024, device=-1):
        super().__init__(score_id, model_id, task, batch_size, device, max_length)
        self._label_name = ["entailment", "neutral", "contradiction"]

    def input_preprocessing(self, ia: dict) -> dict:
        return {"text": ia["answer"]}

    def score_processing(self, output: dict) -> float:
        idx = output['labels'].index('contradiction')
        return output['scores'][idx]

    def score(self, instructanswer_dataset: Dataset) -> Dataset:
        instructanswer_dataset = instructanswer_dataset.map(self.input_preprocessing,
                                                            desc=f"{self.__class__.__name__}_preprocessing")
        output = []
        for out in tqdm(self._model(KeyDataset(instructanswer_dataset, "text"), self._label_name, multi_label=False,
                                    batch_size=self._batch_size, max_length=self._max_length),
                        total=len(instructanswer_dataset), desc=self.__class__.__name__):
            output.append(self.score_processing(out))
        instructanswer_dataset = instructanswer_dataset.remove_columns("text")
        instructanswer_dataset = instructanswer_dataset.add_column(f"{self._score_id}_score", output)
        instructanswer_dataset = instructanswer_dataset.add_column(f"{self._score_id}_model_id", [self._model_id] * len(output))
        return instructanswer_dataset


class ReplacedTokenScorer(ScorerBase):
    def __init__(self, score_id, model_id, max_length=512, batch_size=8, device=-1):
        self._score_id = score_id
        self._model_id = model_id
        self._discriminator = ElectraForPreTraining.from_pretrained(model_id)
        self._tokenizer = ElectraTokenizerFast.from_pretrained(model_id)
        self._ner_model = SequenceTagger.load("flair/ner-english-ontonotes-large")
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = device
        self._device_pt = f"cuda:{self._device}" if self._device >= 0 else "cpu"
        self._discriminator.to(self._device_pt)

    def _run_ner(self, text_batch: List[str]) -> List[Sentence]:
        chr_len = self._max_length * 6  # expect a token has six character
        setence_batch = []
        for t in text_batch:
            try:
                # truncate flair sentence to approximately 512 tokens to avoid OOM in ner model
                _t_split = t.split(" ")
                n_space = len(_t_split) - 1 if len(_t_split) > 1 else 0
                setence_batch.append(Sentence(t[:chr_len + n_space]))
            except ValueError:
                # flair has a bug that has to be caught, replaced with empty string.
                # Sentence("ʼin's")
                setence_batch.append(Sentence(""))
        self._ner_model.predict(setence_batch, mini_batch_size=self._batch_size)
        return setence_batch

    def _run_discriminator(self, text_batch: List[str]) -> Tuple[List[List[float]], List[List[int]]]:
        prob_list = []
        inputs = self._tokenizer(text_batch, return_tensors="pt", truncation=True, padding=True,
                                 max_length=self._max_length, return_offsets_mapping=True).to(self._device_pt)
        offset_mapping = inputs.pop("offset_mapping")
        offset_mapping = [i for i in offset_mapping.to('cpu').tolist() if i != [0, 0]]
        discriminator_outputs_real = self._discriminator(**inputs)
        for prob in torch.sigmoid(discriminator_outputs_real.logits).cpu().detach().tolist():
            prob_list.append(prob)

        return prob_list, offset_mapping

    def _get_entity_score(self, text_batch: List[str]) -> List[float]:
        """
        The score is calculated as max{score_entity} if entity is identified else 0.0.
        Effectively it makes the downstream filter only applies on text with entity.
        """

        ner_result = self._run_ner(text_batch)
        score_list, offset_mapping = self._run_discriminator(text_batch)

        def linear_search(target: Tuple[int, int], offset: List[List[int]]) -> Tuple[Optional[int], Optional[int]]:
            """
            Given target: [3, 7] and offset [[0, 2], [2, 3], [3, 5], [5, 7], [7, 9]],
            return starting index and ending index of [3, 7] in offset which is 2, 3
            """
            start_idx, end_idx = None, None
            target_s, target_e = target
            for i, o in enumerate(offset):
                if o[0] == target_s:
                    start_idx = i
            for i, o in enumerate(offset):
                if o[1] == target_e:
                    end_idx = i
            return start_idx, end_idx

        # based on ner result, and token offset mapping, map the score to named entities and calclulate entity score
        score_dict_list = []
        for sent, score, offset in zip(ner_result, score_list, offset_mapping):
            score_dict = {}
            for t in sent.get_spans('ner'):
                score_idx = linear_search((t.start_position, t.end_position), offset)
                if t.tag not in score_dict:
                    score_dict[t.tag] = []
                if score_idx[0] is not None and score_idx[1] is not None:
                    score_dict[t.tag].append(np.mean([i for i in score[score_idx[0]: score_idx[1] + 1]]))

            for t in score_dict:
                score_dict[t] = np.mean(score_dict[t])

            score_dict_list.append(score_dict)

        return [max(score_dict.values()) if score_dict else 0.0 for score_dict in score_dict_list]

    def _batching(self, iterable: list) -> Iterable:
        length = len(iterable)
        for ndx in range(0, length, self._batch_size):
            yield iterable[ndx:min(ndx + self._batch_size, length)]

    def score(self, instructanswer_dataset: Dataset) -> Dataset:
        output = []
        for d in tqdm(self._batching(instructanswer_dataset),
                      desc=self.__class__.__name__,
                      total=len(instructanswer_dataset)//self._batch_size+1):
            output.extend(self._get_entity_score(d["answer"]))
        instructanswer_dataset = instructanswer_dataset.add_column(f"{self._score_id}_score", output)
        instructanswer_dataset = instructanswer_dataset.add_column(f"{self._score_id}_model_id", [self._model_id] * len(output))
        return instructanswer_dataset


class LengthScorer(ScorerBase):
    def __init__(self, score_id: str):
        self._score_id = score_id

    def input_preprocessing(self, ia: dict) -> dict:
        return {
            f"{self._score_id}_score": len(ia["answer"]),
            f"{self._score_id}_model_id": "rule"
        }

    def score(self, instructanswer_dataset: Dataset) -> Dataset:
        instructanswer_dataset = instructanswer_dataset.map(self.input_preprocessing,
                                                            desc=self.__class__.__name__)
        return instructanswer_dataset


class ScorerPipeline:
    def __init__(self):
        self._scorer_list = []

    def add(self, scorer_list) -> None:
        self._scorer_list.extend(scorer_list)

    def score(self, instructanswer_dataset: Dataset) -> Dataset:
        for scorer in self._scorer_list:
            instructanswer_dataset = scorer.score(instructanswer_dataset)
        return instructanswer_dataset
