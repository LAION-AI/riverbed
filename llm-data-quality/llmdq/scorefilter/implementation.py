from typing import List
import re
from datasets import Dataset
from llmdq.config import FilterConfig
from llmdq.scorefilter.base import ScoreFilterBase
from llmdq.config import AbsoluteFilterConfig, ZScoreFilterConfig


class AbsoluteScoreFilter(ScoreFilterBase):
    filter_type = "absolute"

    def __init__(self, config_list: List[AbsoluteFilterConfig]):
        super().__init__()
        self._config_list = config_list

    def is_pass(self, instructanswer: dict) -> bool:
        score_id_list = [re.sub("_score$", "", k) for k in instructanswer.keys() if k.endswith("_score")]
        for score_id in score_id_list:
            for config in self._config_list:
                if score_id == config.score_id:
                    if config.direction == "ge" and instructanswer[score_id + "_score"] <= config.threshold:
                        return False
                    if config.direction == "le" and instructanswer[score_id + "_score"] >= config.threshold:
                        return False
        return True


class ZScoreFilter(ScoreFilterBase):
    filter_type = "zscore"

    def __init__(self, config_list: List[ZScoreFilterConfig]):
        super().__init__()
        self._config_list = config_list

    def _get_dataset_statistics(self, instructanswer_dataset: Dataset) -> None:
        self._dataset_stat = {}

        if not instructanswer_dataset:
            return

        data_len = len(instructanswer_dataset)
        score_id_list = [re.sub("_score$", "", k) for k in instructanswer_dataset.column_names if k.endswith("_score")]

        tmp_stat = {}
        for score_id in score_id_list:
            self._dataset_stat[score_id] = {}
            tmp_stat[score_id + "_running_x1"] = 0
            tmp_stat[score_id + "_running_x2"] = 0

        for score_id in score_id_list:
            for ia in instructanswer_dataset:
                tmp_stat[score_id + "_running_x1"] += ia[score_id + "_score"]
                tmp_stat[score_id + "_running_x2"] += ia[score_id + "_score"] ** 2

            mean = tmp_stat[score_id + "_running_x1"]/data_len
            std = (tmp_stat[score_id + "_running_x2"]/data_len - mean**2) ** 0.5

            self._dataset_stat[score_id]["mean"] = mean
            if std == 0:
                raise Exception(f"Dataset statistics {score_id} has zero std")
            self._dataset_stat[score_id]["std"] = std

    def is_pass(self, instructanswer: dict) -> bool:
        score_id_list = [re.sub("_score$", "", k) for k in instructanswer.keys() if k.endswith("_score")]
        for config in self._config_list:
            score_stat = self._dataset_stat[config.score_id]
            for score_id in score_id_list:
                if score_id == config.score_id:
                    z_score = (instructanswer[score_id + "_score"] - score_stat["mean"]) / score_stat["std"]
                    if config.direction == "ge" and z_score <= config.threshold:
                        return False
                    if config.direction == "le" and z_score >= config.threshold:
                        return False
        return True


class FilterPipeline(ScoreFilterBase):
    def __init__(self, config: FilterConfig):
        super().__init__()
        self._absfilter = AbsoluteScoreFilter(config.absolute)
        self._zscorefilter = ZScoreFilter(config.relative)

    def _get_dataset_statistics(self, instructanswer_dataset: Dataset):
        self._zscorefilter._get_dataset_statistics(instructanswer_dataset)

    def is_pass(self, instructanswer: dict):
        return (
                self._absfilter.is_pass(instructanswer) and
                self._zscorefilter.is_pass(instructanswer)
                )
