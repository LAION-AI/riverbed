from typing import List, Literal
from pydantic import BaseSettings


class AbsoluteFilterConfig(BaseSettings):
    score_id: str
    threshold: float
    direction: Literal["ge", "le"]


class ZScoreFilterConfig(BaseSettings):
    score_id: str
    threshold: float
    direction: Literal["ge", "le"]


class FilterConfig(BaseSettings):
    absolute: List[AbsoluteFilterConfig]
    relative: List[ZScoreFilterConfig]


class Config(BaseSettings):
    scorer: List[dict]
    scorefilter: FilterConfig
    clustering: List[dict]
