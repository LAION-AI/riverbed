# coding=utf-8
# Copyright 2021-2022, Ontocord, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math, os
import copy
from sklearn.cluster import MiniBatchKMeans
from time import time
import numpy as np
from collections import Counter
import kenlm
import statistics
import torch
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast, CLIPProcessor, CLIPModel, BertModel
import torch.nn.functional as F
import random
import spacy
import json
from dateutil.parser import parse as dateutil_parse
import pandas as pd
import itertools
from nltk.corpus import stopwords as nltk_stopwords
import pickle
from collections import OrderedDict
from fast_pytorch_kmeans import KMeans
import torch
import tqdm
import gzip
import multiprocessing
from torch import nn
from ..utils import *
from ..searcher_indexer import *

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

try:
  if minilm_model is not None: 
    pass
except:
   labse_tokenizer= labse_model=  clip_processor = minilm_tokenizer= clip_model= minilm_model= spacy_nlp= stopwords_set = None
   
#################################################################################
#VISUALIZER CODE
#this forms part of the view portion of the MVC
#used for exploring and viewing different parts of data and results from the analyzer. 
#display results and pandas data frames in command line, notebook or as a flask API service. 
class RiverbedVisualizer:
  def __init__(self, project_name, analyzer, searcher_indexer, processor):
    self.project_name, self.analyzer, self.searcher_indexer, self.processor = project_name, analyzer, searcher_indexer, processor
  
  def draw(self):
    pass
  
  def df(self, slice):
    pass
  
