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
# TOKENIZER CODE
class RiverbedTokenizer:

  def __init__(self):
    self.compound = None
    self.synonyms = None
    self.token2weight = None
  
  def idx2token(self):
    return list (self.tokenweight.items())
  
  def token2idx(self):
    return OrderedDict([(term, idx) for idx, term in enumerate(self.tokenweight.items())])
    
  def tokenize(self, doc, min_compound_weight=0,  max_compound_word_size=10000, compound=None, token2weight=None, synonyms=None, use_synonym_replacement=False):
    if synonyms is None: synonyms = {} if not hasattr(self, 'synonyms') or not self.synonyms else self.synonyms
    if token2weight is None: token2weight = {} if not hasattr(self, 'token2weight') or not self.token2weight else self.token2weight    
    if compound is None: compound = {} if not hasattr(self, 'compound') or not self.token2weight else self.compound
    if not use_synonym_replacement: synonyms = {} 
    doc = [synonyms.get(d,d) for d in doc.split(" ") if d.strip()]
    len_doc = len(doc)
    for i in range(len_doc-1):
        if doc[i] is None: continue
        tokenArr = doc[i].strip("_").replace("__", "_").split("_")
        if tokenArr[0] in compound:
          min_compound_len = compound[tokenArr[0]][0]
          max_compound_len = min(max_compound_word_size, compound[tokenArr[0]][-1])
          for j in range(min(len_doc, i+max_compound_len), i+1, -1):
            if j <= i+min_compound_len-1: break
            token = ("_".join(doc[i:j])).strip("_").replace("__", "_")
            tokenArr = token.split("_")
            if len(tokenArr) <= max_compound_len and token in token2weight and token2weight.get(token, 0) >= min_compound_weight:
              old_token = token
              doc[j-1] = synonyms.get(token, token).strip("_").replace("__", "_")
              #if old_token != doc[j-1]: print (old_token, doc[j-1])
              for k in range(i, j-1):
                  doc[k] = None
              break
    return (" ".join([d for d in doc if d]))

  def save_pretrained(self, tokenizer_name):
      os.system(f"mkdir -p {tokenizer_name}")
      pickle.dump(self, open(f"{tokenizer_name}/tokenizer.pickle", "wb"))
    
  @staticmethod
  def from_pretrained(tokenizer_name):
      self = pickle.load(open(f"{tokenizer_name}/tokenizer.pickle", "rb"))
      return self
