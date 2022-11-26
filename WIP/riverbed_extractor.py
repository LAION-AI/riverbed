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
import langid
from .utils import *
from .searcher_indexer import *
from ..char_manager import junk, special_char
from ..stopwords import stopwords as all_stopwords
from ..langid_manager import *
from ..banned_words import banned_words
from ..flagged_words import flagged_words
from ..cjk import lang_is_cjk

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

try:
  if minilm_model is not None: 
    pass
except:
   labse_tokenizer= labse_model=  clip_processor = minilm_tokenizer= clip_model= minilm_model= spacy_nlp= stopwords_set = None

import sys, os
import itertools
try:
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))         
except:
    pass


_lang_2_max_stopword_len = dict([(lang, max(s.count(" ")+1 if not lang_is_cjk(lang) else len(s) for s in arr)) for lang, arr in all_stopwords.items()])
_lang_2_max_bannedword_len = dict([(lang, max(s.count(" ")+1 if not lang_is_cjk(lang) else len(s) for s in arr)) for lang, arr in banned_words.items()])
_lang_2_max_flaggedword_len = dict([(lang, max(s.count(" ")+1 if not lang_is_cjk(lang) else len(s) for s in arr)) for lang, arr in flagged_words.items()])

def extract_junk_ratio(self, data, infield, outfield):
    s = data[infield]
    s = s.lower().strip()
    len_s = len(s)
    if len_s == 0: return data
    data[outfield] = len([s2 for s2 in s if s2 in junk])/len(s)
    return data
  
def extract_compound_ratio(self, data, infield, outfield):
    s = data[infield]
    s = s.lower().strip()
    len_s = len(s)
    if len_s == 0: return data
    s = s.split()
    data[outfield] = sum([0 if "_" not in s2 else s2.count("_")**2 for s2 in s])/len(s)
    return data
  
def extract_stopword_ratio(self, data, infield, outfield):  
    if hasattr(self, 'src_lang'):
      src_lang = self.src_lang
    else:
      src_lang='en'
    s = data[infield]
    s = s.lower().strip()
    len_s = len(s)
    if len_s == 0: 
      data[outfield] = 0
      return data 
    if lang_is_cjk(src_lang):
      s_arr = s
    else:
      s_arr = [s2.strip(special_char) for s2 in s.lower().split() if s2.strip(special_char)]
        
    stop_cnt = total_cnt = 1
    if not stopwords:
        data[outfield] = 0
        return data 
    else:
        word_len = lang_2_max_stopword_len.get(src_lang, max_word_len)
        len_s = len(s_arr)
        stop_cnt = 0
        total_cnt = 0
        for i in range(len_s):
          if s_arr[i] is None: continue
          for j in range(min(len_s, i+word_len), i, -1):
            word = "".join(s_arr[i:j]) if is_cjk else " ".join(s_arr[i:j])
            if word in stopwords:
              stop_cnt += 1
              s_arr[i] = "".join(s_arr[i:j]) if is_cjk else " ".join(s_arr[i:j]) 
              for k in range(i+1, j):
                s_arr[k] = None
              break
          total_cnt += 1
        data[outfield] =  (stop_cnt/total_cnt) 
        return data

def extract_flagged_words_ratio(self, data, infield, outfield, ):
        if hasattr(self, 'lang_groups'):
          lang_groups = self.lang_groups
        else:
          lang_groups=[]  
        if hasattr(self, '_bannedwords'):
          _bannedwords = self._bannedwords
        else:
          _bannedwords = self._bannedwords = set(list(itertools.chain(*[list(banned_words.get(lang, [])) for lang in list(lang_groups)+['en']])))
        if hasattr(self, '_flaggedwords'):
          _flaggedwords = self._flaggedwords 
        else:
          _flaggedwords = self._flaggedwords = set(list(itertools.chain(*[list(flagged_words.get(lang, [])) for lang in list(lang_groups)+['en']])))
        if hasattr(self, 'src_lang'):
          src_lang = self.src_lang
        else:
          src_lang='en'
        s = data[infield]
        s = s.lower().strip()
        len_s = len(s)
        if len_s == 0: return 0
        if lang_is_cjk(src_lang):
          s_arr = s
        else:
          s_arr = [s2.strip(special_char) for s2 in s.lower().split() if s2.strip(special_char)]
        b_cnt = 0
        f_cnt = 0
        total_cnt = 0
        for i in range(len_s):
          if s_arr[i] is None: continue
          word_len = max_flagged_banned_word_len
          for j in range(min(len_s, i+word_len),i,-1):
            word = "".join(s_arr[i:j]) if is_cjk else " ".join(s_arr[i:j])
            is_flagged = word in _flaggedwords
            is_banned = word in _bannedwords
            if is_flagged or is_banned:
              if is_flagged: f_cnt += 1
              if is_banned: b_cnt += 1
              s_arr[i] =  word
              for k in range(i+1, j):
                s_arr[k] = None
          total_cnt += 1
        data[outfield] = (f_cnt/total_cnt) *  (1+(1000*b_cnt/total_cnt))
        return data
         
def extract_langid(self, data, infield, outfield):
      line = data[infield]
      lang =  langid.classify(line)
      if lang:
        lang = lang[0]
      else:
        lang = "en"
      data[outfiled] = lang
      return data
    

def extract_ents(self, data, infield, outfield):
    line = data[infield]
    curr_ents = list(itertools.chain(*[[(e.text, e.label_)] if '||' not in e.text else [(e.text.split("||")[0].strip(), e.label_), (e.text.split("||")[-1].strip(), e.label_)] for e in spacy_nlp(line).ents]))
    curr_ents = list(set([e for e in curr_ents if e[0]]))
    curr_ents.sort(key=lambda a: len(a[0]), reverse=True)
    ent_cnts = Counter(v[1].lower()+"_cnt" for v in curr_ents)
    for feature_label, cnt in ent_cnts.items():
          data[feature_label] = cnt
    data[outfield] = curr_ents
    return data
  
def extract_intro_with_date(self, data, infield, outfield):
      text, position, ents = data[infield], data['position'], data['ents']
      if position < 0.05 and text.strip() and (len(text) < 50 and text[0] not in "0123456789" and text[0] == text[0].upper() and text.split()[-1][0] == text.split()[-1][0].upper()):
        date = [e[0] for e in ents if e[1] == 'DATE']
        if date: 
          date = date[0]
          date = dateutil_parse_ext(date)
        if  date: 
          data[outfield] = 'intro: date of '+ date +"; "+text + " || "
        else:
          data[outfield] = 'intro: ' +text + " || "
      return data
    
def extract_section_with_date(self, data, infield, outfield):
      text, position, ents = data[infield], data['position'], data['ents']
      if  position >= 0.05 and position < 0.95 and text.strip() and (len(text) < 50 and text[0] not in "0123456789" and text[0] == text[0].upper() and text.split()[-1][0] == text.split()[-1][0].upper()):
        date = [e[0] for e in ents if e[1] == 'DATE']
        if date: 
          date = date[0]
          date = dateutil_parse_ext(date)
        if  date: 
          data[outfield] =  'section: date of '+ date +"; "+text + " || "
        else:
          data[outfield] =   'section: ' +text + " || "
      return data

def extract_conclusion_with_date(self, data, infield, outfield):
      text, position, ents = data[infield], data['position'], data['ents']
      if  position >= 0.95 and text.strip() and (len(text) < 50 and text[0] not in "0123456789" and text[0] == text[0].upper() and text.split()[-1][0] == text.split()[-1][0].upper()):
        date = [e[0] for e in ents if e[1] == 'DATE']
        if date: 
          date = date[0]
          date = dateutil_parse_ext(date)
        if  date: 
          return 'conclusion: date of '+ date +"; "+text + " || "
        else:
          return 'conclusion: ' +text + " || "
      return data
    
def extract_perplexity(self,data, infield, outfield):
     data[outfield] = self.get_perplexity(data[infield])
     return data

  
  
# for extracting a prefix for a segment of text. a segment can contain multiple spans.
# the prefixes are used to create more informative embeddings for a span.
# tuples of (feature_label, lower_band, upper_band, extractor). assumes prefix extraction has occured.
#TODO: other potential features include 
#   - similarity of embedding from its cluster centroid
#   - term_frequency-inverse_document_frequency weight
#   - inverse cluster size as a feature.
#   - cosine distance to nearest neighbor cluster head

#NOTE: Consider whether we want to make the below a hash for easier readability
#NOTE: The last  field is for batch_size for extractors that are more efficient in running in batches
default_extractors = [
      ('ents', None, None, extract_ents, 'text', 'ents', 1), 
      ('intro_with_date', None, None, extract_intro_with_date, 'text', 'prefix', 1), 
      ('section_with_date', None, None, extract_section_with_date, 'text', 'prefix', 1), 
      ('conclusion_with_date', None, None, extract_conclusion_with_date, 'text', 'prefix', 1), 
      ('langid', None, None, extract_langid, 'text', 'langid', 1), 
      ('stopword_ratio', .5, 1.5, extract_stopword_ratio, 'text', 'stopword_ratio', 1),
      ('junk_ratio', .5, 1.5, extract_junk_ratio, 'text', 'junk_ratio', 1),
      ('compound_word_ratio', .5, 1.5, extract_compound_word_ratio, 'text', 'compound_word_ratio', 1),
      ('flagged_words_ratio', .5, 1.5, extract_flagged_words_ratio, 'text', 'flagged_words_ratio', 1),
      ('perplexity', .5, 1.5, extract_perplexity, 'text', 'perplexity', 1),
      ]

    
RELATIVE_LOW = 0
RELATIVE_MEDIUM = 1
RELATIVE_HIGH= 2  
  
#################################################################################
#FEATURE EXRACTORS
#Given a corpus, create feature detectors
#Given new text, label the text with the features.
#Feature extractors DO NOT transform the text. Instead, it create new fields
#in a json/dict object. On type of feature is the "prefix" feature which is 
#a free form text that has information about a paragraph or multiple sentence section
#of a document. 
class RiverbedFeatureExtractor(nn.Module):
 
  def __init__(self, model):
    super().__init__()
    global labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set
    self.model = model
    self.project_name  = model.project_name 
    labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set = init_models()
   
  
  # returns data which can be used to store in the feature_label for a span. if upper_band and lower_band are set, then an additional label X_level stores
  # the relative level label as well.
  # NOTE: If we are running in incremental mode, 
  def extract(self, batch, running_features_per_label, need_level=None, running_features_size=100, feature_extractors=default_extractors):
    
    def batch_iter(batch, batch_size):
      if batch_size == 1:
        for data in batch: 
          yield [data]
      else:
        a_batch = []
        for data in batch:
          a_batch.append(data)
          if len(a_batch) >= batch_size:
            yield a_batch
            a_batch = []
        if a_batch:
          yield a_batch
    ###
    assert type(batch) is list  
    if need_level is None:
      need_level = {
        'need_to_high': True,
        'need_to_low': True,
        'need_to_medium':True
      }
    for extractor_specs in feature_extractors:
      feature_label, lower_band, upper_band, extractor, infield, outfield, batch_size = extractor_specs
      need_to_high = True
      need_to_low = True
      need_to_medium = True
      running_features = running_features_per_label[feature_label] = running_features_per_label.get(feature_label, [])
      #do an initial run to get the initial standard deviation and mean
      prev_batch = []
      if lower_band is not None:
        if len(running_features) < running_features_size:
          for a_batch in batch_iter(batch, batch_size):
              if len(a_batch) == 1 and batch_size == 1:
                b_batch = [extractor(self, a_batch[0], infield, outfield)]
              else:
                b_batch = extractor(self, a_batch, infield, outfield)
              for data in b_batch:
                  if not data.get(outfield): 
                    running_features.append(None)
                    continue
                  p = data[outfield]
                  running_features.append(p)
              if len(running_features) < running_features_size: break
                
        stdv = statistics.stdev(running_features)
        mn = statistics.mean (running_features)
        relative_label = self.RELATIVE_LOW
      idx = 0
      for a_batch in batch_iter(batch, batch_size):
        if len(a_batch) == 1 and batch_size == 1:
          b_batch = [extractor(self, a_batch[0], infield, outfield)]
        else:
          b_batch = extractor(self, a_batch, infield, outfield)
        if lower_band is not None:
          for data in b_batch:
            ret = data[outfield]
            running_features.append(ret)
            idx += 1
            if idx % int(running_features_size/2) == 0:
              stdv = statistics.stdev(running_features)
              mn = statistics.mean (running_features)
              while len(running_features) > running_features_size:
                running_features.pop()              
            if abs(ret-mn) >= stdv*upper_band and need_level['need_to_high']:
              relative_label = self.RELATIVE_HIGH
              need_level['need_to_high'] = False
              need_level['need_to_low'] = True
              need_level['need_to_medium'] = True
            elif  abs(ret-mn) < stdv*upper_band and abs(ret-mn) > stdv*lower_band and need_level['need_to_medium']:
              relative_label = self.RELATIVE_MEDIUM
              need_level['need_to_high'] = True
              need_level['need_to_low'] = True
              need_level['need_to_medium'] = False
            elif abs(ret-mn) <= stdv*lower_band and need_level['need_to_low']:
              relative_label = self.RELATIVE_LOW
              need_level['need_to_high'] = False
              need_level['need_to_low'] = True
              need_level['need_to_medium'] = False
            data[outfield+"_level"] = relative_label 

    return batch
