#@title Basic Filtering Code
"""
Copyright, 2021-2022 Ontocord, LLC, All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#adapted from https://github.com/piisa/muliwai/blob/main/preprocess_manager.py, and
#https://github.com/piisa/muliwai/blob/7ccf36d016fa66a9b1ed00b0ce8b89d01a57dfc9/langid_manager.py which are under Apache 2.0

from .stopwords import all_stopwords
from .char_manager import *
import fasttext, langid
import os
fasttext_model = os.path.abspath(os.path.dirname(__file__))  +"/bin/lid.176.ftz"
lang_model = fasttext.load_model(fasttext_model)

from collections import Counter
def get_ngram(sent, window_size=3, lang="en"):
  if lang in {"zh", "ja", "ko", "th"}:
    tokens = sent
    ret= ["".join(tokens[i : i + window_size])   for i in range(len(tokens) - window_size)]
  else:
    tokens = sent.split()
  ret= [" ".join(tokens[i : i + window_size])   for i in range(len(tokens) - window_size)]
  return Counter(ret)

def get_ngram_score(sent, window_size=3, lang="en"):
  aHash = get_ngram(sent, window_size, lang)
  sent_len = sent.count(" ")+1
  for key in list(aHash.keys()):
    aHash[key] = aHash[key]/sent_len
  return aHash.most_common()[1]
  
def get_special_char_score (text, special_characters_default=None):
  global junk
  if special_characters_default is None: special_characters_default = junk
  return len([a for a in text if a in special_characters_default])/len(text)


    
lang_2_max_stopword_len = dict([(lang, max(s.count(" ")+1 if lang not in {'zh', 'zh-classical', 'zh-min-nan', 'zh-yue', 'ko', 'ja', 'th', 'jv'} else len(s) for s in arr)) for lang, arr in all_stopwords.items()])

def get_stopword_score(lang, doc, max_word_len=3, cjk_scale=1.5):
    is_cjk = lang in {'zh', 'zh-classical', 'zh-min-nan', 'zh-yue', 'ko', 'ja', 'th', 'jv'}
    stopwords =  all_stopwords.get(lang, {})
    if not stopwords: return 1
    doc = doc.lower().strip()
    if is_cjk: 
      s_arr = list("".join(doc.split())) 
    else: 
      s_arr = doc.split()
    word_len = lang_2_max_stopword_len.get(lang, max_word_len)
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
    stopword_score =  (stop_cnt/total_cnt) 
    if is_cjk: stopword_score = stopword_score*cjk_scale
    return (stopword_score)

def get_score_moving_avg(lang, text, scores_per_lang, simple_moving_avg_window=10, fn=None):
    if fn is None: fn = get_stopword_score
    _min_cutoff = _stdev = _mean = _median = 0
    _score = fn(lang, text)
    _min_cutoff = _mean = _median = _stdev = _quantiles = None
    scores_per_lang[lang] = _scores = _scores_per_lang.get(lang,[])
    _scores.append(_score)
    if len(_scores) < 2:
      _min_cutoff = _score
    else:
      if len(_scores) > simple_moving_avg_window:
         _scores_per_lang[lang] = _scores=_scores[-simple_moving_avg_window:]
      _stdev = statistics.stdev(_scores)
      _mean = statistics.mean (_scores)        
      _median = statistics.median (_scores)        
      _min_cutoff = _mean-(_stdev*_stdev_lower_bound)
    return _min_cutoff, _stdev, _mean, _median 
        
import re

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)


  
