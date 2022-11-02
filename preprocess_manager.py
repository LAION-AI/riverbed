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
import fasttext, langid
import os
fasttext_model = os.path.abspath(os.path.dirname(__file__)))  +"/bin/lid.176.ftz"
lang_model = fasttext.load_model(fasttext_model)


def get_lang_groups(src_lang):
    """ we use langid because it's pretty fast but it has difficulties in low resource languages
    langid can sometimes mistake languages that are in the same group. that is ok for our purpose as
    we mainly use the langid check to confirm the labels from other models. """
    lang_groups={src_lang}
    if src_lang in {'ig', 'sn', 'ny', 'st', 'zu', 'xh', 'rw', 'sw', 'yo', 'so'}:
      lang_groups = {'ig', 'sn', 'ny', 'st', 'zu', 'xh', 'rw', 'sw', 'yo', 'so'}
    elif src_lang in {'mr', 'ne', 'hi', }:
      lang_groups = {'mr', 'ne', 'hi', }
    elif src_lang in {'fr', 'br'}:
      lang_groups = {'fr','la', 'br' }
    elif src_lang in {'pt', }:
      lang_groups = {'pt','la', 'gl' }
    elif src_lang in {'eo', 'es', 'oc', 'ca', 'eu', 'an', 'gl' }:
      lang_groups = {'eo', 'es', 'oc', 'ca', 'eu', 'an', 'gl', 'la' }
    elif src_lang in {'arz', 'ar', 'fa', 'ur', 'az', 'azb', 'ckb', 'ps' }:
      lang_groups = {'arz', 'ar', 'fa', 'ur', 'az', 'azb', 'ckb', 'ps' }
    elif src_lang in {'id', 'ms', }:
      lang_groups = {'id', 'ms',}
    elif src_lang in {'as', 'bn', 'bpy'}:
      lang_groups = {'as', 'bn', 'bpy'}
    elif src_lang in {'af', 'nl', }:
      lang_groups = {'af', 'nl',}
    elif src_lang in {'bo', 'dz', }:
      lang_groups = {'bo', 'dz',}
    elif src_lang in {'bs', 'hr', }:
      lang_groups = {'bs', 'hr',}
    elif src_lang in {'bxr', 'mn', }:
      lang_groups = {'bxr', 'mn',}
    elif src_lang in {'ceb', 'tl', }:
      lang_groups = {'ceb', 'tl',}
    elif src_lang in {'cs', 'sk', }:
      lang_groups = {'cs', 'sk',}
    elif src_lang in {'da', 'no', }:
      lang_groups = {'da', 'no',}
    elif src_lang in {'eml', 'wa', }:
      lang_groups = {'eml', 'wa',}
    elif src_lang in {'de', 'lb', 'pl', 'dsb'}:
      lang_groups = {'de', 'lb', 'pl', 'dsb'}
    elif src_lang in {'id', 'jv', 'ms', 'tl',}:
      lang_groups = {'id', 'jv', 'ms', 'tl', }
    elif src_lang in {'av', 'ru', 'bg', 'ba', 'kk', 'ky', 'uk', 'be', 'ce', 'cv'}:
      lang_groups = {'av', 'ru', 'bg', 'ba', 'kk', 'ky', 'uk', 'be', 'ce', 'cv'}
    return lang_groups
    
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

import re

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)


#TODO: add resiliparse lang detect?
def lang_id(document, cleanup_emoji=False, len_cutoff=1000):
  global lang_model
  document = document.lower().replace("\n", " ")
  if len_cutoff and len(document) > len_cutoff: document = document[:len_cutoff]
  if cleanup_emoji:
    document = emoji_pattern.sub(r'', document)
  pred = lang_model.predict(document)
  lang = pred[0][0].replace("__label__", "")
  score_pred = pred[1][0]
  lang2 = langid.classify(document)
  lang2 = lang2[0]
  lang_group = get_lang_groups(lang2)
  if lang2 != lang or (score_pred > 0.4 and score_pred < 0.7):
    if lang not in lang_group:
      pass
      #print (lang, lang2, score_pred)
    else:
      score_pred = score_pred*1.5
  return lang, score_pred
  
