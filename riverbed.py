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

# NOTES:
# we want to create feature detectors to segment spans of text. One way is to do clustering of embeddings of text.
# this will roughly correspond to area of similarities or interestingness. 
# we could do changes in perplexity, changes in embedding similarity, and detection of patterns such as section headers.
# we could have heuristics hand-crafted rules, like regexes for ALL CAPs folowed by non ALL CAPS, or regions of low #s of stopwords, followed by high #s of stopwords.
# or regions of high count of numbers ($1000,000).

# we could also run segments of text through counting the "_", based on sentence similarities, etc. and create a series.
# below is a simple detection of change from a std dev from a running mean, but we could do some more complex fitting using:
# the library ruptures. https://centre-borelli.github.io/ruptures-docs/examples/text-segmentation/

# with region labels, we can do things like tf-idf of words, and then do a mean of the tf-idf of a span. A span with high avg tf-idf means it is interesting or relevant. 

import math, os
import copy
import fasttext
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from time import time
import numpy as np
from collections import Counter
import kenlm
import statistics
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import random
import spacy
import json
from dateutil.parser import parse as dateutil_parse
import pandas as pd
from snorkel.labeling import labeling_function
import itertools
from nltk.corpus import stopwords
import pickle
if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

try:
  if sim_model is not None: 
    pass
except:
  sim_model = None
  sim_tokenizer = None
if sim_model is None:
  
  sim_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
  if device == 'cuda':
    sim_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').half().eval().to(device)
  else:
    sim_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').eval().to(device)
  spacy_nlp = spacy.load('en_core_web_md')
  stopwords_set = set(stopwords.words('english') + ['could', 'should', 'shall', 'can', 'might', 'may', 'include', 'including'])

class Riverbed:
  def __init__(self):
    pass

  @staticmethod
  def pp(log_score, length):
        return float((10.0 ** (-log_score / length)))

  def get_perplexity(self,  doc, kenlm_model=None):
    if kenlm_model is None: kenlm_model = {} if not hasattr(self, 'kenlm_model') else self.kenlm_model
    doc_log_score = doc_length = 0
    doc = doc.replace("\n", " ")
    for line in doc.split(". "):
        log_score = kenlm_model.score(line)
        length = len(line.split()) + 1
        doc_log_score += log_score
        doc_length += length
    return self.pp(doc_log_score, doc_length)

  def get_ontology(self):
    ontology = {}
    synonyms = {} if not hasattr(self, 'synonyms') else self.synonyms
    for key, val in synonyms.items():
      ontology[val] = ontology.get(val, []) + [key]
    return ontology

  #TODO: option to do ngram2weight, ontology and synonyms in lowercase
  #TODO: hiearhical clustering
  def create_ontology_and_synonyms(self, file_name, synonyms=None, stopword=None, ngram2weight=None, words_per_ontology_cluster = 10, kmeans_batch_size=1024, epoch = 10):
    if synonyms is None: synonyms = {} if not hasattr(self, 'synonyms') else self.synonyms
    if ngram2weight is None: ngram2weight = {} if not hasattr(self, 'ngram2weight') else self.ngram2weight    
    if stopword is None: stopword = {} if not hasattr(self, 'stopword') else self.stopword
    old_synonyms = synonyms 
    model = fasttext.train_unsupervised(file_name, epoch=epoch)
    terms = model.get_words()
    true_k=int(len(terms)/words_per_ontology_cluster)
    x=np.vstack([model.get_word_vector(term) for term in model.get_words()])
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                                init_size=max(true_k*3,1000), batch_size=kmeans_batch_size).fit(x)
    ontology = {}
    for term, label in zip(terms, km.labels_):
        ontology[label] = ontology.get(label, [])+[term]
    synonyms = {}
    for key, vals in ontology.items():
      items = [v for v in vals if "_" in v]
      if len(items) > 1:
        old_syn_upper =  [old_synonyms[v] for v in vals if "_" in v and v in old_synonyms and old_synonyms[v][0].upper() == old_synonyms[v][0]]
        old_syn_lower = [old_synonyms[v] for v in vals if "_" in v and v in old_synonyms and old_synonyms[v][0].upper() != old_synonyms[v][0]]
        items_upper_case = []
        if old_syn_upper:
          old_syn_upper =  Counter(old_syn_upper)
          syn_label = old_syn_upper.most_common(1)[0][0]
          items_upper_case = [v for v in items if (old_synonyms.get(v) == syn_label) or (old_synonyms.get(v) is None and v[0].upper() == v[0])]
          for v in copy.copy(items_upper_case):
            for v2 in items:
              if old_synonyms.get(v)  is None and (v in v2 or v2 in v):
                items_upper_case.append(v2)
          items_upper_case = list(set(items_upper_case))
          if len(items_upper_case) > 1:
            for word in items_upper_case:
              synonyms[word] = syn_label     
        if old_syn_lower: 
          old_syn_lower =  Counter(old_syn_lower)
          syn_label = old_syn_lower.most_common(1)[0][0]
          items = [v for v in items if old_synonyms.get(v) in (None, syn_label) and v not in items_upper_case]
          if len(items) > 1:
            if len(items) > 1:            
              for word in items:
                synonyms[word] = syn_label     
        
        if not old_syn_upper and not old_syn_lower:
          items_upper_case = [v for v in items if v[0].upper() == v[0]]
          for v in copy.copy(items_upper_case):
            for v2 in items:
              if v in v2 or v2 in v:
                items_upper_case.append(v2)
          items_upper_case = list(set(items_upper_case))
          if len(items_upper_case)  > 1:
            items_upper_case.sort(key=lambda a: ngram2weight.get(a, len(a)))
            syn_label = items_upper_case[0]
            for word in items_upper_case:
              synonyms[word] = syn_label
            items = [v for v in items if v not in items_upper_case]
          if len(items) > 1:
            items.sort(key=lambda a: ngram2weight.get(a, len(a)))
            syn_label = [a for a in items if a[0].lower() == a[0]][0]
            for word in items:
              synonyms[word] = syn_label
      items = [v for v in vals if "_" not in v]
      if len(items) > 1:
        items.sort(key=lambda a: ngram2weight.get(a, len(a)))
        stopwords_only = [a for a in items if a in stopword or a in stopwords_set]
        if stopwords_only: 
          label = stopwords_only[0]
          for word in stopwords_only:
              synonyms[word] = label
        not_stopwords = [a for a in items if a not in stopword and a not in stopwords_set]
        if not_stopwords: 
          label = not_stopwords[0]
          for word in not_stopwords:
              synonyms[word] = label
    for word, key in old_synonyms.items():
      synonyms[word] = key
      synonyms[key] = key
    return synonyms
 

  def tokenize(self, doc, min_compound_weight=0,  compound=None, ngram2weight=None, synonyms=None, use_synonyms=False):
    if synonyms is None: synonyms = {} if not hasattr(self, 'synonyms') else self.synonyms
    if ngram2weight is None: ngram2weight = {} if not hasattr(self, 'ngram2weight') else self.ngram2weight    
    if compound is None: compound = {} if not hasattr(self, 'compound') else self.compound
    if not use_synonyms: synonyms = {} 
    doc = [synonyms.get(d,d) for d in doc.split(" ") if d.strip()]
    len_doc = len(doc)
    for i in range(len_doc-1):
        if doc[i] is None: continue
                
        wordArr = doc[i].strip("_").replace("__", "_").split("_")
        if wordArr[0] in compound:
          max_compound_len = compound[wordArr[0]]
          for j in range(min(len_doc, i+max_compound_len), i+1, -1):
            word = ("_".join(doc[i:j])).strip("_").replace("__", "_")
            wordArr = word.split("_")
            if len(wordArr) <= max_compound_len and word in ngram2weight and ngram2weight.get(word, 0) >= min_compound_weight:
              old_word = word
              doc[j-1] = synonyms.get(word, word).strip("_").replace("__", "_")
              #if old_word != doc[j-1]: print (old_word, doc[j-1])
              for k in range(i, j-1):
                  doc[k] = None
              break
    return (" ".join([d for d in doc if d]))


  # creating tokenizer with a kenlm model as well as getting ngram weighted by the language modeling weights (not the counts) of the words
  # we can run this in incremental mode or batched mode (just concatenate all the files togehter)
  #TODO: To save memory, save away the __tmp__.arpa file at each iteration (sorted label), and re-read in the cumulative arpa file while processing the new arpa file. 
  def create_tokenizer(self, project_name, files, unigram=None,  lmplz_loc="./riverbed/bin/lmplz", stopword_max_len=10, num_stopwords=75, max_ngram_size=25, \
                non_words = "،♪↓↑→←━\₨₡€¥£¢¤™®©¶§←«»⊥∀⇒⇔√­­♣️♥️♠️♦️‘’¿*’-ツ¯‿─★┌┴└┐▒∎µ•●°。¦¬≥≤±≠¡×÷¨´:।`~�_“”/|!~@#$%^&*•()【】[]{}-_+–=<>·;…?:.,\'\"", kmeans_batch_size=1024,\
                min_compound_weight=1.0, do_final_tokenize=True, stopword=None, min_num_words=5, do_collapse_values=True, do_tokenize=True, use_synonyms=False):
      #TODO, strip non_words
      
      ngram2weight =self.ngram2weight = {} if not hasattr(self, 'ngram2weight') else self.ngram2weight
      compound = self.compound = {} if not hasattr(self, 'compound') else self.compound
      synonyms = self.synonyms = {} if not hasattr(self, 'synonyms') else self.synonyms
      ontology= self.ontology = {} if not hasattr(self, 'ontology') else self.ontology
      stopword = self.stopword = {} if not hasattr(self, 'stopword') else self.stopword
    
      if lmplz_loc != "./riverbed/bin/lmplz" and not os.path.exists("./lmplz"):
        os.system(f"cp {lmplz_loc} ./lmplz")
        lmplz = "./lmplz"
      else:
        lmplz = lmplz_loc
      os.system(f"chmod u+x {lmplz}")
      if unigram is None: unigram = {}
      if ngram2weight:
        for word in ngram2weight.keys():
          if "_" not in word: unigram[word] = min(unigram.get(word,0), ngram2weight[word])
      arpa = {}
      if os.path.exists(f"{project_name}.arpa"):
        with open(f"{project_name}.arpa", "rb") as af:
          for line in af:
            line = line.decode().strip().split("\t")
            if len(line) > 1:
              arpa[line[1]] = min(float(line[0]), arpa.get(line[1], 0))
      #TODO, we should try to create consolidated files of around 1GB to get enough information in the arpa files
      for doc_id, file_name in enumerate(files):
        if not do_tokenize: 
          num_iter = 1
        else:
          num_iter = max(1,int(max_ngram_size/(5 *(doc_id+1))))
        #we can repeatdly run the below to get long ngrams
        #after we tokenize for ngram and replace with words with underscores (the_projected_revenue) at each step, we redo the ngram
        for times in range(num_iter):
            if times == 0:
              os.system(f"cp {file_name} __tmp__{file_name}")
            print (f"iter {file_name}", times)
            if ngram2weight:
              with open(f"__tmp__2_{file_name}", "w", encoding="utf8") as tmp2:
                with open(f"__tmp__{file_name}", "r") as f:
                  for l in f:
                    l = self.tokenize(l.strip(),  min_compound_weight=min_compound_weight, compound=compound, ngram2weight=ngram2weight, synonyms=synonyms, use_synonyms=use_synonyms)
                    if do_final_tokenize and times == num_iter-1:
                      l = self.tokenize(l.strip(), min_compound_weight=0, compound=compound, ngram2weight=ngram2weight,  synonyms=synonyms, use_synonyms=use_synonyms)
                    tmp2.write(l+"\n")  
              os.system(f"mv __tmp__2_{file_name} __tmp__{file_name}")
              if use_synonyms: synonyms = self.create_ontology_and_synonyms(f"__tmp__{file_name}", stopword=stopword, ngram2weight=ngram2weight, synonyms=synonyms, kmeans_batch_size=kmeans_batch_size)     
            if do_collapse_values:
              os.system(f"./{lmplz} --collapse_values  --discount_fallback  --skip_symbols -o 5 --prune {min_num_words}  --arpa {file_name}.arpa <  __tmp__{file_name}") ##
            else:
              os.system(f"./{lmplz}  --discount_fallback  --skip_symbols -o 5 --prune {min_num_words}  --arpa {file_name}.arpa <  __tmp__{file_name}") ##
            do_ngram = False
            with open(f"{file_name}.arpa", "rb") as f:    
              for line in  f: 
                line = line.decode().strip()
                if not line: 
                  continue
                if line.startswith("\\1-grams:"):
                  do_ngram = True
                elif do_ngram:
                  line = line.split("\t")
                  if len(line) > 1:
                    arpa[line[1]] = min(float(line[0]), arpa.get(line[1], 100))
                  #print (line)
                  try:
                    weight = float(line[0])
                  except:
                    continue
                  weight = math.exp(weight)
                  line = line[1]
                  if not line: continue
                  line = line.split()
                  if [l for l in line if l in non_words or l in ('<unk>', '<s>', '</s>')]: continue
                  word = "_".join(line)
                  wordArr = word.split("_")
                  if wordArr[0]  in ('<unk>', '<s>', '</s>', ''):
                    wordArr = wordArr[1:]
                  if wordArr[-1]  in ('<unk>', '<s>', '</s>', ''):
                    wordArr = wordArr[:-1]
                  if wordArr:
                    # we are prefering stopwords that starts an n-gram. 
                    if len(wordArr[0]) <= stopword_max_len:
                      sw = wordArr[0].lower()
                      unigram[sw] = min(unigram.get(sw,100), weight)
                    if weight >= min_compound_weight:
                      compound[wordArr[0]] = max(len(wordArr), compound.get(wordArr[0],0))
                    weight = weight * len(wordArr)            
                    ngram2weight[word] = min(ngram2weight.get(word, 100), weight) 
        if do_final_tokenize:
            print (f"final tokenize {file_name}", times)
            with open(f"__tmp__2_{file_name}", "w", encoding="utf8") as tmp2:
                with open(f"__tmp__{file_name}", "r") as f:
                  for l in f:
                    l = self.tokenize(l.strip(), min_compound_weight=min_compound_weight, compound=compound, ngram2weight=ngram2weight, synonyms=synonyms, use_synonyms=use_synonyms)
                    l = self.tokenize(l.strip(), min_compound_weight=0, compound=compound, ngram2weight=ngram2weight, synonyms=synonyms, use_synonyms=use_synonyms)
                    tmp2.write(l+"\n")  
            os.system(f"mv __tmp__2_{file_name} __tmp__{file_name}")
        if not use_synonyms or do_final_tokenize:
            synonyms = self.create_ontology_and_synonyms(f"__tmp__{file_name}", stopword=stopword, ngram2weight=ngram2weight, synonyms=synonyms, kmeans_batch_size=kmeans_batch_size)    
      
      #ouotput the final kenlm .arpa file for calculating the perplexity
      ngram_cnt = {}
      for key in arpa.keys():
        n = key.count(" ")
        ngram_cnt[n] = ngram_cnt.get(n,[]) + [key]
      with open(f"__tmp__.arpa", "w", encoding="utf8") as tmp_arpa:
        tmp_arpa.write("\\data\\\n")
        tmp_arpa.write(f"ngram 1={len(ngram_cnt[0])}\n")
        tmp_arpa.write(f"ngram 2={len(ngram_cnt[1])}\n")
        tmp_arpa.write(f"ngram 3={len(ngram_cnt[2])}\n")
        tmp_arpa.write(f"ngram 4={len(ngram_cnt[3])}\n")
        tmp_arpa.write(f"ngram 5={len(ngram_cnt[4])}\n")
        for i in range(5):
          tmp_arpa.write("\n")
          j =i+1
          tmp_arpa.write(f"\\{j}-grams:\n")
          for dat in ngram_cnt[i]:
            if arpa[dat] > 0:
              arpa[dat] =  0
            tmp_arpa.write(f"{arpa[dat]}\t{dat}\t0\n")
        tmp_arpa.write("\n\\end\\\n\n")
      os.system(f"mv __tmp__.arpa {project_name}.arpa")
      top_stopword={} 
      #TODO, cleamnup tmp files
      if unigram:
          stopword_list = [l for l in unigram.items() if len(l[0]) > 0]
          stopword_list.sort(key=lambda a: a[1])
          len_stopword_list = len(stopword_list)
          top_stopword = stopword_list[:min(len_stopword_list, num_stopwords)] #+ \

      for word, weight in top_stopword:
        stopword[word] = min(stopword.get(word, 100), weight)
      self.ngram2weight, self.compound, self.synonyms, self.stopword, self.ontology = ngram2weight, compound, synonyms, stopword, ontology 
      self.kenlm_model = kenlm.LanguageModel(f"{project_name}.arpa") 
      return {'ngram2weight':ngram2weight, 'compound': compound, 'synonyms': synonyms, 'stopword': stopword,  'kenlm_model': self.kenlm_model} 

  ################
  # code for doing labeling of spans of text with different features, including clustering
  # assumes each batch is NOT shuffeled.

  @staticmethod
  def dateutil_parse_ext(text):
    try: 
      int(text.strip())
      return None
    except:
      pass
    try:
      text = text.replace("10-K", "")
      ret= dateutil_parse(text.replace("-", " "), fuzzy_with_tokens=True)
      if type(ret) is tuple: ret = ret[0]
      return ret.strftime('%x').strip()
    except:
      return None

  def intro_with_date(self, span):
    text, position, ents = span['text'], span['position'], span['ents']
    if position < 0.05 and text.strip() and (len(text) < 50 and text[0] not in "0123456789" and text[0] == text[0].upper() and text.split()[-1][0] == text.split()[-1][0].upper()):
      date = [e[0] for e in ents if e[1] == 'DATE']
      if date: 
        date = date[0]
        date = self.dateutil_parse_ext(date)
      if  date: 
        return 'intro: date of '+ date +"; "+text + " || "
      else:
        return 'intro: ' +text + " || "

  def section_with_date(self, span):
    text, position, ents = span['text'], span['position'], span['ents']
    if  position >= 0.05 and position < 0.95 and text.strip() and (len(text) < 50 and text[0] not in "0123456789" and text[0] == text[0].upper() and text.split()[-1][0] == text.split()[-1][0].upper()):
      date = [e[0] for e in ents if e[1] == 'DATE']
      if date: 
        date = date[0]
        date = self.dateutil_parse_ext(date)
      if  date: 
        return 'section: date of '+ date +"; "+text + " || "
      else:
        return  'section: ' +text + " || "
    return None

  def conclusion_with_date(self, span):
    text, position, ents = span['text'], span['position'], span['ents']
    if  position >= 0.95 and text.strip() and (len(text) < 50 and text[0] not in "0123456789" and text[0] == text[0].upper() and text.split()[-1][0] == text.split()[-1][0].upper()):
      date = [e[0] for e in ents if e[1] == 'DATE']
      if date: 
        date = date[0]
        date = self.dateutil_parse_ext(date)
      if  date: 
        return 'conclusion: date of '+ date +"; "+text + " || "
      else:
        return 'conclusion: ' +text + " || "
    return None

  RELATIVE_LOW = 0
  RELATIVE_MEDIUM = 1
  RELATIVE_HIGH= 2
  # for extracting a prefix for a segment of text. a segment can contain multiple spans.
  default_prefix_extractors = [
      ('intro_with_date', intro_with_date), \
      ('section_with_date', section_with_date), \
      ('conclusion_with_date', conclusion_with_date) \
      ]

  # for feature extraction on a single span and potentially between spans in a series. 
  # tuples of (feature_label, lower_band, upper_band, extractor). assumes prefix extraction has occured.
  # returns data which can be used to store in the feature_label for a span. if upper_band and lower_band are set, then an additional label X_level stores
  # the relative level label as well.
  #
  #TODO: other potential features include similarity of embedding from its cluster centroid
  #compound words %
  #stopwords %
  #tf-idf weight
  
  default_span_level_feature_extractors = [
      ('perplexity', .5, 1.5, lambda self, span: 0.0 if self.kenlm_model is None else self.get_perplexity(span['tokenized_text'])),
      ('prefix', None, None, lambda self, span: "" if " || " not in span['text'] else  span['text'].split(" || ", 1)[0].strip()),
      ('date', None, None, lambda self, span: "" if " || " not in span['text'] else span['text'].split(" || ")[0].split(":")[-1].split("date of")[-1].strip("; ")), 
  ]

  # for labeling the spans in the batch. assumes feature extractions above. (span_label, snorkel_labling_lfs, snorkel_label_cardinality, snorkel_epochs)
  default_lfs = []

  #Mean Pooling - Take attention mask into account for correct averaging
  #TODO, mask out the prefix for data that isn't the first portion of a prefixed text.
  @staticmethod
  def mean_pooling(model_output, attention_mask):
    with torch.no_grad():
      token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
      input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
      return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

  # the similarity models sometimes put too much weight on proper names, etc. but we might want to cluster by general concepts
  # such as change of control, regulatory actions, etc. The proper names themselves can be collapsed to one canonical form. 
  # Similarly, we want similar concepts (e.g., compound words) to cluster to one canonical form.
  # we do this by collapsing to an NER label and/or creating a synonym map from compound words to known words. See create_ontology_and_synonyms
  # and we use that data to simplify the sentence here.  
  # TODO: have an option NOT to simplify the prefix. 
  def simplify_text(self, text, ents, ner_to_simplify=()):
    ngram2weight, compound, synonyms  = self.ngram2weight, self.compound, self.synonyms
    if not ner_to_simplify and not synonyms and not ents: return text, ents
    # first tokenize
    if compound or synonyms or ngram2weight:
      text = self.tokenize(text)  
    ents2 = []
    # as a fallback, we will use spacy for other ners.  
    for idx, ent in enumerate(ents):
        entity, label = ent
        if f"@#@{idx}@#@" not in text: continue
        ents2.append((entity, label,  text.count(f"@#@{idx}@#@")))
        if label in ner_to_simplify:   
          if label == 'ORG':
            text = text.replace(f"@#@{idx}@#@", 'The Organization')
          elif label == 'PERSON':
            text = text.replace(f"@#@{idx}@#@", 'The Person')
          elif label == 'FAC':
            text = text.replace(f"@#@{idx}@#@", 'The Facility')
          elif label in ('GPE', 'LOC'):
            text = text.replace(f"@#@{idx}@#@", 'The Location')
          elif label in ('DATE', ):
            text = text.replace(f"@#@{idx}@#@", 'The Date')
          elif label in ('LAW', ):
            text = text.replace(f"@#@{idx}@#@", 'The Law')  
          elif label in ('MONEY', ):
            text = text.replace(f"@#@{idx}@#@", 'The Amount')
          else:
            text = text.replace(f"@#@{idx}@#@", entity)
        else:
          text = text.replace(f"@#@{idx}@#@", entity)              
    for _ in range(3):
      text = text.replace("The Person and The Person", "The Person").replace("The Person The Person", "The Person").replace("The Person, The Person", "The Person")
      text = text.replace("The Facility and The Facility", "The Facility").replace("The Facility The Facility", "The Facility").replace("The Facility, The Facility", "The Facility")
      text = text.replace("The Organization and The Organization", "The Organization").replace("The Organization The Organization", "The Organization").replace("The Organization, The Organization", "The Organization")
      text = text.replace("The Location and The Location", "The Location").replace("The Location The Location", "The Location").replace("The Location, The Location", "The Location")
      text = text.replace("The Date and The Date", "The Date").replace("The Date The Date", "The Date").replace("The Date, The Date", "The Date")
      text = text.replace("The Law and The Law", "The Law").replace("The Law The Law", "The Law").replace("The Law, The Law", "The Law")
      text = text.replace("The Amount and The Amount", "The Amount").replace("The Amount The Amount", "The Amount").replace("The Amount, The Amount", "The Amount")
      
    return text, ents2

  def create_spans(self, curr_file_size, batch, text_span_size=1000, ner_to_simplify=()):
      ngram2weight, compound, synonyms  = self.ngram2weight, self.compound, self.synonyms
      batch2 = []
      for idx, span in enumerate(batch):
        file_name, curr_lineno, ents, text  = span['file_name'], span['lineno'], span['ents'], span['text']
        for idx, ent in enumerate(ents):
          text = text.replace(ent[0], f' @#@{idx}@#@ ')
        # we tokenize to make ngram words underlined, so that we don't split a span in the middle of a ngram.
        text  = self.tokenize(text, use_synonyms=False) 
        len_text = len(text)
        prefix = ""
        if "||" in text:
          prefix, _ = text.split("||",1)
          prefix = prefix.strip()
        offset = 0
        while offset < len_text:
          max_rng  = min(len_text, offset+text_span_size+1)
          if text[max_rng-1] != ' ':
            if ' ' in text[max_rng:]:
              max_rng = max_rng + text[max_rng:].index(' ')
            else:
              max_rng = len_text
          if prefix and offset > 0:
            text2 = prefix +" || ... " + text[offset:max_rng].strip().replace("_", " ").replace("  ", " ").replace("  ", " ")
          else:
            text2 = text[offset:max_rng].strip().replace("_", " ").replace("  ", " ").replace("  ", " ")
          tokenized_text, ents2 = self.simplify_text(text2, ents, ner_to_simplify) 
          sub_span = copy.deepcopy(span)
          sub_span['position'] += offset/curr_file_size
          sub_span['offset'] = offset
          sub_span['text'] = text2
          sub_span['tokenized_text'] = tokenized_text 
          sub_span['ents'] = ents2
          batch2.append(sub_span)
          offset = max_rng

      return batch2

  #compute features and embeddings in one batch.
  def create_embeds_and_features_one_batch(self, curr_file_size, jsonl_file_idx, span2jsonl_file_idx, batch, cluster_batch, cluster_vecs, embed_batch_size=100, text_span_size=1000, running_features_per_label={}, ner_to_simplify=(), span_level_feature_extractors=default_span_level_feature_extractors, running_features_size=100):
    ngram2weight, compound, synonyms, kenlm_model  = self.ngram2weight, self.compound, self.synonyms, self.kenlm_model
    batch = self.create_spans(curr_file_size, batch, text_span_size=text_span_size, ner_to_simplify=ner_to_simplify)
    feature_labels = []
    features = []
    relative_levels = []
    for feature_label, lower_band, upper_band, extractor in span_level_feature_extractors:
      need_to_high = True
      need_to_low = True
      need_to_medium = True
      prior_change = -1
      feature_labels.append(feature_label)
      features.append([])
      relative_levels.append([])
      features_per_label = features[-1]
      relative_level_per_label = relative_levels[-1]
      running_features = running_features_per_label[feature_label] = running_features_per_label.get(feature_label, [])
      if lower_band is not None:
        if len(running_features) < running_features_size:
          for span in batch:
            p = extractor(self, span)
            running_features.append(p)
            if len(running_features) >= running_features_size:
                break
        stdv = statistics.stdev(running_features)
        mn = statistics.mean (running_features)
        relative_label = self.RELATIVE_LOW
      for idx, span in enumerate(batch):
        p = extractor(self, span)
        features_per_label.append(p)
        if lower_band is not None:
          running_features.append(p)
          if len(running_features) >= running_features_size:    
            stdv = statistics.stdev(running_features)
            mn = statistics.mean (running_features)
          if len(running_features) > running_features_size:
            running_features.pop()    
          if abs(p-mn) >= stdv*upper_band and need_to_high:
            relative_label = self.RELATIVE_HIGH
            prior_change = idx
            need_to_high = False
            need_to_low = True
            need_to_medium = True
          elif  abs(p-mn) < stdv*upper_band and abs(p-mn) > stdv*lower_band  and need_to_medium:
            relative_label = self.RELATIVE_MEDIUM
            prior_change = idx
            need_to_high = True
            need_to_low = True
            need_to_medium = False
          elif abs(p-mn) <= stdv*lower_band and need_to_low:
            relative_label = self.RELATIVE_LOW
            prior_change = idx
            need_to_high = False
            need_to_low = True
            need_to_medium = False
          running_features.append(p)
          relative_level_per_label.append(relative_label)
    
    for idx, span in enumerate(batch):
      file_name, curr_lineno, offset = span['file_name'], span['lineno'], span['offset']
      span['idx']= jsonl_file_idx
      span['cluster_label']= None
      span['cluster_label_before']= None
      span['cluster_label_after']= None
      for feature_label, features_per_label, relative_level_per_label in  zip(feature_labels, features, relative_levels):
        span[feature_label] = features_per_label[idx]
        if relative_level_per_label: span[feature_label+"_level"] = relative_level_per_label[idx]
      ent_cnts = Counter(v[1].lower()+"_cnt" for v in span['ents'])
      for feature_label, cnt in ent_cnts.items():
        span[feature_label] = cnt
      cluster_batch.append(span)
      span2jsonl_file_idx[(file_name, curr_lineno, offset)] = jsonl_file_idx
      jsonl_file_idx += 1

    for rng in range(0, len(batch), embed_batch_size):
      max_rng = min(len(batch), rng+embed_batch_size)
      #TODO: save away the vectors (in a mmap file) to enable ANN search of batches 
      toks = sim_tokenizer([a['tokenized_text'].replace("_", " ") for a in batch[rng:max_rng]], padding=True, truncation=True, return_tensors="pt").to(device)
      with torch.no_grad():
        dat = sim_model(**toks)
      dat = self.mean_pooling(dat, toks.attention_mask).cpu().numpy()
      if cluster_vecs is None:
        cluster_vecs = dat
      else:
        cluster_vecs = np.vstack([cluster_vecs, dat])

    return cluster_batch, cluster_vecs, span2jsonl_file_idx, jsonl_file_idx

  def create_cluster_for_spans(self, batch_id_prefix, cluster_batch, cluster_vecs, clusters, span2cluster_label,  span_per_cluster=20, kmeans_batch_size=1000000, ):
      true_k=int(len(cluster_batch)/span_per_cluster)
      km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                                  init_size=max(true_k*3,1000), batch_size=kmeans_batch_size).fit(cluster_vecs)
      new_cluster = {}
      for item, label in zip(cluster_batch, km.labels_):
        span = (item['file_name'], item['lineno'], item['offset'],)
        label = batch_id_prefix+str(label)
        new_cluster[label] = new_cluster.get(label, [])+[span]
      if not clusters: 
        clusters = new_cluster
        for label, items in clusters.items():
          for span in items:
            span2cluster_label[span] = label
      else:
        for label, items in new_cluster.items():
          cluster_labels = [span2cluster_label[span] for span in items if span in span2cluster_label]
          items2 = [span for span in items if span not in span2cluster_label]
          if cluster_labels:
            most_common = Counter(cluster_labels).most_common(1)[0]
            if most_common[1] >= 2: #if two or more of the span in a cluster has already been labeled, use that label for the rest of the spans
              label = most_common[0]
              items = [span for span in items if span2cluster_label.get(span) in (label, None)]
            else:
              items = items2
          else:
            items = items2
          for span in items:
            if span not in clusters.get(label, []):
                clusters[label] = clusters.get(label, []) + [span]
            span2cluster_label[span] = label

      return clusters, span2cluster_label 

  #we create clusters in an incremental fashion from cluster_batch
  #cluster_batch should be larger batches than embeds_and_features's batch
  def create_cluster_and_label_one_batch(self, jsonl_file, batch_id_prefix, jsonl_file_idx, jsonl_file_idx_for_curr_batch, retained_spans_per_cluster, span_lfs, cluster_batch, cluster_vecs, clusters, span2cluster_label, \
                                        span_per_cluster, kmeans_batch_size, label2tf=None, df=None, domain_stopword_set=stopwords_set, verbose_snrokel=False):
    ngram2weight  = self.ngram2weight
    clusters, span2cluster_label = self.create_cluster_for_spans(batch_id_prefix, cluster_batch, cluster_vecs, clusters, span2cluster_label, span_per_cluster=span_per_cluster, kmeans_batch_size=kmeans_batch_size)
    #all leaf nodes of the cluster are stored as a triple of (file_name, lineno, offset)
    cluster_leaf2idx = dict([((b['file_name'], b['lineno'], b['offset']), idx) for idx, b in enumerate(cluster_batch) ])
    new_cluster_batch = []
    new_cluster_vecs = []
    if label2tf is None: label2tf = {}
    if df is None: df = {}
    
    #we compute the weighted tf-idf with respect to each word in each clusters
    len_clusters = len(clusters)
    for label, values in clusters.items():  
      for item in values:
        if item in cluster_leaf2idx:
          span = cluster_batch[cluster_leaf2idx[item]]
          text = span['tokenized_text']
          text = text.replace('The Organization','').replace('The_Organization','')
          text = text.replace('The Person','').replace('The_Person','')
          text = text.replace('The Facility','').replace('The_Facility','')
          text = text.replace('The Date','').replace('The_Date','')
          text = text.replace('The Law','').replace('The_Law','')
          text = text.replace('The Amount','').replace('The_Amount','')
          ents =  list(itertools.chain(*[[a[0].replace(" ", "_")]*a[-1] for a in span['ents']]))
          if span['offset'] == 0:
            if "||" in text:
              prefix, text = text.split("||",1)
              prefix = prefix.split(":")[-1].split(";")[-1].strip()
              text = prefix.split() + text.replace("(", " ( ").replace(")", " ) ").split() + ents
            else:
               text = text.replace("(", " ( ").replace(")", " ) ").split() + ents
          else:
            text = text.split("||",1)[-1].strip().split() + ents
          len_text = len(text)
          text = [a for a in text if len(a) > 1 and ("_" not in a or (a.count("_")+1 != len([b for b in a.lower().split("_") if  b in domain_stopword_set])))  and a.lower() not in domain_stopword_set and a[0].lower() in "abcdefghijklmnopqrstuvwxyz"]
          cnts = Counter(text)
          aHash = label2tf[label] =  label2tf.get(label, {})
          for word, cnt in cnts.items():
            aHash[word] = cnt/len_text
          for word in cnts.keys():
            df[word] = df.get(word,0) + 1
      
    #create a new label from the tfidf of the words in this cluster
    #TODO, see how we might save away the tf-idf info as features, then we would need to recompute the tfidf if new items are added to cluster
    label2label = {}
    for label, tf in label2tf.items():
      if label.startswith(batch_id_prefix):
        tfidf = copy.copy(tf)    
        for word in list(tfidf.keys()):
          tfidf[word]  = tfidf[word] * min(1.5, ngram2weight.get(word, 1)) * math.log(len_clusters/(1+df[word]))
        top_words2 = [a[0].lower().strip("~!@#$%^&*()<>,.:;")  for a in Counter(tfidf).most_common(min(len(tfidf), 40))]
        top_words2 = [a for a in top_words2 if a not in domain_stopword_set and ("_" not in a or (a.count("_")+1 != len([b for b in a.split("_") if  b in domain_stopword_set])))]
        top_words = []
        for t in top_words2:
          if t not in top_words:
            top_words.append(t)
        if top_words:
          if len(top_words) > 5: top_words = top_words[:5]
          label2 = ", ".join(top_words) 
          label2label[label] = label2

    for old_label, new_label in label2label.items():
      if new_label != old_label and old_label in clusters:
        a_cluster = clusters[old_label]
        del clusters[old_label]
        clusters[new_label] = a_cluster
        for item in clusters[new_label]:
          span2cluster_label[item] = new_label
    for label, values in clusters.items():          
      items = [item for item in values if item in cluster_leaf2idx]
      for item in items:
        cluster_batch[cluster_leaf2idx[item]]['cluster_label'] = label
    prior_span = None
    for span in cluster_batch:
      if prior_span is not None:
        span['cluster_label_before'] = prior_span['cluster_label']
        prior_span['cluster_label_after'] = span['cluster_label']
      prior_span = span
    for old_label, new_label in label2label.items():
      if old_label != new_label and old_label in label2tf:
        tf = label2tf[old_label]
        del  label2tf[old_label]
        label2tf[new_label] = tf

    # at this point cluster_batch should have enough data for all snorkel labeling functions
    if span_lfs:
      df_train = pd.DataFrom(cluster_batch)
      for span_label, lfs, snorkel_label_cardinality, snorkel_epochs in span_lfs:
        df_train = df_train.shuffle()    
        applier = PandasLFApplier(lfs=fs)
        L_train = applier.apply(df=df_train)
        label_model = LabelModel(cardinality=snorkel_label_cardinality, verbose=verbose_snrokel)
        label_model.fit(L_train=L_train,n_epochs=snorkel_epochs)
        for idx, label in enumerate(label_model.predict(L=L_train,tie_break_policy="abstain")):
          cluster_batch[cluster_leaf2idx[item]][span_label] = label
        # note, we only use these models once, since we are doing this in an incremental fashion.
        # we would want to create a final model by training on all re-labeled data from the jsonl file
    
    # all labeling and feature extraction is complete. now save away the batch
    for b in cluster_batch:
      if b['idx'] >= jsonl_file_idx_for_curr_batch:
        jsonl_file.write(json.dumps(b)+"\n")

    # we bootstrap the next clustering with a retained number of sample of the prior clusters so we can connect new clusters to the old clusters
    # this permits us to do a type of incremental clustering
    for key, values in clusters.items():          
      items = [item for item in values if item in cluster_leaf2idx]
      if len(items) > retained_spans_per_cluster:
        items = random.sample(items, retained_spans_per_cluster)
      new_cluster_batch.extend([cluster_batch[cluster_leaf2idx[item]] for item in items])
      new_cluster_vecs.extend([cluster_vecs[cluster_leaf2idx[item]] for item in items])
    new_cluster_vecs = np.vstack(new_cluster_vecs)
    cluster_batch , cluster_vecs = new_cluster_batch, new_cluster_vecs
    jsonl_file_idx_for_curr_batch = jsonl_file_idx
    return clusters, jsonl_file_idx_for_curr_batch, cluster_batch, cluster_vecs, span2cluster_label, label2tf, df   



  def apply_span_feature_detect_and_labeling(self, project_name, files, text_span_size=1000, max_lines_per_section=10, max_len_for_prefix=100, min_len_for_prefix=20, embed_batch_size=100, 
                                                features_batch_size = 10000000, labeling_batch_size=10000000, kmeans_batch_size=1024, \
                                                span_per_cluster= 20, retained_spans_per_cluster=5, \
                                                ner_to_simplify=(), span_level_feature_extractors=default_span_level_feature_extractors, running_features_size=100, \
                                                prefix_extractors = default_prefix_extractors, dedup=True, \
                                                span_lfs = [], verbose_snrokel=True, \
                                                batch_id_prefix = 0, seen = None, span2jsonl_file_idx = None, \
                                                clusters = None, label2tf = None, df = None, span2cluster_label = None, label_models = None, auto_create_tokenizer=True, \
                                                ):
    self.ngram2weight = {} if not hasattr(self, 'ngram2weight') else self.ngram2weight
    self.compound = {} if not hasattr(self, 'compound') else self.compound
    self.synonyms = {} if not hasattr(self, 'synonyms') else self.synonyms
    stopword = self.stopword = {} if not hasattr(self, 'stopword') else self.stopword
    
    if os.path.exists(f"{project_name}.arpa") and (not hasattr(self, 'kenlm_model') or self.kenlm_model is None):
      kenlm_model = self.kenlm_model = kenlm.LanguageModel(f"{project_name}.arpa")
    kenlm_model = self.kenlm_model if hasattr(self, 'kenlm_model') else None
    if kenlm_model is None and auto_create_tokenizer:
      self.create_tokenizer(project_name, files, )
      kenlm_model = self.kenlm_model = kenlm.LanguageModel(f"{project_name}.arpa")      
    running_features_per_label = {}
    file_name = files.pop()
    f = open(file_name) 
    domain_stopword_set = set(list(stopwords_set) + list(stopword.keys()))
    prior_line = ""
    batch = []
    curr = ""
    cluster_batch = []
    cluster_vecs = None
    curr_date = ""
    curr_position = 0
    next_position = 0
    curr_file_size = os.path.getsize(file_name)
    position = 0
    line = ""
    lineno = -1
    curr_lineno = 0

    #TODO, load the below from an config file
    if seen is None: seen = {}
    if span2jsonl_file_idx is None: span2jsonl_file_idx = {}
    if clusters is None: clusters = {}
    if label2tf is None: label2tf = {}
    if df is None: df = {}
    if span2cluster_label is None: span2cluster_label = {}
    if label_models is None: label_models = []  
    jsonl_file_idx = 0 if not span2jsonl_file_idx else max(span2jsonl_file_idx.values())
    jsonl_file_idx_for_curr_batch= jsonl_file_idx
    

    with open(f"{project_name}.jsonl", "w", encoding="utf8") as jsonl_file:
      while True:
        try:
          line = f.readline()
          if line: lineno+=1 
        except:
          line = ""
        if len(line) == 0:
          #print ("reading next")
          if curr: 
            hash_id = hash(curr)
            if not dedup or (hash_id not in seen):
                curr_ents = list(itertools.chain(*[[(e.text, e.label_)] if '||' not in e.text else [(e.text.split("||")[0].strip(), e.label_), (e.text.split("||")[-1].strip(), e.label_)] for e in spacy_nlp(curr).ents]))
                curr_ents = list(set([e for e in curr_ents if e[0]]))
                curr_ents.sort(key=lambda a: len(a[0]), reverse=True)
                batch.append({'file_name': file_name, 'lineno': curr_lineno, 'text': curr, 'ents': curr_ents, 'position':curr_position})
                seen[hash_id] = 1
          prior_line = ""
          curr = ""
          if not files: break
          file_name = files.pop()
          f = open(file_name)
          l = f.readline()
          lineno = 0
          curr_lineno = 0
          curr_date = ""
          curr_position = 0
          curr_file_size = os.path.getsize(file_name)
          position = 0
        position = next_position/curr_file_size
        next_position = next_position + len(line)+1
        line = line.strip().replace("  ", " ")
        if not line: continue
        if len(line) < min_len_for_prefix and len(line) > 0:
          prior_line = prior_line + " " + line
          continue
        line = prior_line+" " + line
        prior_line = ""
        line = line.replace("  ", " ").replace("\t", " ").strip("_ ")

        #turn the file position into a percentage
        if len(line) < max_len_for_prefix:
          ents = list(itertools.chain(*[[(e.text, e.label_)] if '||' not in e.text else [(e.text.split("||")[0].strip(), e.label_), (e.text.split("||")[-1].strip(), e.label_)] for e in spacy_nlp(line).ents]))
          ents = [e for e in ents if e[0]]
          ents = [[a[0], a[1], b] for a, b in Counter(ents).items()]
          for prefix, extract in prefix_extractors:
            extracted_text = extract(self, {'text':line, 'position':position, 'ents':ents}) 
            if extracted_text:
              line = extracted_text
              if curr: 
                curr = curr.replace(". .", ". ").replace("..", ".").replace(":.", ".")
                hash_id = hash(curr)
                if not dedup or (hash_id not in seen):
                  curr_ents = list(itertools.chain(*[[(e.text, e.label_)] if '||' not in e.text else [(e.text.split("||")[0].strip(), e.label_), (e.text.split("||")[-1].strip(), e.label_)] for e in spacy_nlp(curr).ents]))
                  curr_ents = list(set([e for e in curr_ents if e[0]]))
                  curr_ents.sort(key=lambda a: len(a[0]), reverse=True)
                  batch.append({'file_name': file_name, 'lineno': curr_lineno, 'text': curr, 'ents': curr_ents, 'position':curr_position})
                  seen[hash_id] = 1
                curr = ""
                curr_lineno = lineno
                curr_position = position
              break
        if curr: 
          curr = curr +" " + line
        else: 
          curr = line
        curr = curr.replace("  ", " ")

        # process the batches
        if len(batch) >= features_batch_size:
          cluster_batch, cluster_vecs, span2jsonl_file_idx, jsonl_file_idx = \
            self.create_embeds_and_features_one_batch(curr_file_size, jsonl_file_idx, span2jsonl_file_idx, batch, cluster_batch, cluster_vecs, embed_batch_size, text_span_size,  running_features_per_label, ner_to_simplify, span_level_feature_extractors, running_features_size)
          batch = []
        # clustering, labeling and creating snorkel model in chunks
        if cluster_batch and cluster_vecs is not None and cluster_vecs.shape[0] >= labeling_batch_size:
          batch_id_prefix += 1
          clusters, jsonl_file_idx_for_curr_batch, cluster_batch, cluster_vecs, span2cluster_label, label2tf, df  = \
            self.create_cluster_and_label_one_batch(jsonl_file, f"{batch_id_prefix}_", jsonl_file_idx, jsonl_file_idx_for_curr_batch, retained_spans_per_cluster, span_lfs, cluster_batch, cluster_vecs, clusters, span2cluster_label, \
                                        span_per_cluster, kmeans_batch_size, label2tf, df,  domain_stopword_set, verbose_snrokel)
          cluster_batch, cluster_vecs = [], None
      
      # do one last batch and finish processing if there's anything left
      if curr: 
          curr = curr.replace(". .", ". ").replace("..", ".").replace(":.", ".")
          hash_id = hash(curr)
          if not dedup or (hash_id not in seen):
            curr_ents = list(itertools.chain(*[[(e.text, e.label_)] if '||' not in e.text else [(e.text.split("||")[0].strip(), e.label_), (e.text.split("||")[-1].strip(), e.label_)] for e in spacy_nlp(curr).ents]))
            curr_ents = list(set([e for e in curr_ents if e[0]]))
            curr_ents.sort(key=lambda a: len(a[0]), reverse=True)
            batch.append({'file_name': file_name, 'lineno': curr_lineno, 'text': curr, 'ents': curr_ents, 'position':curr_position})
            seen[hash_id] = 1
          curr = ""
          curr_lineno = 0
          curr_position = position
      if batch: 
        cluster_batch, cluster_vecs, span2jsonl_file_idx, jsonl_file_idx = \
          self.create_embeds_and_features_one_batch(curr_file_size, jsonl_file_idx, span2jsonl_file_idx, batch, cluster_batch, cluster_vecs, embed_batch_size, text_span_size, running_features_per_label, ner_to_simplify, span_level_feature_extractors, running_features_size)
        batch = []
      if cluster_batch and cluster_vecs is not None:
        batch_id_prefix += 1
        self.clusters, jsonl_file_idx_for_curr_batch, cluster_batch, cluster_vecs, span2cluster_label, label2tf, df = \
            self.create_cluster_and_label_one_batch(jsonl_file, f"{batch_id_prefix}_",  jsonl_file_idx, jsonl_file_idx_for_curr_batch, retained_spans_per_cluster, span_lfs, cluster_batch, cluster_vecs, clusters, span2cluster_label, \
                                        span_per_cluster, kmeans_batch_size, label2tf, df, domain_stopword_set, verbose_snrokel )
        cluster_batch, cluster_vecs = [], None

    #now create global labeling functions based on all the labeled data
    #have an option to use a different labeling function, such as regression trees. 
    #we don't necessarily need snorkel lfs after we have labeled the dataset.

    if span_lfs:
      df_train = pd.DataFrame(f"{project_name}.jsonl").shuffle()
      for span_label, lfs, snorkel_label_cardinality, snorkel_epochs in span_lfs:
        applier = PandasLFApplier(lfs=lfs)
        L_train = applier.apply(df=df_train)
        label_models.append(span_label, LabelModel(cardinality=snorkel_label_cardinality,verbose=verbose_snrokel))
        
    return {'clusters': clusters, 'span2cluster_label': span2cluster_label, 'span2jsonl_file_idx': span2jsonl_file_idx, 'label_models': label_models, \
            'batch_id_prefix': batch_id_prefix, 'seen': seen, 'label2tf': label2tf, 'df': df,  'label_models': label_models} 

  def save_pretrained(self, project_name):
      pickle.dump(self, open(f"{project_name}.pickle", "wb"))
    
  @staticmethod
  def from_pretrained(project_name):
      self = pickle.load(open(f"{project_name}.pickle", "rb"))
      return self

