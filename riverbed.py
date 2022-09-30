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

# with region labels, we can do things like tf-idf of tokens, and then do a mean of the tf-idf of a span. A span with high avg tf-idf means it is interesting or relevant. 

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
from .utils import *
from .searcher import *

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

try:
  if minilm_model is not None: 
    pass
except:
   labse_tokenizer= labse_model=  clip_processor = minilm_tokenizer= clip_model= minilm_model= spacy_nlp= stopwords_set = None

# The Riverbed code includes a RiverbedTokenizer, RiverbedModel and RiverbedDocumenProcessor for information retrieval processing. 
# The tokenizer stores the stopwords, compound, token2weight and synonyms data structure.
# the model stores a copy of synonyms and the kenlm model and a searcher to search token embeddings/ontology.


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
      pickle.dump(self, open(f"{tokenizer_name}/{tokenizer_name}.pickle", "wb"))
    
  @staticmethod
  def from_pretrained(tokenizer_name):
      self = pickle.load(open(f"{tokenizer_name}/{tokenizer_name}.pickle", "rb"))
      return self

#################################################################################
#MODEL CODE
#this class is used to model whether a segment of text is similar to training data (via perplexity using kenlm). 
#this class also allows us to analyze the words used in the training data. 
#it is used in conjunction with the RiverbedTokenizer to find compound words in a sentence. 
class RiverbedModel(nn.Module):

  def __init__(self):
   super().__init__()
   global labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set 
   labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set = init_models()
   self.searcher = Searcher()
   self.tokenizer = None
   self.synonyms = None
   self.clusters = None
   self.mmap_file = ""
  
  def search(self, *args, **kwargs):
    self.searcher(*args, **kwargs)
    
  # get the downsampled sentence embeddings. can be used to train the downsampler(s).
  def forward(self, *args, **kwargs):
    if 'text' in kwargs:
      text = kwargs['text']
      # we tokenize using the RiverbedTokenizer to take into account n-grams
      text = self.tokenizer.tokenize(text) 
      #we create the downsample sentence embeding with mean of the ngram and the non-ngram 
      #sentences. 
      # we assume that at this point we have already added new custom embeddings/tokens in the underlying tokenizer
      # not happy -> no_happy, which is similar to "sad", "angry", "unhappy"
      #then we tokenize using the embedder tokenizer
    dat = self.searcher(*args, **kwargs)
    return dat
  

  @staticmethod
  def _pp(log_score, length):
    return float((10.0 ** (-log_score / length)))

  def get_perplexity(self,  doc, kenlm_model=None):
    if kenlm_model is None: kenlm_model = {} if not hasattr(self, 'kenlm_model') else self.kenlm_model
    doc_log_score = doc_length = 0
    doc = doc.replace("\n", " ")
    for line in doc.split(". "):
        if "_" in line:
          log_score = min(kenlm_model.score(line),kenlm_model.score(line.replace("_", " ")))
        else:
          log_score = kenlm_model.score(line)
        length = len(line.split()) + 1
        doc_log_score += log_score
        doc_length += length
    return self._pp(doc_log_score, doc_length)
  
  #NOTE: we use the '¶' in front of a token to designate a token is a parent in an ontology. 
  #the level of the ontology is determined by the number of '¶'.
  #More '¶' means higher up the ontology (closer to the root nodes/top_parents). Leaf tokens have no '¶'
  def get_ontology(self, synonyms=None):
    ontology = {}
    if synonyms is None:
      synonyms = {} if not hasattr(self, 'synonyms') else self.synonyms
    for key, val in synonyms.items():
      ontology[val] = ontology.get(val, []) + [key]
    return ontology

  # find the top parent nodes that have no parents
  def get_top_parents(self, synonyms=None):
    top_parents = []
    if synonyms is None:
      synonyms = {} if not hasattr(self, 'synonyms') else self.synonyms
    
    ontology = self.get_ontology(synonyms)
    for parent in ontology:
      if parent not in synonyms:
        top_parents.append(parent)
    return top_parents

  # cluster one batch of tokens/embeddings, assuming some tokens have already been clustered
  def _cluster_one_batch(self, cluster_embeddings, idxs, terms2, true_k, synonyms=None, stopwords=None, token2weight=None, min_incremental_cluster_overlap=2 ):
    global device
    #print ('cluster_one_batch', len(idxs))
    if synonyms is None: synonyms = {} if not hasattr(self, 'synonyms') else self.synonyms
    if token2weight is None: token2weight = {} if not hasattr(self, 'token2weight') else self.token2weight    
    if stopwords is None: stopwords = {} if not hasattr(self, 'stopwords') else self.stopwords
    if device == 'cuda':
      kmeans = KMeans(n_clusters=true_k, mode='cosine')
      km_labels = kmeans.fit_predict(torch.from_numpy(cluster_embeddings[idxs]).to(device))
      km_labels = [l.item() for l in km_labels.cpu()]
    else:
      km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                                          init_size=max(true_k*3,1000), batch_size=1024).fit(cluster_embeddings[idxs])
      km_labels = km.labels_
    ontology = {}
    #print (true_k)
    for term, label in zip(terms2, km_labels):
      ontology[label] = ontology.get(label, [])+[term]
    #print (ontology)
    for key, vals in ontology.items():
      items = [v for v in vals if "_" in v and not v.startswith('¶')]
      if len(items) > 1:
          old_syn_upper =  [synonyms[v] for v in vals if "_" in v and v in synonyms and synonyms[v][1].upper() == synonyms[v][0]]
          old_syn_lower = [synonyms[v] for v in vals if "_" in v and v in synonyms and synonyms[v][1].upper() != synonyms[v][0]]
          items_upper_case = []
          if old_syn_upper:
            old_syn_upper =  Counter(old_syn_upper)
            if old_syn_upper.most_common(1)[0]  >= min_incremental_cluster_overlap:
              syn_label = old_syn_upper.most_common(1)[0][0]
              items_upper_case = [v for v in items if (synonyms.get(v) == syn_label) or (synonyms.get(v) is None and v[0].upper() == v[0])]
              for v in copy.copy(items_upper_case):
                for v2 in items:
                  if synonyms.get(v)  is None and (v in v2 or v2 in v):
                    items_upper_case.append(v2)
              items_upper_case = list(set(items_upper_case))
              if len(items_upper_case) > 1:
                for token in items_upper_case:
                  synonyms[token] = syn_label
              else:
                old_syn_upper = None
            else:
              old_syn_upper = None
          if old_syn_lower: 
            old_syn_lower =  Counter(old_syn_lower)
            if old_syn_lower.most_common(1)[0][1] >= min_incremental_cluster_overlap:
              syn_label = old_syn_lower.most_common(1)[0][0]
              items = [v for v in items if synonyms.get(v) in (None, syn_label) and v not in items_upper_case]
              if len(items) > 1:
                for token in items:
                  synonyms[token] = syn_label
              else:
                old_syn_lower = None
            else:
              old_syn_lower = None
          if not old_syn_upper and not old_syn_lower:
            items_upper_case = [v for v in items if v[0].upper() == v[0]]
            for v in copy.copy(items_upper_case):
              for v2 in items:
                if v in v2 or v2 in v:
                  items_upper_case.append(v2)
            items_upper_case = list(set(items_upper_case))
            if len(items_upper_case)  > 1:
              items_upper_case.sort(key=lambda a: token2weight.get(a, len(a)))
              syn_label = '¶'+items_upper_case[0]
              for token in items_upper_case:
                synonyms[token] = syn_label
              items = [v for v in items if v not in items_upper_case]
            if len(items) > 1:
              items.sort(key=lambda a: token2weight.get(a, len(a)))
              syn_label = '¶'+[a for a in items if a[0].lower() == a[0]][0]
              for token in items:
                synonyms[token] = syn_label
      items = [v for v in vals if v not in synonyms]
      if len(items) > 1:
        items.sort(key=lambda a: token2weight.get(a, len(a)))
        parents_only = [a for a in items if a.startswith('¶')]
        if parents_only: 
          label = '¶'+parents_only[0]
          for token in parents_only:
              synonyms[token] = label        
        stopwords_only = [a for a in items if a.lower() in stopwords or a in stopwords_set]
        if stopwords_only: 
          label = '¶'+stopwords_only[0]
          for token in stopwords_only:
              synonyms[token] = label
        not_stopwords = [a for a in items if a.lower() not in stopwords and a not in stopwords_set]
        if not_stopwords: 
          label = '¶'+not_stopwords[0]
          for token in not_stopwords:
              synonyms[token] = label
    return synonyms

  # create a hiearchical structure given leaves that have already been clustered
  def _create_ontology(self, model_name, synonyms=None, stopwords=None, token2weight=None, token2idx=None, \
                      kmeans_batch_size=50000, epoch = 10, embed_batch_size=7000, min_prev_ids=10000, embedder="minilm", \
                      max_ontology_depth=4, max_cluster_size=100, recluster_type="individual", min_incremental_cluster_overlap=2):
    if synonyms is None: synonyms = {} if not hasattr(self, 'synonyms') else self.synonyms
    if token2weight is None: token2weight = {} if not hasattr(self, 'token2weight') else self.token2weight    
    if stopwords is None: stopwords = {} if not hasattr(self, 'stopwords') else self.stopwords
    # assumes token2weight is an ordered dict, ordered roughly by frequency
    if not token2weight: return synonyms
    if embedder == "clip":
      embed_dim = clip_model.config.text_config.hidden_size
    elif embedder == "minilm":
      embed_dim = minilm_model.config.hidden_size
    elif embedder == "labse":
      embed_dim = labse_model.config.hidden_size      
    cluster_embeddings = np_memmap(self.mmap_file, shape=[len(token2weight), embed_dim])
    tokens = list(token2weight.keys())
    if token2idx is None:
      token2idx = dict([(token, idx) for idx, token in enumerate(tokens)])
    print ('creating ontology')
    for level in range(max_ontology_depth): 
      ontology = self.get_ontology(synonyms)
      parents = [parent for parent in ontology.keys() if len(parent) - len(parent.lstrip('¶')) == level + 1]
      if len(parents) < max_cluster_size: break
      idxs = [token2idx[parent.lstrip('¶')] for parent in parents]
      true_k = int(max(2, int(len(parents)/max_cluster_size)))
      synonyms = self._cluster_one_batch(cluster_embeddings, idxs, parents, true_k, synonyms=synonyms, stopwords=stopwords, token2weight=token2weight, )
      ontology = self.get_ontology(synonyms)
      cluster_batch=[]
      parents_parent = [parent for parent in ontology.keys() if len(parent) - len(parent.lstrip('¶')) == level + 2]
      for parent in parents_parent: 
        cluster = ontology[parent]
        if len(cluster) > max_cluster_size:
            #print ('recluster larger to small clusters', parent)
            for token in cluster:
              del synonyms[token] 
            if recluster_type=="individual":
              idxs = [token2idx[token.lstrip('¶')] for token in  cluster]
              true_k=int(max(2, (len(idxs))/max_cluster_size))
              synonyms = self._cluster_one_batch(cluster_embeddings, idxs, cluster, true_k, synonyms=synonyms, stopwords=stopwords, token2weight=token2weight, min_incremental_cluster_overlap=min_incremental_cluster_overlap)    
              cluster_batch = []
            else:
              cluster_batch.extend(cluster)
              if len(cluster_batch) > kmeans_batch_size:
                idxs = [token2idx[token.lstrip('¶')] for token in  cluster_batch]
                true_k=int(max(2, (len(idxs))/max_cluster_size))
                synonyms = self._cluster_one_batch(cluster_embeddings, idxs, cluster_batch, true_k, synonyms=synonyms, stopwords=stopwords, token2weight=token2weight, min_incremental_cluster_overlap=min_incremental_cluster_overlap)    
                cluster_batch = []
      if cluster_batch: 
        idxs = [token2idx[token.lstrip('¶')] for token in  cluster_batch]
        true_k=int(max(2, (len(idxs))/max_cluster_size))
        synonyms = self._cluster_one_batch(cluster_embeddings, idxs, cluster_batch, true_k, synonyms=synonyms, stopwords=stopwords, token2weight=token2weight, min_incremental_cluster_overlap=min_incremental_cluster_overlap)    
        cluster_batch = []
                
    return synonyms
  
  def _create_token_embeds_and_synonyms(self, model_name, synonyms=None, stopwords=None, token2weight=None, prefered_cluster_size = 10, \
                                      kmeans_batch_size=50000, epoch = 10, embed_batch_size=7000, min_prev_ids=10000, embedder="minilm", \
                                      max_ontology_depth=4,  max_cluster_size=100, do_ontology=True, recluster_type="batch", \
                                      min_incremental_cluster_overlap=2):
    global clip_model, minilm_model, labse_model
    if synonyms is None: synonyms = {} if not hasattr(self, 'synonyms') else self.synonyms
    if token2weight is None: token2weight = {} if not hasattr(self, 'token2weight') else self.token2weight    
    if stopwords is None: stopwords = {} if not hasattr(self, 'stopwords') else self.stopwords
    # assumes token2weight is an ordered dict, ordered roughly by frequency
    tokens = list(token2weight.keys())
    if not tokens: return synonyms
    token2idx = dict([(token, idx) for idx, token in enumerate(tokens)])
    if embedder == "clip":
      embed_dim = clip_model.config.text_config.hidden_size
    elif embedder == "minilm":
      embed_dim = minilm_model.config.hidden_size
    elif embedder == "labse":
      embed_dim = labse_model.config.hidden_size
    cluster_embeddings = np_memmap(f"{model_name}/{model_name}.{embedder}_tokens", shape=[len(token2weight), embed_dim])
    terms_idx = [idx for idx, term in enumerate(tokens) if term not in synonyms and term[0] != '¶' ]
    print ('creating embeds', len(terms_idx))
    terms_idx_in_synonyms = [idx for idx, term in enumerate(tokens) if term in synonyms and term[0] != '¶']
    len_terms_idx = len(terms_idx)
    #increase the terms_idx list to include non-parent tokens that have empty embeddings
    for rng in tqdm.tqdm(range(0, len(terms_idx), embed_batch_size)):
      max_rng = min(len(terms_idx), rng+embed_batch_size)
      if embedder == "clip":
        toks = clip_processor([tokens[idx].replace("_", " ") for idx in terms_idx[rng:max_rng]], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
          cluster_embeddings = clip_model.get_text_features(**toks).cpu().numpy()
      elif embedder == "minilm":
        toks = minilm_tokenizer([tokens[idx].replace("_", " ") for idx in terms_idx[rng:max_rng]], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
          cluster_embeddings = minilm_model(**toks)
          cluster_embeddings = mean_pooling(cluster_embeddings, toks.attention_mask).cpu().numpy()
      elif embedder == "labse":
        toks = labse_tokenizer([tokens[idx].replace("_", " ") for idx in terms_idx[rng:max_rng]], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
          cluster_embeddings = labse_model(**toks).pooler_output.cpu().numpy()          
      cluster_embeddings = np_memmap(self.mmap_file, shape=[len(tokens), cluster_embeddings.shape[1]], dat=cluster_embeddings, idxs=terms_idx[rng:max_rng])  
    len_terms_idx = len(terms_idx)
    times = -1
    times_start_recluster = max(0, (int(len(terms_idx)/int(kmeans_batch_size*.7))-3))
    print ('clusering embeds', len(terms_idx))
    for rng in tqdm.tqdm(range(0,len_terms_idx, int(kmeans_batch_size*.7))):
      times += 1
      max_rng = min(len_terms_idx, rng+int(kmeans_batch_size*.7))
      prev_ids = [idx for idx in terms_idx[:rng] if tokens[idx] not in synonyms]
      terms_idx_in_synonyms.extend([idx for idx in terms_idx[:rng] if tokens[idx] in synonyms])
      terms_idx_in_synonyms = list(set(terms_idx_in_synonyms))
      terms_idx_in_synonyms = [idx for idx in terms_idx_in_synonyms if tokens[idx] in synonyms]
      max_prev_ids = max(int(kmeans_batch_size*.15), int(.5*min_prev_ids))
      if len(prev_ids) > max_prev_ids:
        prev_ids = random.sample(prev_ids, max_prev_ids)
      avail_prev_ids= 2*max_prev_ids-len(prev_ids)
      if len(terms_idx_in_synonyms) > avail_prev_ids: 
          prev_ids.extend(random.sample(terms_idx_in_synonyms, avail_prev_ids))
      else: 
          prev_ids.extend(terms_idx_in_synonyms)
      idxs = prev_ids + terms_idx[rng:max_rng]
      #print ('clustering', len(idxs))
      true_k=int(max(2, (len(idxs))/prefered_cluster_size))
      terms2 = [tokens[idx] for idx in idxs]
      synonyms = self._cluster_one_batch(cluster_embeddings, idxs, terms2, true_k, synonyms=synonyms, stopwords=stopwords, token2weight=token2weight, min_incremental_cluster_overlap=min_incremental_cluster_overlap)
      if times >= times_start_recluster:
        cluster_batch=[]
        ontology = self.get_ontology(synonyms)
        for parent, cluster in ontology.items(): 
          if len(parent) - len(parent.lstrip('¶')) != 1: continue 
          if max_rng != len_terms_idx and len(cluster) < prefered_cluster_size*.5:
            for token in cluster:
              del synonyms[token]
          elif len(cluster) > max_cluster_size:
              #print ('recluster larger to small clusters', parent)
              for token in cluster:
                del synonyms[token] 
              if recluster_type=="individual":
                idxs = [token2idx[token] for token in  cluster]
                true_k=int(max(2, (len(idxs))/prefered_cluster_size))
                synonyms = self._cluster_one_batch(cluster_embeddings, idxs, cluster, true_k, synonyms=synonyms, stopwords=stopwords, token2weight=token2weight, min_incremental_cluster_overlap=min_incremental_cluster_overlap)    
                cluster_batch = []
              else:
                cluster_batch.extend(cluster)
                if len(cluster_batch) > kmeans_batch_size:
                  idxs = [token2idx[token] for token in  cluster_batch]
                  true_k=int(max(2, (len(idxs))/prefered_cluster_size))
                  synonyms = self._cluster_one_batch(cluster_embeddings, idxs, cluster_batch, true_k, synonyms=synonyms, stopwords=stopwords, token2weight=token2weight, min_incremental_cluster_overlap=min_incremental_cluster_overlap)    
                  cluster_batch = []
        if cluster_batch: 
          idxs = [token2idx[token] for token in  cluster_batch]
          true_k=int(max(2, (len(idxs))/prefered_cluster_size))
          synonyms = self._cluster_one_batch(cluster_embeddings, idxs, cluster_batch, true_k, synonyms=synonyms, stopwords=stopwords, token2weight=token2weight, min_incremental_cluster_overlap=min_incremental_cluster_overlap)    
          cluster_batch = []

    if do_ontology: synonyms = self._create_ontology(model_name, synonyms=synonyms, stopwords=stopwords, token2weight=token2weight,  token2idx=token2idx, kmeans_batch_size=50000, epoch = 10, \
                                                    embed_batch_size=embed_batch_size, min_prev_ids=min_prev_ids, embedder=embedder, max_ontology_depth=max_ontology_depth, max_cluster_size=max_cluster_size, \
                                                     recluster_type=recluster_type, min_incremental_cluster_overlap=min_incremental_cluster_overlap)
    return synonyms

  
  #creating tokenizer and a model. 
  #TODO, strip non_tokens
  @staticmethod
  def create_tokenizer_and_model(model_name, files, lmplz_loc="./riverbed/bin/lmplz", stopwords_max_len=10, \
                                 num_stopwords=75, min_compound_word_size=25, max_ontology_depth=4, max_cluster_size=100, \
                                 min_incremental_cluster_overlap=2, lstrip_stopwords=False, rstrip_stopwords=False, \
                                 non_tokens = "،♪↓↑→←━\₨₡€¥£¢¤™®©¶§←«»⊥∀⇒⇔√­­♣️♥️♠️♦️‘’¿*’-ツ¯‿─★┌┴└┐▒∎µ•●°。¦¬≥≤±≠¡×÷¨´:।`~�_“”/|!~@#$%^&*•()【】[]{}-_+–=<>·;…?:.,\'\"", \
                                 kmeans_batch_size=50000, dedup_compound_words_larger_than=None,  prefered_cluster_size = 10, \
                                 embed_batch_size=7000, min_prev_ids=10000, min_compound_weight=1.0, \
                                 stopwords=None, min_num_tokens=5, do_collapse_values=True, use_synonym_replacement=False, \
                                 embedder="minilm", do_ontology=True, recluster_type="batch", model=None):
      global device
      global labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set 
      labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set = init_models()
   
      os.system(f"mkdir -p {model_name}")
      assert dedup_compound_words_larger_than is None or min_compound_word_size <= dedup_compound_words_larger_than, "can't have a minimum compound words greater than what is removed"
      for name in files:
        assert os.path.getsize(name) < 5000000000, f"{name} size should be less than 5GB. Break up the file into 5GB shards."
      use_model(embedder)
      
      if model is not None:
        self = model
        tokenizer = self.tokenizer
      else:
        tokenizer = RiverbedTokenizer()
        self = RiverbedModel()
        self.tokenizer = tokenizer
        
      self.mmap_file = f"{model_name}/{embedder}_tokens.mmap"
      token2weight = self.tokenizer.token2weight = OrderedDict() if not hasattr(self, 'token2weight') else self.tokenizer.token2weight
      compound = self.tokenizer.compound = {} if not hasattr(self, 'compound') else self.tokenizer.compound
      synonyms = self.tokenizer.synonyms = {} if not hasattr(self, 'synonyms') else self.tokenizer.synonyms
      stopwords = self.tokenizer.stopwords = {} if not hasattr(self, 'stopwords') else self.tokenizer.stopwords
      self.synonyms = self.tokenizer.synonyms
      for token in stopwords_set:
        token = token.lower()
        stopwords[token] = stopwords.get(token, 1.0)
      if lmplz_loc != "./riverbed/bin/lmplz" and not os.path.exists("./lmplz"):
        os.system(f"cp {lmplz_loc} ./lmplz")
        lmplz = "./lmplz"
      else:
        lmplz = lmplz_loc
      os.system(f"chmod u+x {lmplz}")
      unigram = {}
      arpa = {}
      if token2weight:
        for token in token2weight.keys():
          if "_" not in token: unigram[token] = min(unigram.get(token,0), token2weight[token])
      if os.path.exists(f"{model_name}/{model_name}.arpa"):
        os.system(f"cp {model_name}/{model_name}.arpa {model_name}/__tmp__{model_name}.arpa")
        with open(f"{model_name}/{model_name}.arpa", "rb") as af:
          n = 0
          do_ngram = False
          for line in af:
            line = line.decode().strip()
            if line.startswith("\\1-grams:"):              
              n = 1
              do_ngram = True
            elif line.startswith("\\2-grams:"):
              n = 2
              do_ngram = True
            elif line.startswith("\\3-grams:"):
              n = 3
              do_ngram = True
            elif line.startswith("\\4-grams:"):
              n = 4
              do_ngram = True
            elif line.startswith("\\5-grams:"):
              n = 5
              do_ngram = True
      #TODO, check if a dedup file has already been created. If so, use that instead and don't do any dedup.
      #TODO, we should try to create consolidated files of around 1GB to get enough information in the arpa files
      for doc_id, name in enumerate(files):
        if dedup_compound_words_larger_than:
          dedup_compound_words_num_iter = max(0, 1+math.ceil(dedup_compound_words_larger_than/(5 *(doc_id+1))))
        else:
          dedup_compound_words_num_iter = 0
        num_iter = max(1,math.ceil(min_compound_word_size/(5 *(doc_id+1))))
        #we can repeatedly run the below to get long ngrams
        #after we tokenize for ngram and replace with tokens with underscores (the_projected_revenue) at each step, we redo the ngram count
        num_iter = max(num_iter,dedup_compound_words_num_iter)
        print ('doing', name, 'num iter', num_iter, "dedup at", dedup_compound_words_num_iter)
        prev_file = name
        for times in range(num_iter):
            print (f"iter {name}", times)
            # we only do synonym and embedding creation as the second to last or last step of each file processed 
            # b/c this is very expensive. we should do this right before the last counting if we
            # do synonym replacement so we have a chance to create syonyms for the replacement.
            # otherwise, we should do it after the last count. See below.
            synonyms_created=  False          
            if use_synonym_replacement and times == num_iter-1 and token2weight:
                synonyms_created = True
                self.synonyms = self.tokenizer.synonyms = synonyms = self._create_token_embeds_and_synonyms(model_name, stopwords=stopwords, token2weight=token2weight, synonyms=synonyms, kmeans_batch_size=kmeans_batch_size, min_incremental_cluster_overlap=min_incremental_cluster_overlap, \
                  prefered_cluster_size=prefered_cluster_size, embedder=embedder, embed_batch_size=embed_batch_size, min_prev_ids=min_prev_ids, max_ontology_depth=max_ontology_depth, max_cluster_size=max_cluster_size, do_ontology=do_ontology, recluster_type=recluster_type)   
            if token2weight:
              # if we want to make this faster, we can parrallelize this in a pool
              do_dedup = dedup_compound_words_larger_than is not None and times == dedup_compound_words_num_iter-1
              tmp_name = f"{model_name}/__tmp__{name}"
              if tmp_name.endswith(".gz"): tmp_name = tmp_name[:-len(".gz")]
              with open(tmp_name, "w", encoding="utf8") as tmp2:
                if prev_file.endswith(".gz"):
                  f = gzip.open(prev_file)
                else:
                  f = open(prev_file, "rb")
                seen_dedup_compound_words={}  
                deduped_num_tokens=0
                while True:
                  l = f.readline()
                  if not l: break   
                  l = l.decode().replace("\\n", "\n")
                  l = l.replace("_", " ").replace("  ", " ").strip()
                  if l:   
                    orig_l = l
                    l = tokenizer.tokenize(l,  min_compound_weight=min_compound_weight, compound=compound, token2weight=token2weight, synonyms=synonyms, use_synonym_replacement=use_synonym_replacement)
                    if times == num_iter-1:
                      l = tokenizer.tokenize(l, min_compound_weight=0, compound=compound, token2weight=token2weight,  synonyms=synonyms, use_synonym_replacement=use_synonym_replacement)
                    if do_dedup:                       
                      l = l.split()
                      dedup_compound_word = [w for w in l if "_" in w and w.count("_") + 1 > dedup_compound_words_larger_than]
                      if dedup_compound_word:
                        l = [w if ("_" not in w or w.count("_") + 1 <= dedup_compound_words_larger_than or w not in seen_dedup_compound_words) else '...' for w in l]
                        l2 = " ".join(l).replace(' ... ...', ' ...').strip()
                        if l2.endswith(" ..."): l2 = l2[:-len(" ...")]
                        if dedup_compound_word and l2.replace("_", " ") != orig_l:
                          deduped_num_tokens += 1
                        #  print ('dedup ngram', dedup_compound_word, l2)
                        for w in dedup_compound_word:
                          seen_dedup_compound_words[w] = 1
                        l = l2 
                      else:
                        l = " ".join(l)               
                    tmp2.write(l+"\n")  
                seen_dedup_compound_words = None
              os.system(f"gzip {tmp_name}")
              prev_file = f"{tmp_name}.gz" 
              if do_dedup:
                print ('deduped', deduped_num_tokens)
                dedup_name = f"{model_name}/{name}"
                if dedup_name.endswith(".gz"): dedup_name = dedup_name[:-len(".gz")]
                dedup_name +=".dedup.gz"
                os.system(f"cp {prev_file} {dedup_name}")
            if do_collapse_values:
              os.system(f"./{lmplz} --collapse_values  --discount_fallback  --skip_symbols -o 5 --prune {min_num_tokens}  --arpa {model_name}/__tmp__{name}.arpa <  {prev_file}") ##
            else:
              os.system(f"./{lmplz}  --discount_fallback  --skip_symbols -o 5 --prune {min_num_tokens}  --arpa {model_name}/__tmp__{name}.arpa <  {prev_file}") ##
            do_ngram = False
            n = 0
            with open(f"{model_name}/__tmp__{name}.{times}.arpa", "w", encoding="utf8") as tmp_arpa:
              with open(f"{model_name}/__tmp__{name}.arpa", "rb") as f:    
                for line in  f: 
                  line = line.decode().strip()
                  if not line: 
                    continue
                  if line.startswith("\\1-grams:"):
                    n = 1
                    do_ngram = True
                  elif line.startswith("\\2-grams:"):
                    n = 2
                    do_ngram = True
                  elif line.startswith("\\3-grams:"):
                    n = 3
                    do_ngram = True
                  elif line.startswith("\\4-grams:"):
                    n = 4
                    do_ngram = True
                  elif line.startswith("\\5-grams:"):
                    n = 5
                    do_ngram = True
                  elif do_ngram:
                    line = line.split("\t")
                    try:
                      weight = float(line[0])
                    except:
                      continue                  
                    if len(line) > 1 and (times >= dedup_compound_words_num_iter-1):
                      weight2 = weight
                      if weight2 > 0: weight2 = 0
                      tmp_arpa.write(f"{n}\t{line[1]}\t{weight2}\n")
                      #print ("got here")
                    weight = math.exp(weight)
                    line = line[1]
                    if not line: continue
                    line = line.split()
                    if [l for l in line if l in non_tokens or l in ('<unk>', '<s>', '</s>')]: continue
                    if not(len(line) == 1 and line[0] in stopwords):
                      if lstrip_stopwords:
                        while line:
                          if line[0].lower() in stopwords:
                            line = line[1:]
                          else:
                            break
                      if rstrip_stopwords:
                        while line:
                          if line[-1].lower() in stopwords:
                            line = line[:-1]
                          else:
                            break
                    token = "_".join(line)
                    if token.startswith('¶') and token not in token2weight: #unless this token is a parent synonym, we will strip our special prefix
                      token = token.lstrip('¶')
                    tokenArr = token.split("_")
                    if tokenArr[0]  in ('<unk>', '<s>', '</s>', ''):
                      tokenArr = tokenArr[1:]
                    if tokenArr[-1]  in ('<unk>', '<s>', '</s>', ''):
                      tokenArr = tokenArr[:-1]
                    if tokenArr:
                      # we are prefering stopwords that starts an n-gram. 
                      if (not lstrip_stopwords or len(tokenArr) == 1) and len(tokenArr[0]) <= stopwords_max_len:
                        sw = tokenArr[0].lower()
                        unigram[sw] = min(unigram.get(sw,100), weight)

                      #create the compound words length data structure
                      if weight >= min_compound_weight:
                        compound[tokenArr[0]] = [min(len(tokenArr),  compound.get(tokenArr[0],[100,0])[0]), max(len(tokenArr), compound.get(tokenArr[0],[100,0])[-1])]
                      weight = weight * len(tokenArr)            
                      token2weight[token] = min(token2weight.get(token, 100), weight) 
            os.system(f"rm {model_name}/__tmp__{name}.arpa")
            top_stopwords={} 
            if unigram:
                stopwords_list = [l for l in unigram.items() if len(l[0]) > 0]
                stopwords_list.sort(key=lambda a: a[1])
                len_stopwords_list = len(stopwords_list)
                top_stopwords = stopwords_list[:min(len_stopwords_list, num_stopwords)]
            for token, weight in top_stopwords:
              stopwords[token] = min(stopwords.get(token, 100), weight)
            if os.path.exists(f"{model_name}/__tmp__{model_name}.arpa"):
              os.system(f"cat {model_name}/__tmp__{model_name}.arpa {model_name}/__tmp__{name}.{times}.arpa > {model_name}/__tmp__{model_name}.arpa")
              os.system(f"rm {model_name}/__tmp__{name}.{times}.arpa")
            else:
              os.system(f"mv {model_name}/__tmp__{name}.{times}.arpa {model_name}/__tmp__{model_name}.arpa")
            if times == num_iter-1  and not synonyms_created:
                self.synonyms = self.tokenizer.synonyms = synonyms = self._create_token_embeds_and_synonyms(model_name, stopwords=stopwords, token2weight=token2weight, synonyms=synonyms, kmeans_batch_size=kmeans_batch_size, min_incremental_cluster_overlap=min_incremental_cluster_overlap, \
                  prefered_cluster_size=prefered_cluster_size, embedder=embedder, embed_batch_size=embed_batch_size, min_prev_ids=min_prev_ids, max_ontology_depth=max_ontology_depth, max_cluster_size=max_cluster_size, do_ontology=do_ontology, recluster_type=recluster_type)   
        
      print ('len syn', len(synonyms))
      self.tokenizer.token2weight, self.tokenizer.compound, self.tokenizer.synonyms, self.tokenizer.stopwords = token2weight, compound, synonyms, stopwords
      self.synonyms = self.tokenizer.synonyms
      
      #now create the clusters
      token2idx = self.tokenizer.token2idx()
      clusters = {}
      ontology = self.get_ontology(synonyms)
      for parent, a_cluster in ontology.items(): 
          level = len(parent) - len(parent.lstrip('¶'))-1
          label = (level, token2idx[parent.lstrip('¶')])
          for child in a_cluster:
            if child[0] != '¶':
              span = token2idx[child]
            else:
              span =  (level-1, token2idx[child.lstrip('¶')])
            clusters[label] = clusters.get(label, []) + [span]
      self.searcher.recreate_embeddings_idx(mmap_file=self.mmap_file, clusters=clusters, content_data_store=self.tokenizer.idx2token())
      #TODO: Clean up tmp files as we go along.
      #create the final kenlm .arpa file for calculating the perplexity. we do this in files in order to keep main memory low.
      print ('consolidating arpa')
      process_count = 5*(multiprocessing.cpu_count())
      os.system(f"sort {model_name}/__tmp__{model_name}.arpa -p {process_count} -o {model_name}/__tmp__{model_name}.arpa")
      ngram_cnt = [0]*5
      with open(f"{model_name}/__tmp__{model_name}.arpa", "rb") as f:  
        with open(f"{model_name}/__tmp__1_consolidated_{model_name}.arpa", "w", encoding="utf8") as tmp_arpa:
          prev_n = None
          prev_dat = None
          prev_val = 0
          for l in f:
            l = l.decode().strip()
            n, dat, val = l.split("\t")
            n = int(n)
            val = float(val)
            if val > 0:
              val =  0
            if prev_dat is not None and prev_dat != dat:
              ngram_cnt[prev_n-1] += 1
              tmp_arpa.write(f"{prev_val}\t{prev_dat}\t0\n")
              prev_val = val
            else:
              prev_val = min(val, prev_val)
            prev_dat = dat
            if prev_n != n:
              tmp_arpa.write(f"\\{n}-grams:\n")
            prev_n = n
          if prev_dat is not None:
            ngram_cnt[prev_n-1] += 1
            tmp_arpa.write(f"{prev_val}\t{prev_dat}\t0\n")
          tmp_arpa.write("\n\\end\\\n\n")
        with open(f"{model_name}/__tmp__2_consolidated_{model_name}.arpa", "w", encoding="utf8") as tmp_arpa2:
          tmp_arpa2.write("\\data\\\n")
          tmp_arpa2.write(f"ngram 1={ngram_cnt[0]}\n")
          tmp_arpa2.write(f"ngram 2={ngram_cnt[1]}\n")
          tmp_arpa2.write(f"ngram 3={ngram_cnt[2]}\n")
          tmp_arpa2.write(f"ngram 4={ngram_cnt[3]}\n")
          tmp_arpa2.write(f"ngram 5={ngram_cnt[4]}\n\n")
        os.system(f"cat {model_name}/__tmp__2_consolidated_{model_name}.arpa {model_name}/__tmp__1_consolidated_{model_name}.arpa > {model_name}/__tmp__consolidated_{model_name}.arpa")
      print ('creating kenlm model')
      self.kenlm_model = kenlm.LanguageModel(f"{model_name}/__tmp__consolidated_{model_name}.arpa") 
      
      os.system(f"mv {model_name}/__tmp__consolidated_{model_name}.arpa {model_name}/{model_name}.arpa")
      os.system(f"rm -rf {model_name}/__tmp__*")
      return tokenizer, self

  def save_pretrained(self, model_name):
      os.system(f"mkdir -p {model_name}")
      torch.save(self, open(f"{model_name}/{model_name}.pickle", "wb"))
    
  @staticmethod
  def from_pretrained(model_name):
      self = torch.load(open(f"{model_name}/{model_name}.pickle", "rb"))
      return self


#################################################################################
# SPAN AND DOCUMENT PROCESSOR
# An  ingester-model-view-controller framework for span and document processing. 

# this class is for creating and featuring spans from multiple documents, which are fragments of one or more sentences (not necessarily a paragraph).
# each span is a dict/json object. A span can specific to semantic search or bm25, and can include be indexed (span_idx) by (file name, line no, offset). The spans are also clustered. Finally the spans are serialzied 
# into a jsonl.gz file.
# assumes the sentences inside a document are NOT shuffeled, but documents can be shuffled. 
class RiverbedProcessor(MixingProcessor):
  RELATIVE_LOW = 0
  RELATIVE_MEDIUM = 1
  RELATIVE_HIGH= 2
  
  # for feature extraction on a single span and potentially between spans in a series. 
  # tuples of (feature_label, lower_band, upper_band, extractor). assumes prefix extraction has occured.
  # returns data which can be used to store in the feature_label for a span. if upper_band and lower_band are set, then an additional label X_level stores
  # the relative level label as well.
  #
  #TODO: other potential features include similarity of embedding from its cluster centroid
  #compound words %
  #stopwords %
  #term_frequency-inverse_document_frequency weight
  
  default_span_level_feature_extractors = [
      ('perplexity', .5, 1.5, lambda self, span: 0.0 if self.riverbed_model is None else self.riverbed_model.get_perplexity(span['tokenized_text'])),
      ('prefix', None, None, lambda self, span: "" if " || " not in span['text'] else  span['text'].split(" || ", 1)[0].strip()),
      ('date', None, None, lambda self, span: "" if " || " not in span['text'] else span['text'].split(" || ")[0].split(":")[-1].split("date of")[-1].strip("; ")), 
  ]

  # for extracting a prefix for a segment of text. a segment can contain multiple spans.
  default_prefix_extractors = [
      ('intro_with_date', _intro_with_date), \
      ('section_with_date', _section_with_date), \
      ('conclusion_with_date', _conclusion_with_date) \
      ]
  
  
  def _intro_with_date(self, span):
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

  def _section_with_date(self, span):
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

  def _conclusion_with_date(self, span):
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


  # the similarity models sometimes put too much weight on proper names, etc. but we might want to cluster by general concepts
  # such as change of control, regulatory actions, etc. The proper names themselves can be collapsed to one canonical form (The Person). 
  # Similarly, we want similar concepts (e.g., compound words) to cluster to one canonical form.
  # we do this by collapsing to an NER label and/or creating a synonym map from compound words to known tokens. See _create_ontology
  # and we use that data to generalize the sentence here.  
  #assumes the text has already been tokenized and ents has been sorted with longerst entity to shortest entity
  # TODO: have an option NOT to generalize the prefix. 
  def _generalize_text_and_filter_ents(self, tokenizer, tokenized_text, ents, ner_to_generalize=(), use_synonym_replacement=False):
      if not ner_to_generalize and not synonyms and not ents: return tokenized_text, ents

      #do a second tokenize if we want to do synonym replacement.
      if use_synonym_replacement:
        tokenized_text = tokenizer.tokenize(text, use_synonym_replacement=True)  
      filtered_ents = []

      #replace with placeholders
      for idx, ent in enumerate(ents):
          entity, label = ent
          if "@#@" not in text: break
          if f"@#@{idx}@#@" not in text: continue
          text = text.replace(f"@#@{idx}@#@", entity) 
      text = text.replace("_", " ")

      #see if there are multiple of the same labels 
      label_cnts = dict([(a,b) for a, b in Counter([label for entity, label in ents]).items() if b > 1])
      max_label_cnts = copy.copy(label_cnts)
      entity_2_id = {}
      for idx, ent in enumerate(ents):
          entity, label = ent
          entity_id = entity_2_id.get(entity) 
          if "@#@" not in tokenized_text: break
          if f"@#@{idx}@#@" not in tokenized_text: continue
          if entity not in entity_2_id and label in label_cnts:
            entity_id = entity_2_id[entity] = 1 + (max_label_cnts[label] - label_cnts[label])
            label_cnts[label] = label_cnts[label] - 1
          filtered_ents.append((entity, label,  text.count(f"@#@{idx}@#@")))
          if label in ner_to_generalize:   
            if label == 'ORG':
              tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Organization' + ('' if entity_id is None else f" {entity_id}"))
            elif label == 'PERSON':
              tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Person'+ ('' if entity_id is None else f" {entity_id}"))
            elif label == 'FAC':
              tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Facility'+ ('' if entity_id is None else f" {entity_id}"))
            elif label in ('GPE', 'LOC'):
              tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Location'+ ('' if entity_id is None else f" {entity_id}"))
            elif label in ('DATE', ):
              tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Date'+ ('' if entity_id is None else f" {entity_id}"))
            elif label in ('LAW', ):
              tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Law'+ ('' if entity_id is None else f" {entity_id}"))  
            elif label in ('EVENT', ):
              tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Event'+ ('' if entity_id is None else f" {entity_id}"))            
            elif label in ('MONEY', ):
              tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Amount'+ ('' if entity_id is None else f" {entity_id}"))
            else:
              tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", entity.replace(" ", "_"))
          else:
            tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", entity.replace(" ", "_"))    

      for _ in range(3):
        tokenized_text = tokenized_text.replace("The Person and The Person", "The Person").replace("The Person The Person", "The Person").replace("The Person, The Person", "The Person")
        tokenized_text = tokenized_text.replace("The Facility and The Facility", "The Facility").replace("The Facility The Facility", "The Facility").replace("The Facility, The Facility", "The Facility")
        tokenized_text = tokenized_text.replace("The Organization and The Organization", "The Organization").replace("The Organization The Organization", "The Organization").replace("The Organization, The Organization", "The Organization")
        tokenized_text = tokenized_text.replace("The Location and The Location", "The Location").replace("The Location The Location", "The Location").replace("The Location, The Location", "The Location")
        tokenized_text = tokenized_text.replace("The Date and The Date", "The Date").replace("The Date The Date", "The Date").replace("The Date, The Date", "The Date")
        tokenized_text = tokenized_text.replace("The Law and The Law", "The Law").replace("The Law The Law", "The Law").replace("The Law, The Law", "The Law")
        tokenized_text = tokenized_text.replace("The Event and The Event", "The Event").replace("The Event The Event", "The Event").replace("The Event, The Event", "The Event")
        tokenized_text = tokenized_text.replace("The Amount and The Amount", "The Amount").replace("The Amount The Amount", "The Amount").replace("The Amount, The Amount", "The Amount")

      return tokenized_text, filtered_ents



  # code more informative labels for the span clusters
  def _create_informative_parent_label_from_tfidf(self, clusters, span2idx, span2data, span2cluster_label, span_label2user_label=None, \
                                            label2term_frequency=None, document_frequency=None, domain_stopwords_set=stopwords_set, max_levels=4):
      if label2term_frequency is None: label2term_frequency = {}
      if document_frequency is None: document_frequency = {}
      if span_label2user_label is None: span_label2user_label = {}
      #we gather info for term_frequency-inverse_document_frequency with respect to each token in each clusters
      for label, values in clusters.items(): 
        if label[0] == 0 and label not in span_label2user_label:
          for item in values:
            if span in span2idx:
              data = span2data[span]
              text = data['tokenized_text']
              #we don't want the artificial labels to skew the tfidf calculations
              #assumes we don't have more than 10 of the same label
              text = text.replace('The Organization','').replace('The_Organization','')
              text = text.replace('The Person','').replace('The_Person','')
              text = text.replace('The Facility','').replace('The_Facility','')
              text = text.replace('The Location','').replace('The_Location','')          
              text = text.replace('The Date','').replace('The_Date','')
              text = text.replace('The Law','').replace('The_Law','')
              text = text.replace('The Amount','').replace('The_Amount','')
              text = text.replace('The Event','').replace('The_Event','')

              #we add back the entities we had replaced with the artificial labels into the term_frequency-inverse_document_frequency calculations
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
              text = [a for a in text if len(a) > 1 and ("_" not in a or (a.count("_")+1 != len([b for b in a.lower().split("_") if  b in domain_stopwords_set])))  and a.lower() not in domain_stopwords_set and a[0].lower() in "abcdefghijklmnopqrstuvwxyz"]
              cnts = Counter(text)
              aHash = label2term_frequency[label] =  label2term_frequency.get(label, {})
              for token, cnt in cnts.items():
                aHash[token] = cnt/len_text
              for token in cnts.keys():
                document_frequency[token] = document_frequency.get(token,0) + 1

      #Now, acually create the new label from the tfidf of the tokens in this cluster
      #TODO, see how we might save away the tfidf or get the tfidf from the bm25 indexer. 
      for label, term_frequency in label2term_frequency.items():
          tfidf = copy.copy(term_frequency)    
          for token in list(tfidf.keys()):
            tfidf[token]  = tfidf[token] * min(1.5, tokenizer.token2weight.get(token, 1)) * math.log(1.0/(1+document_frequency[token]))
          top_tokens2 = [a[0].lower().strip("~!@#$%^&*()<>,.:;")  for a in Counter(tfidf).most_common(min(len(tfidf), 40))]
          top_tokens2 = [a for a in top_tokens2 if a not in domain_stopwords_set and ("_" not in a or (a.count("_")+1 != len([b for b in a.split("_") if  b in domain_stopwords_set])))]
          top_tokens = []
          for t in top_tokens2:
            if t not in top_tokens:
              top_tokens.append(t)
          if top_tokens:
            if len(top_tokens) > 5: top_tokens = top_tokens[:5]
            new_label = ", ".join(top_tokens) 
            span_label2user_label[label] = new_label

      #create parent labels
      for old_label, new_label in span_label2user_label.items():
        for parent_old_label in [(level, old_label[1]) for level in range(1, max_levels)]:
          if parent_old_label clusters:
            span_label2user_label[parent_old_label]= ("¶"*parent_old_label[0])+new_label

      return span_label2user_label, label2term_frequency, document_frequency


# This class uses the RivberbedPreprocessor to generate spans and index the spans. It also provides APIs for searching the spans.
#we can use the ontology for query expansion as part of the bm25 search. 
class RiverbedSearcherIndexer:

  def __init__(self, project_name, processor, span2idx, batch, retained_batch, span_lfs,  span2cluster_label, \
                start_idx = 0, embed_search_field="text", bm25_field="text", text_span_size=1000, embedder="minilm", do_ontology=True, running_features_per_label={}, \
                ner_to_generalize=(), span_level_feature_extractors=default_span_level_feature_extractors, \
                running_features_size=100, label2term_frequency=None, document_frequency=None, domain_stopwords_set=stopwords_set,\
                use_synonym_replacement=False, ):
    super().__init__()
    self.idx = start_idx
    self.embed_search_field = embed_search_field
    self.bm25_field = bm25_field
    self.searcher = Searcher()


  def tokenize(self, *args, **kwargs):
    return self.tokenizer.tokenize(*args, **kwargs)

  #transform a doc into a span batch, breaking up doc into spans
  #all spans/leaf nodes of a cluster are stored as a triple of (name, lineno, offset)
  def _create_spans_batch(self, curr_file_size, doc, text_span_size=1000, ner_to_generalize=(), use_synonym_replacement=False):
      batch2 = []
      if True:
        name, curr_lineno, ents, text  = doc['name'], doc['lineno'], doc['ents'], doc['text']
        for idx, ent in enumerate(ents):
          text = text.replace(ent[0], f' @#@{idx}@#@ ')
        # we do placeholder replacement tokenize to make ngram tokens underlined, so that we don't split a span in the middle of an ner token or ngram.
        text  = self.tokenizer.tokenize(text, use_synonym_replacement=False) 
        len_text = len(text)
        prefix = ""
        if "||" in text:
          prefix, _ = text.split("||",1)
          prefix = prefix.strip()
        offset = 0
        while offset < len_text:
          max_rng  = min(len_text, offset+text_span_size+1)
          if text[max_rng-1] != ' ':
            #TODO: extend for non english periods and other punctuations
            if '. ' in text[max_rng:]:
              max_rng = max_rng + text[max_rng:].index('. ')+1
            elif ' ' in text[max_rng:]:
              max_rng = max_rng + text[max_rng:].index(' ')
            else:
              max_rng = len_text
          if prefix and offset > 0:
            text2 = prefix +" || ... " + text[offset:max_rng].strip().replace("_", " ").replace("  ", " ").replace("  ", " ")
          else:
            text2 = text[offset:max_rng].strip().replace("_", " ").replace("  ", " ").replace("  ", " ")
          tokenized_text, ents2 = _generalize_text_and_filter_ents(self.tokenizer, tokenized_text2, ents, ner_to_generalize, use_synonym_replacement=use_synonym_replacement) 
          if prefix and offset > 0:
            _, text2 = text2.split(" || ... ", 1)
          sub_span = copy.deepcopy(span)
          sub_span['position'] += offset/curr_file_size
          sub_span['offset'] = offset
          sub_span['text'] = text2
          sub_span['tokenized_text'] = tokenized_text 
          sub_span['ents'] = ents2
          batch2.append(sub_span)
          offset = max_rng

      return batch2
    

  
  def _create_span_features(self, batch, span_level_feature_extractors, running_features_per_label, running_features_size):
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
      span['cluster_label']= None
      span['cluster_label_before']= None
      span['cluster_label_after']= None
      for feature_label, features_per_label, relative_level_per_label in  zip(feature_labels, features, relative_levels):
        span[feature_label] = features_per_label[idx]
        if relative_level_per_label: span[feature_label+"_level"] = relative_level_per_label[idx]
      ent_cnts = Counter(v[1].lower()+"_cnt" for v in span['ents'])
      for feature_label, cnt in ent_cnts.items():
        span[feature_label] = cnt
    return batch
  
  #TODO: add inverse cluster size as a feature.
  #      cosine distance to nearest neighbor cluster head

  # similar to _create_token_embeds_and_synonyms, except for spans     
  #(1) compute features and embeddings in one batch for tokenized text.
  #(2) create clusters in an incremental fashion from batch
  #all leaf nodes are spans
  #spanf2idx is a mapping from the span to the actual underlying storage idx (e.g., a jsonl file or database)
  #span2cluster_label is like the synonym data-structure for tokens.

  def _create_span_embeds_and_span2cluster_label(self):
    pass
  
    # gets in a lines iterator and outputs subsequent dict generator
  def process(self, lines_iterator, *args, **kwargs):
    for line in lines_iterator:
      doc =  _get_content_from_line(line, self.search_field)
      if not l: 
        yield None
        continue
      try:
        line = line.decode()
      except:
        pass
      offset = 0
      for data in self._create_spans_batch(curr_file_size, doc, text_span_size=kwargs.get('text_span_size',1000), \
                                           ner_to_generalize=kwargs.get('ner_to_generalize', ()),0\
                                           use_synonym_replacement=kwargs.get('use_synonym_replacement', False)):
        yield data
      self.idx += 1
      
  # the main method for processing documents and their spans. 
  def index(self, content_store_objs):
    global clip_model, minilm_model, labse_model
    model = self.model
    tokenizer = self.tokenizer
    searcher = self.searcher.switch_search_context(f"{project_name}.jsonl")
    os.system(f"mkdir -p {project_name}.jsonl_idx")
    span2idx = self.span2idx = OrderedDict() if not hasattr(self, 'span2idx') else self.span2idx
    span_clusters = self.span_clusters = {} if not hasattr(self, 'span_clusters') else self.span_clusters
    label2term_frequency = self.label2term_frequency = {} if not hasattr(self, 'label2term_frequency') else self.label2term_frequency
    document_frequency = self.document_frequency = {} if not hasattr(self, 'document_frequency') else self.document_frequency
    span2cluster_label = self.span2cluster_label = {} if not hasattr(self, 'span2cluster_label') else self.span2cluster_label
    if (not hasattr(model, 'kenlm_model') or model.kenlm_model is not None) and auto_create_tokenizer_and_model:
      tokenizer, model = self.tokenizer, self.model = RiverbedModel.create_tokenizer_and_model(project_name, files, )
    kenlm_model = self.model.kenlm_model 
    use_model(embedder)
    running_features_per_label = {}
    name = files.pop()
    f = open(name) 
    domain_stopwords_set = set(list(stopwords_set) + list(stopwords.keys()))


    if seen is None: seen = {}
    
    for name, curr_file_size, content_store_obj in content_store_objs.items():
      # we will collapse adjacent lines that are short. 
      prior_line = ""
      batch = []
      retained_batch = []
      curr = ""
      cluster_embeddings = None
      curr_date = ""
      curr_position = 0
      next_position = 0
      curr_file_size = os.path.getsize(name)
      position = 0
      line = ""
      lineno = -1
      curr_lineno = 0
      seen={}
      for line in content_store_obj:
        line = line.strip() #TODO: read the data from the field
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
                  seen[hash_id] = 1
                  curr_ents = list(itertools.chain(*[[(e.text, e.label_)] if '||' not in e.text else [(e.text.split("||")[0].strip(), e.label_), (e.text.split("||")[-1].strip(), e.label_)] for e in spacy_nlp(curr).ents]))
                  curr_ents = list(set([e for e in curr_ents if e[0]]))
                  curr_ents.sort(key=lambda a: len(a[0]), reverse=True)
                  for item in self._create_spans_batch(curr_file_size, {'text': curr, 'ents': curr_ents, }, text_span_size=text_span_size, ner_to_generalize=ner_to_generalize, use_synonym_replacement=use_synonym_replacement)
                    item['name'] =name
                    item['lineno'] = curr_lineno
                    item['position'] = curr_position
                    batch.append(item)
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
          batch_id_prefix += 1
          retained_batch, span2idx, span2cluster_label, label2term_frequency, document_frequency = self._create_span_embeds_and_span2cluster_label(project_name, curr_file_size, jsonl_file_idx, span2idx, batch, \
                                                      retained_batch, jsonl_file,  f"{batch_id_prefix}_", span_lfs,  span2cluster_label, text_span_size, \
                                                      kmeans_batch_size=kmeans_batch_size, epoch = epoch, embed_batch_size=embed_batch_size, min_prev_ids=min_prev_ids, \
                                                      max_ontology_depth=max_ontology_depth, max_top_parents=max_top_parents, do_ontology=True, embedder=embedder, \
                                                      running_features_per_label=running_features_per_label, ner_to_generalize=ner_to_generalize, span_level_feature_extractors=span_level_feature_extractors, \
                                                      running_features_size=running_features_size, label2term_frequency=label2term_frequency, document_frequency=document_frequency, domain_stopwords_set=domain_stopwords_set, \
                                                      verbose_snrokel=verbose_snrokel,  span_per_cluster=span_per_cluster, use_synonym_replacement=use_synonym_replacement, )  
          batch = []
      
      # do one last line and finish processing if there's anything left
      if curr: 
          curr = curr.replace(". .", ". ").replace("..", ".").replace(":.", ".")
          hash_id = hash(curr)
          if not dedup or (hash_id not in seen):
              curr_ents = list(itertools.chain(*[[(e.text, e.label_)] if '||' not in e.text else [(e.text.split("||")[0].strip(), e.label_), (e.text.split("||")[-1].strip(), e.label_)] for e in spacy_nlp(curr).ents]))
              curr_ents = list(set([e for e in curr_ents if e[0]]))
              curr_ents.sort(key=lambda a: len(a[0]), reverse=True)
              for item in self._create_spans_batch(curr_file_size, {'text': curr, 'ents': curr_ents, }, text_span_size=text_span_size, ner_to_generalize=ner_to_generalize, use_synonym_replacement=use_synonym_replacement)
                item['name'] =name
                item['lineno'] = curr_lineno
                item['position'] = curr_position
                batch.append(item)
          
      #do the last bactch    
      if batch: 
          batch_id_prefix += 1
          retained_batch, span2idx, span2cluster_label, label2term_frequency, document_frequency = self._create_span_embeds_and_span2cluster_label(project_name, curr_file_size, jsonl_file_idx, spanf2idx, batch, \
                                                      retained_batch, jsonl_file,  f"{batch_id_prefix}_", span_lfs,  span2cluster_label, text_span_size, \
                                                      kmeans_batch_size=kmeans_batch_size, epoch = epoch, embed_batch_size=embed_batch_size, min_prev_ids=min_prev_ids,  \
                                                      max_ontology_depth=max_ontology_depth, max_top_parents=max_top_parents, do_ontology=True, embedder=embedder,\
                                                      running_features_per_label=running_features_per_label, ner_to_generalize=ner_to_generalize, span_level_feature_extractors=span_level_feature_extractors, \
                                                      running_features_size=running_features_size, label2term_frequency=label2term_frequency, document_frequency=document_frequency, domain_stopwords_set=domain_stopwords_set, \
                                                      verbose_snrokel=verbose_snrokel)  
          batch = []
          
      
      
          
    span2idx, span_clusters, label2term_frequency, document_frequency, span2cluster_label = self.span2idx, self.span_clusters, self.label2term_frequency, self.document_frequency, self.span2cluster_label                    
    self.searcher = searcher.switch_search_context(project_name, data_iterator=data_iterator, search_field="tokenized_text", bm25_field="text", embedder=embedder, \
                             auto_embed_text=True, auto_create_bm25_idx=True, auto_create_embeddings_idx=True)
    
    #TODO: cleanup label2term_frequency, document_frequency
    return self

  def gzip_jsonl_file(self):
    os.system(f"gzip {project_name}/spans.jsonl")
    GzipFileByLine(f"{project_name}/spans.jsonl")
    
  def save_pretrained(self, project_name):
      os.system(f"mkdir -p {project_name}")
      torch.save(self, open(f"{project_name}/{project_name}.pickle", "wb"))
    
  @staticmethod
  def from_pretrained(project_name):
      self = torch.load(open(f"{project_name}/{project_name}.pickle", "rb"))
      return self

#given the results of the indexer and processor, perform analysis on the data seralized to jsonl.gz file and one or more searches of the indexed data.     
class RiverbedAnalyzer:
  def __init__(self, project_name, searcher_indexer, processor):
    self.project_name, self.searcher_indexer, self.processor = project_name, searcher_indexer, processor
    
  def search_and_label(self, positive_query_set, negative_query_set):
    pass
  
  #returns a model - could be any transformer classification head or sci-kitlearn supervised learning system
  def fit(self, labaled_content_data_store, predicted_labels):
    pass
  
  #returns a tag
  def predict(self, model, example):
    pass
  
  #label all entries in the jsonl spans and emits a labeled_file_name.jsonl.gz file. Fills in the prediction in the label_field.
  #will not label the iterm if the confidence score is below filter_below_confidence_score
  def label_all_in_project(self, model, label_file_name, label_field, filter_below_confidence_score=0.0):
    pass

#this forms part of the view portion of the MVC
#used for exploring and viewing different parts of data and results from the analyzer. 
#display results and pandas data frames in command line, notebook or as a flask API service. 
class RiverbedVisualizer:
  def __init__(self, project_name, analyzer, searcher_indexer, processor):
    self.project_name, self.analyzer, self.searcher_indexer, self.processor = project_name, analyzer, searcher_indexer, processor
    
