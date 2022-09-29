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
from sklearn.cluster import AgglomerativeClustering
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
# the model stores a copy of synonyms and the kenlm data structures. the model will allow us to find an ontology
# for tokens in the model, and the perplexity of a text.
# the processor sores in the span_clusters, span2class_label, span2idx, label_models, label2tf, and df data structures. 


#################################################################################
# TOKENIZER CODE
class RiverbedTokenizer:

  def __init__(self):
    pass

  def token2idx(self):
    return OrderedDict([(term, idx) for idx, term in enumerate(self.tokenweight.items())])
    
  def tokenize(self, doc, min_compound_weight=0,  max_compound_word_size=10000, compound=None, token2weight=None, synonyms=None, use_synonym_replacement=False):
    if synonyms is None: synonyms = {} if not hasattr(self, 'synonyms') else self.synonyms
    if token2weight is None: token2weight = {} if not hasattr(self, 'token2weight') else self.token2weight    
    if compound is None: compound = {} if not hasattr(self, 'compound') else self.compound
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
class RiverbedModel(nn.Module):

  def __init__(self):
   super().__init__()
   global labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set 
   labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set = init_models()
   self.searcher = Searcher()

    
  # get the downsampled sentence embeddings. can be used to train the downsampler(s).
  def forward(self, *args, **kwargs):
    if 'text' in kwargs:
      text = kwargs['text']
      # we tokenize using the RiverbedTokenizer to take into account n-grams
      text = self.tokenizer.tokenize(text) 
      #we create the downsample sentence embeding with mean of the ngram and the non-ngram 
      #sentences. 
      # we assume that at this point we have already increased the embeddings/tokens in the underlying tokenizer
      # not happy -> no_happy, which is similar to "sad", "angry", "unhappy"
      #then we tokenize using the embedder tokenizer
    dat = self.searcher(*args, **kwargs)
    with torch.no_grad():
      if self.embedder == "clip":
        dat = clip_model.get_text_features(*args, **kwargs)
      elif self.embedder == "minilm":
        dat = minilm_model(*args, **kwargs)
        dat = mean_pooling(dat, kwargs['attention_mask'])
      elif self.embedder == "labse":
        dat = labse_model(*args, **kwargs).pooler_output   
    dat = torch.nn.functional.normalize(dat, dim=1)
    dat = self.downsampler(dat)
    if self.universal_embed_mode:
      dat = cosine_similarity(dat, prototypes)
      dat = torch.nn.functional.normalize(dat, dim=1)
      dat = self.universal_downsampler(dat)
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

  # cluster one batch of tokens/vectors, assuming some tokens have already been clustered
  def _cluster_one_batch(self, cluster_vecs, idxs, terms2, true_k, synonyms=None, stopwords=None, token2weight=None, min_incremental_cluster_overlap=2 ):
    global device
    #print ('cluster_one_batch', len(idxs))
    if synonyms is None: synonyms = {} if not hasattr(self, 'synonyms') else self.synonyms
    if token2weight is None: token2weight = {} if not hasattr(self, 'token2weight') else self.token2weight    
    if stopwords is None: stopwords = {} if not hasattr(self, 'stopwords') else self.stopwords
    if device == 'cuda':
      kmeans = KMeans(n_clusters=true_k, mode='cosine')
      km_labels = kmeans.fit_predict(torch.from_numpy(cluster_vecs[idxs]).to(device))
      km_labels = [l.item() for l in km_labels.cpu()]
    else:
      km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                                          init_size=max(true_k*3,1000), batch_size=1024).fit(cluster_vecs[idxs])
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
    cluster_vecs = np_memmap(f"{model_name}/{model_name}.{embedder}_tokens", shape=[len(token2weight), embed_dim])
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
      synonyms = self._cluster_one_batch(cluster_vecs, idxs, parents, true_k, synonyms=synonyms, stopwords=stopwords, token2weight=token2weight, )
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
              synonyms = self._cluster_one_batch(cluster_vecs, idxs, cluster, true_k, synonyms=synonyms, stopwords=stopwords, token2weight=token2weight, min_incremental_cluster_overlap=min_incremental_cluster_overlap)    
              cluster_batch = []
            else:
              cluster_batch.extend(cluster)
              if len(cluster_batch) > kmeans_batch_size:
                idxs = [token2idx[token.lstrip('¶')] for token in  cluster_batch]
                true_k=int(max(2, (len(idxs))/max_cluster_size))
                synonyms = self._cluster_one_batch(cluster_vecs, idxs, cluster_batch, true_k, synonyms=synonyms, stopwords=stopwords, token2weight=token2weight, min_incremental_cluster_overlap=min_incremental_cluster_overlap)    
                cluster_batch = []
      if cluster_batch: 
        idxs = [token2idx[token.lstrip('¶')] for token in  cluster_batch]
        true_k=int(max(2, (len(idxs))/max_cluster_size))
        synonyms = self._cluster_one_batch(cluster_vecs, idxs, cluster_batch, true_k, synonyms=synonyms, stopwords=stopwords, token2weight=token2weight, min_incremental_cluster_overlap=min_incremental_cluster_overlap)    
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
    cluster_vecs = np_memmap(f"{model_name}/{model_name}.{embedder}_tokens", shape=[len(token2weight), embed_dim])
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
          cluster_vecs = clip_model.get_text_features(**toks).cpu().numpy()
      elif embedder == "minilm":
        toks = minilm_tokenizer([tokens[idx].replace("_", " ") for idx in terms_idx[rng:max_rng]], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
          cluster_vecs = minilm_model(**toks)
          cluster_vecs = mean_pooling(cluster_vecs, toks.attention_mask).cpu().numpy()
      elif embedder == "labse":
        toks = labse_tokenizer([tokens[idx].replace("_", " ") for idx in terms_idx[rng:max_rng]], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
          cluster_vecs = labse_model(**toks).pooler_output.cpu().numpy()          
      cluster_vecs = np_memmap(f"{model_name}/{model_name}.{embedder}_tokens", shape=[len(tokens), cluster_vecs.shape[1]], dat=cluster_vecs, idxs=terms_idx[rng:max_rng])  
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
      synonyms = self._cluster_one_batch(cluster_vecs, idxs, terms2, true_k, synonyms=synonyms, stopwords=stopwords, token2weight=token2weight, min_incremental_cluster_overlap=min_incremental_cluster_overlap)
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
                synonyms = self._cluster_one_batch(cluster_vecs, idxs, cluster, true_k, synonyms=synonyms, stopwords=stopwords, token2weight=token2weight, min_incremental_cluster_overlap=min_incremental_cluster_overlap)    
                cluster_batch = []
              else:
                cluster_batch.extend(cluster)
                if len(cluster_batch) > kmeans_batch_size:
                  idxs = [token2idx[token] for token in  cluster_batch]
                  true_k=int(max(2, (len(idxs))/prefered_cluster_size))
                  synonyms = self._cluster_one_batch(cluster_vecs, idxs, cluster_batch, true_k, synonyms=synonyms, stopwords=stopwords, token2weight=token2weight, min_incremental_cluster_overlap=min_incremental_cluster_overlap)    
                  cluster_batch = []
        if cluster_batch: 
          idxs = [token2idx[token] for token in  cluster_batch]
          true_k=int(max(2, (len(idxs))/prefered_cluster_size))
          synonyms = self._cluster_one_batch(cluster_vecs, idxs, cluster_batch, true_k, synonyms=synonyms, stopwords=stopwords, token2weight=token2weight, min_incremental_cluster_overlap=min_incremental_cluster_overlap)    
          cluster_batch = []

    if do_ontology: synonyms = self._create_ontology(model_name, synonyms=synonyms, stopwords=stopwords, token2weight=token2weight,  token2idx=token2idx, kmeans_batch_size=50000, epoch = 10, \
                                                    embed_batch_size=embed_batch_size, min_prev_ids=min_prev_ids, embedder=embedder, max_ontology_depth=max_ontology_depth, max_cluster_size=max_cluster_size, \
                                                     recluster_type=recluster_type, min_incremental_cluster_overlap=min_incremental_cluster_overlap)
    return synonyms

  
  # creating tokenizer and a model. 
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
      for file_name in files:
        assert os.path.getsize(file_name) < 5000000000, f"{file_name} size should be less than 5GB. Break up the file into 5GB shards."
      if embedder == "clip":
        clip_model = clip_model.to(device)
        minilm_model =  minilm_model.cpu()
        labse_model =  labse_model.cpu()
      elif embedder == "minilm":
        clip_model = clip_model.cpu()
        minilm_model =  minilm_model.to(device)
        labse_model =  labse_model.cpu()
      elif embedder == "labse":
        clip_model = clip_model.cpu()
        minilm_model =  minilm_model.cpu()
        labse_model =  labse_model.to(device)
      
      if model is not None:
        self = model
        tokenizer = self.tokenizer
      else:
        tokenizer = RiverbedTokenizer()
        self = RiverbedModel()
        self.tokenizer = tokenizer
        
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
      for doc_id, file_name in enumerate(files):
        if dedup_compound_words_larger_than:
          dedup_compound_words_num_iter = max(0, 1+math.ceil(dedup_compound_words_larger_than/(5 *(doc_id+1))))
        else:
          dedup_compound_words_num_iter = 0
        num_iter = max(1,math.ceil(min_compound_word_size/(5 *(doc_id+1))))
        #we can repeatedly run the below to get long ngrams
        #after we tokenize for ngram and replace with tokens with underscores (the_projected_revenue) at each step, we redo the ngram count
        num_iter = max(num_iter,dedup_compound_words_num_iter)
        print ('doing', file_name, 'num iter', num_iter, "dedup at", dedup_compound_words_num_iter)
        prev_file = file_name
        for times in range(num_iter):
            print (f"iter {file_name}", times)
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
              tmp_file_name = f"{model_name}/__tmp__{file_name}"
              if tmp_file_name.endswith(".gz"): tmp_file_name = tmp_file_name[:-len(".gz")]
              with open(tmp_file_name, "w", encoding="utf8") as tmp2:
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
              os.system(f"gzip {tmp_file_name}")
              prev_file = f"{tmp_file_name}.gz" 
              if do_dedup:
                print ('deduped', deduped_num_tokens)
                dedup_file_name = f"{model_name}/{file_name}"
                if dedup_file_name.endswith(".gz"): dedup_file_name = dedup_file_name[:-len(".gz")]
                dedup_file_name +=".dedup.gz"
                os.system(f"cp {prev_file} {dedup_file_name}")
            if do_collapse_values:
              os.system(f"./{lmplz} --collapse_values  --discount_fallback  --skip_symbols -o 5 --prune {min_num_tokens}  --arpa {model_name}/__tmp__{file_name}.arpa <  {prev_file}") ##
            else:
              os.system(f"./{lmplz}  --discount_fallback  --skip_symbols -o 5 --prune {min_num_tokens}  --arpa {model_name}/__tmp__{file_name}.arpa <  {prev_file}") ##
            do_ngram = False
            n = 0
            with open(f"{model_name}/__tmp__{file_name}.{times}.arpa", "w", encoding="utf8") as tmp_arpa:
              with open(f"{model_name}/__tmp__{file_name}.arpa", "rb") as f:    
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
            os.system(f"rm {model_name}/__tmp__{file_name}.arpa")
            top_stopwords={} 
            if unigram:
                stopwords_list = [l for l in unigram.items() if len(l[0]) > 0]
                stopwords_list.sort(key=lambda a: a[1])
                len_stopwords_list = len(stopwords_list)
                top_stopwords = stopwords_list[:min(len_stopwords_list, num_stopwords)]
            for token, weight in top_stopwords:
              stopwords[token] = min(stopwords.get(token, 100), weight)
            if os.path.exists(f"{model_name}/__tmp__{model_name}.arpa"):
              os.system(f"cat {model_name}/__tmp__{model_name}.arpa {model_name}/__tmp__{file_name}.{times}.arpa > {model_name}/__tmp__{model_name}.arpa")
              os.system(f"rm {model_name}/__tmp__{file_name}.{times}.arpa")
            else:
              os.system(f"mv {model_name}/__tmp__{file_name}.{times}.arpa {model_name}/__tmp__{model_name}.arpa")
            if times == num_iter-1  and not synonyms_created:
                self.synonyms = self.tokenizer.synonyms = synonyms = self._create_token_embeds_and_synonyms(model_name, stopwords=stopwords, token2weight=token2weight, synonyms=synonyms, kmeans_batch_size=kmeans_batch_size, min_incremental_cluster_overlap=min_incremental_cluster_overlap, \
                  prefered_cluster_size=prefered_cluster_size, embedder=embedder, embed_batch_size=embed_batch_size, min_prev_ids=min_prev_ids, max_ontology_depth=max_ontology_depth, max_cluster_size=max_cluster_size, do_ontology=do_ontology, recluster_type=recluster_type)   
        
      print ('len syn', len(synonyms))
      self.tokenizer.token2weight, self.tokenizer.compound, self.tokenizer.synonyms, self.tokenizer.stopwords = token2weight, compound, synonyms, stopwords
      self.synonyms = self.tokenizer.synonyms
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
      pickle.dump(self, open(f"{model_name}/{model_name}.pickle", "wb"))
    
  @staticmethod
  def from_pretrained(model_name):
      self = pickle.load(open(f"{model_name}/{model_name}.pickle", "rb"))
      return self
