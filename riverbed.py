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

# The Riverbed system is used for information retrieval at scale.
# The Riverbed code includes a RiverbedTokenizer, RiverbedModel and RiverbedDocumenProcessor for information retrieval processing. 
# The tokenizer stores the stopwords, compound, token2weight and synonyms data structure.
# the model stores a copy of synonyms and the kenlm data structures. the model will allow us to find an ontology
# for tokens in the model, and the perplexity of a text.
# the processor sores in the span_clusters, span2class_label, span2idx, label_models, label2tf, and df data structures. 
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
import fasttext
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
from snorkel.labeling import labeling_function
import itertools
from nltk.corpus import stopwords as nltk_stopwords
import pickle
from collections import OrderedDict
from fast_pytorch_kmeans import KMeans
import torch
import tqdm
import gzip
from torch import multiprocessing
import bisect
from functools import partial 
if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

try:
  if minilm_model is not None: 
    pass
except:
   labse_tokenizer= labse_model=  clip_processor = minilm_tokenizer= clip_model= minilm_model= spacy_nlp= stopwords_set = None
  
def np_memmap(f, dat=None, idxs=None, shape=None, dtype=np.float32, ):
  if not f.endswith(".mmap"):
    f = f+".mmap"
  if os.path.exists(f):
    mode = "r+"
  else:
    mode = "w+"
  if shape is None: shape = dat.shape
  memmap = np.memmap(f, mode=mode, dtype=dtype, shape=tuple(shape))
  if dat is None:
    return memmap
  if tuple(shape) == tuple(dat.shape):
    memmap[:] = dat
  else:
    memmap[idxs] = dat
  return memmap

def init_models():
  global labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set 
  if minilm_model is None:
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")   
    minilm_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    labse_tokenizer = BertTokenizerFast.from_pretrained("setu4993/smaller-LaBSE")

    if device == 'cuda':
      clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").half().eval()
      minilm_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').half().eval()
      labse_model = BertModel.from_pretrained("setu4993/smaller-LaBSE").half().eval()
    else:
      clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
      minilm_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').eval()
      lbase_model = BertModel.from_pretrained("setu4993/smaller-LaBSE").eval()

    spacy_nlp = spacy.load('en_core_web_md')
    stopwords_set = set(nltk_stopwords.words('english') + ['...', 'could', 'should', 'shall', 'can', 'might', 'may', 'include', 'including'])

#Mean Pooling - Take attention mask into account for correct averaging
#TODO, mask out the prefix for data that isn't the first portion of a prefixed text.
def mean_pooling(model_output, attention_mask):
    with torch.no_grad():
      token_embeddings = model_output.last_hidden_state
      input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
      return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    
#default is to do 100K lines per chunk, for 100 chunks per batch (or about 10M lines altogether)
def batchify_from_fileobj(f, max_chunks=10^2, chunk_size=None, chunk_lines=10^5, decode=True, apply_fn=None):
  n_chunk = 0
  curr_size = 0
  curr_line = 0
  batch = []
  chunk = []
  if chunk_size is not None:
    for l in f:
      if decode:
        l = l.decode()
      if apply_fn is not None:
        l = apply_fn(l)
      chunk.append(l)      
      curr_size += len(l)
      if curr_size >= chunk_size:
        batch.append(batch)
        n_chunk += 1
        curr_size = 0
        chunk = []
      if n_chunk >= max_chunks:
        return batch
    if chunk: batch.append(chunk)
    return batch
  else:
    for l in f:
      if decode:
        l = l.decode()
      if apply_fn is not None:
        l = apply_fn(l)
      chunk.append(l)      
      curr_line += 1
      if curr_line >= chunk_lines:
        batch.append(chunk)
        n_chunk += 1
        curr_line = 0
        chunk = []
      if n_chunk >= max_chunks:
        return batch
    if chunk: batch.append(chunk)
    return batch
      
    
#################################################################################
# TOKENIZER CODE
class RiverbedTokenizer:

  def __init__(self):
    pass

  def token2idx(self):
    return OrderedDict([(term, idx) for idx, term in enumerate(self.tokenweight.items())])

  
  @staticmethod
  def _tokenize(chunk, min_compound_weight=0,  max_compound_word_size=10000, compound=None, token2weight=None, synonyms=None, use_synonym_replacement=False, return_str=True):
    if not use_synonym_replacement: synonyms = {} 
    is_str = False
    if type(chunk) is str:
      is_str = True
      chunk = [chunk]
    ret = []
    for doc in chunk:
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
        if return_str: 
          ret.append(" ".join([d for d in doc if d]))
        else: 
          ret.append([d for d in doc if d])
    if is_str: return ret[0]
    return ret

  # doc_batch is a list of str, or a list of list of str.
  def tokenize_batch(self, doc_batch, min_compound_weight=0,  max_compound_word_size=10000, compound=None, token2weight=None, synonyms=None, use_synonym_replacement=False, return_str=True):
    if synonyms is None: synonyms = {} if not hasattr(self, 'synonyms') else self.synonyms
    if token2weight is None: token2weight = {} if not hasattr(self, 'token2weight') else self.token2weight    
    if compound is None: compound = {} if not hasattr(self, 'compound') else self.compound
    assert type(doc_batch[0]) is list
    return [self._tokenize(chunk, min_compound_weight=min_compound_weight,  max_compound_word_size=max_compound_word_size, \
                              compound=compound, token2weight=token2weight, synonyms=synonyms, use_synonym_replacement=use_synonym_replacement, \
                              return_str=return_str) for chunk in doc_batch]
    
  # doc_batch is a list of str, or a list of list of str.
  def tokenize_batch_todo(self, doc_batch, min_compound_weight=0,  max_compound_word_size=10000, compound=None, token2weight=None, synonyms=None, use_synonym_replacement=False, return_str=True):
    if synonyms is None: synonyms = {} if not hasattr(self, 'synonyms') else self.synonyms
    if token2weight is None: token2weight = {} if not hasattr(self, 'token2weight') else self.token2weight    
    if compound is None: compound = {} if not hasattr(self, 'compound') else self.compound
    if type(doc_batch) is str:
      doc_batch = [doc_batch]
    if len(doc_batch) == 1:
      return [self._tokenize(doc_batch[0], min_compound_weight=min_compound_weight,  max_compound_word_size=max_compound_word_size, \
                              compound=compound, token2weight=token2weight, synonyms=synonyms, use_synonym_replacement=use_synonym_replacement, \
                              return_str=return_str)]
    chunk_size = max(1, int(len(doc_batch)/multiprocessing.cpu_count()))
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()) 
    ret = pool.imap(partial(RiverbedTokenizer._tokenize, min_compound_weight=min_compound_weight,  max_compound_word_size=max_compound_word_size, \
                                    compound=compound, token2weight=token2weight, synonyms=synonyms, use_synonym_replacement=use_synonym_replacement, \
                                    return_str=return_str),
                                    doc_batch, chunk_size)
    return list(ret)
    
      
      
  def tokenize(self, doc, min_compound_weight=0,  max_compound_word_size=10000, compound=None, token2weight=None, synonyms=None, use_synonym_replacement=False, return_str=True):
    if synonyms is None: synonyms = {} if not hasattr(self, 'synonyms') else self.synonyms
    if token2weight is None: token2weight = {} if not hasattr(self, 'token2weight') else self.token2weight    
    if compound is None: compound = {} if not hasattr(self, 'compound') else self.compound
    if not use_synonym_replacement: synonyms = {} 
    return self._tokenize(doc, min_compound_weight=min_compound_weight,  max_compound_word_size=max_compound_word_size, \
                              compound=compound, token2weight=token2weight, synonyms=synonyms, use_synonym_replacement=use_synonym_replacement, \
                              return_str=return_str)

  def save_pretrained(self, tokenizer_name):
      os.system(f"mkdir -p {tokenizer_name}")
      pickle.dump(self, open(f"{tokenizer_name}/{tokenizer_name}.pickle", "wb"))
    
  @staticmethod
  def from_pretrained(tokenizer_name):
      self = pickle.load(open(f"{tokenizer_name}/{tokenizer_name}.pickle", "rb"))
      return self

#################################################################################
#MODEL CODE
class RiverbedModel:

  def __init__(self):
    init_models()
    pass 
  
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
      global device, clip_model, minilm_model, labse_model
      init_models()
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
          dedup_compound_words_num_iter = max(0, math.ceil(dedup_compound_words_larger_than/(5 *(doc_id+1))))
        else:
          dedup_compound_words_num_iter = 0
        num_iter = max(1,math.ceil(min_compound_word_size/(5 *(doc_id+1))))
        #we can repeatedly run the below to get long ngrams
        #after we tokenize for ngram and replace with tokens with underscores (the_projected_revenue) at each step, we redo the ngram count
        curr_arpa = {}
        print ('doing', file_name, 'num iter', num_iter, "+", dedup_compound_words_num_iter)
        prev_file = file_name
        for times in range(num_iter+dedup_compound_words_num_iter):
            print (f"iter {file_name}", times)
            if dedup_compound_words_larger_than is not None and times == dedup_compound_words_num_iter:
              # sometimes we want to do some pre-processing b/c n-grams larger than a certain amount are just duplicates
              # and can mess up our token counts
              print ('deduping compound words larger than',dedup_compound_words_larger_than)
              tmp_file_name = f"{model_name}/__tmp__{file_name}"
              if tmp_file_name.endswith(".gz"): tmp_file_name = tmp_file_name[:-len(".gz")]
              with open(f"{tmp_file_name}", "w", encoding="utf8") as tmp2:
                deduped_num_tokens = 0
                seen_dedup_compound_words = {}
                if file_name.endswith(".gz"):
                  f = gzip.open(file_name)
                else:
                  f = open(file_name, "rb")
                while True:
                  orig_batch = batchify_from_fileobj(f, apply_fn=lambda a: a.replace("\\n", "\n").strip())
                  if not orig_batch: break
                  batch = tokenizer.tokenize_batch(orig_batch, min_compound_weight=0, compound=compound, token2weight=token2weight,  synonyms=synonyms, use_synonym_replacement=False)
                  for orig_chunk, chunk in zip(orig_batch, batch):
                    for orig_l, l in zip(orig_chunk, chunk):
                      l = l.split()
                      dedup_compound_word = [w for w in l if "_" in w and w.count("_") + 1 > dedup_compound_words_larger_than]
                      if not dedup_compound_word:
                        l2 = " ".join(l).replace("_", " ").strip()
                        tmp2.write(l2+"\n")
                        continue
                      l = [w if ("_" not in w or w.count("_") + 1 <= dedup_compound_words_larger_than or w not in seen_dedup_compound_words) else '...' for w in l]
                      l2 = " ".join(l).replace(' ... ...', ' ...').strip()
                      if dedup_compound_word and l2 != orig_l:
                        deduped_num_tokens += 1
                      l2 = l2.replace("_", " ")
                      if l2.endswith(" ..."): l2 = l2[:-len(" ...")]
                      for w in dedup_compound_word:
                        seen_dedup_compound_words[w] = 1
                      tmp2.write(l2+"\n")
                seen_dedup_compound_words = None
                f.close()
                print ('finished deduping', deduped_num_tokens)
                dedup_name = f"{model_name}/{file_name}.dedup"
                if dedup_name.endswith(".gz"): dedup_name = dedup_name[:-len(".gz")]
                os.system(f"cp {tmp_file_name} {dedup_name}")  
                os.system(f"gzip {dedup_name}")
                prev_file = f"{dedup_name}.gz"       
                curr_arpa = {}
                
            # we only do synonym and embedding creation as the second to last or last step of each file processed 
            # b/c this is very expensive. we should do this right before the last counting if we
            # do synonym replacement so we have a chance to create syonyms for the replacement.
            # otherwise, we should do it after the last count. See below.
            synonyms_created=  False          
            if use_synonym_replacement and times == num_iter+dedup_compound_words_num_iter-1 and token2weight:
                synonyms_created = True
                self.synonyms = self.tokenizer.synonyms = synonyms = self._create_token_embeds_and_synonyms(model_name, stopwords=stopwords, token2weight=token2weight, synonyms=synonyms, kmeans_batch_size=kmeans_batch_size, min_incremental_cluster_overlap=min_incremental_cluster_overlap, \
                  prefered_cluster_size=prefered_cluster_size, embedder=embedder, embed_batch_size=embed_batch_size, min_prev_ids=min_prev_ids, max_ontology_depth=max_ontology_depth, max_cluster_size=max_cluster_size, do_ontology=do_ontology, recluster_type=recluster_type)   
            if token2weight:
              # if we want to make this faster, we can parrallelize this in a pool
              tmp_file_name = f"{model_name}/__tmp__{file_name}"
              if tmp_file_name.endswith(".gz"): tmp_file_name = tmp_file_name[:-len(".gz")]
              with open(tmp_file_name, "w", encoding="utf8") as tmp2:
                if prev_file.endswith(".gz"):
                  f = gzip.open(prev_file)
                else:
                  f = open(prev_file, "rb")
                while True:
                  batch = batchify_from_fileobj(f, apply_fn=lambda a: a.replace("\\n", "\n").strip())
                  if not batch: break
                  batch = tokenizer.tokenize_batch(batch,  min_compound_weight=min_compound_weight, compound=compound, token2weight=token2weight, synonyms=synonyms, use_synonym_replacement=use_synonym_replacement)
                  if times == num_iter-1:
                    batch = tokenizer.tokenize_batch(batch, min_compound_weight=0, compound=compound, token2weight=token2weight,  synonyms=synonyms, use_synonym_replacement=use_synonym_replacement)
                  for chunk in batch: 
                    tmp2.write("\n".join(chunk)+"\n")
              f.close()
              os.system(f"gzip {tmp_file_name}")
              prev_file = f"{tmp_file_name}.gz" 
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
                    if len(line) > 1 and (not dedup_compound_words_larger_than  or  times >= dedup_compound_words_num_iter):
                      weight2 = weight
                      if weight2 > 0: weight2 = 0
                      tmp_arpa.write(f"{n}\t{line[1]}\t{weight2}\n")
                    #print (line
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
            if times == num_iter+dedup_compound_words_num_iter-1  and not synonyms_created:
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

    
#################################################################################
# SPAN AND DOCUMENT PROCESSOR
# includes labeling of spans of text with different features, including clustering
# assumes each batch is NOT shuffeled.    
class RiverbedDocumentProcessor:
  def __init__(self):
    pass
  
  def tokenize(self, *args, **kwargs):
    return self.tokenizer.tokenize(*args, **kwargs)

  @staticmethod
  def dateutil_parse_ext(text):
    try: 
      int(text.strip())
      return None
    except:
      pass
    try:
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

  # the similarity models sometimes put too much weight on proper names, etc. but we might want to cluster by general concepts
  # such as change of control, regulatory actions, etc. The proper names themselves can be collapsed to one canonical form (The Person). 
  # Similarly, we want similar concepts (e.g., compound words) to cluster to one canonical form.
  # we do this by collapsing to an NER label and/or creating a synonym map from compound words to known tokens. See _create_ontology
  # and we use that data to simplify the sentence here.  
  # TODO: have an option NOT to simplify the prefix. 
  def _simplify_text(self, text, ents, ner_to_simplify=(), use_synonym_replacement=False):
    if not ner_to_simplify and not synonyms and not ents: return text, ents
    # assumes the text has already been tokenized and replacing NER with @#@{idx}@#@ 
    tokenized_text = text
    #do a second tokenize if we want to do synonym replacement.
    if use_synonym_replacement:
      tokenized_text = self.tokenize(text, use_synonym_replacement=True)  
    ents2 = []

    for idx, ent in enumerate(ents):
        entity, label = ent
        if "@#@" not in text: break
        if f"@#@{idx}@#@" not in text: continue
        text = text.replace(f"@#@{idx}@#@", entity) 
    text = text.replace("_", " ")

    for idx, ent in enumerate(ents):
        entity, label = ent
        if "@#@" not in tokenized_text: break
        if f"@#@{idx}@#@" not in tokenized_text: continue
        ents2.append((entity, label,  text.count(f"@#@{idx}@#@")))
        if label in ner_to_simplify:   
          if label == 'ORG':
            tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Organization')
          elif label == 'PERSON':
            tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Person')
          elif label == 'FAC':
            tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Facility')
          elif label in ('GPE', 'LOC'):
            tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Location')
          elif label in ('DATE', ):
            tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Date')
          elif label in ('LAW', ):
            tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Law')  
          elif label in ('EVENT', ):
            tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Event')            
          elif label in ('MONEY', ):
            tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Amount')
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
      
    return text, tokenized_text, ents2
  
  #transform a doc batch into a span batch, breaking up doc into spans
  #all spans/leaf nodes of a cluster are stored as a triple of (file_name, lineno, offset)
  def _create_spans_batch(self, curr_file_size, batch, text_span_size=1000, ner_to_simplify=(), use_synonym_replacement=False):
      batch2 = []
      for idx, span in enumerate(batch):
        file_name, curr_lineno, ents, text  = span['file_name'], span['lineno'], span['ents'], span['text']
        for idx, ent in enumerate(ents):
          text = text.replace(ent[0], f' @#@{idx}@#@ ')
        # we do placeholder replacement tokenize to make ngram tokens underlined, so that we don't split a span in the middle of an ner token or ngram.
        text  = self.tokenize(text, use_synonym_replacement=False) 
        len_text = len(text)
        prefix = ""
        if "||" in text:
          prefix, _ = text.split("||",1)
          prefix = prefix.strip()
        offset = 0
        while offset < len_text:
          max_rng  = min(len_text, offset+text_span_size+1)
          if text[max_rng-1] != ' ':
            # extend for non english periods and other punctuations
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
          text2, tokenized_text, ents2 = self._simplify_text(text2, ents, ner_to_simplify, use_synonym_replacement=use_synonym_replacement) 
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

  def _create_cluster_for_spans(self, true_k, batch_id_prefix, spans, cluster_vecs, tmp_clusters, span2cluster_label,  idxs, \
                                span_per_cluster=20, kmeans_batch_size=1024, ):
    global device
    if device == 'cuda':
      kmeans = KMeans(n_clusters=true_k, mode='cosine')
      km_labels = kmeans.fit_predict(torch.from_numpy(cluster_vecs[idxs]).to(device))
      km_labels = [l.item() for l in km_labels.cpu()]
    else:
      km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                                    init_size=max(true_k*3,1000), batch_size=1024).fit(cluster_vecs[idxs])
      km_labels = km.labels_
      
    new_cluster = {}
    for span, label in zip(spans, km_labels):
      label = batch_id_prefix+str(label)
      new_cluster[label] = new_cluster.get(label, [])+[span]
      
    if not tmp_clusters: 
      tmp_clusters = new_cluster
      for label, items in tmp_clusters.items():
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
          if span not in tmp_clusters.get(label, []):
              tmp_clusters[label] = tmp_clusters.get(label, []) + [span]
          span2cluster_label[span] = label
    return tmp_clusters, span2cluster_label 

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

  def _create_informative_label_and_tfidf(self, batch, batch_id_prefix, tmp_clusters, span2idx, tmp_span2batch, span2cluster_label, \
                                          label2tf=None, df=None, domain_stopwords_set=stopwords_set,):
    # code to compute tfidf and more informative labels for the span clusters
    if label2tf is None: label2tf = {}
    if df is None: df = {}
    label2label = {}
    #we gather info for tf-idf with respect to each token in each clusters
    for label, values in tmp_clusters.items(): 
      if label.startswith(batch_id_prefix):
        for item in values:
          if span in span2idx:
            span = tmp_span2batch[span]
            text = span['tokenized_text']
            #we don't want the artificial labels to skew the tf-idf calculations
            text = text.replace('The Organization','').replace('The_Organization','')
            text = text.replace('The Person','').replace('The_Person','')
            text = text.replace('The Facility','').replace('The_Facility','')
            text = text.replace('The Location','').replace('The_Location','')          
            text = text.replace('The Date','').replace('The_Date','')
            text = text.replace('The Law','').replace('The_Law','')
            text = text.replace('The Amount','').replace('The_Amount','')
            text = text.replace('The Event','').replace('The_Event','')
            #we add back the entities we had replaced with the artificial labels into the tf-idf calculations
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
            aHash = label2tf[label] =  label2tf.get(label, {})
            for token, cnt in cnts.items():
              aHash[token] = cnt/len_text
            for token in cnts.keys():
              df[token] = df.get(token,0) + 1
      
    #Now, acually create a new label from the tfidf of the tokens in this cluster
    #TODO, see how we might save away the tf-idf info as features, then we would need to recompute the tfidf if new items are added to cluster
    label2label = {}
    for label, tf in label2tf.items():
      if label.startswith(batch_id_prefix):
        tfidf = copy.copy(tf)    
        for token in list(tfidf.keys()):
          tfidf[token]  = tfidf[token] * min(1.5, self.tokenizer.token2weight.get(token, 1)) * math.log(1.0/(1+df[token]))
        top_tokens2 = [a[0].lower().strip("~!@#$%^&*()<>,.:;")  for a in Counter(tfidf).most_common(min(len(tfidf), 40))]
        top_tokens2 = [a for a in top_tokens2 if a not in domain_stopwords_set and ("_" not in a or (a.count("_")+1 != len([b for b in a.split("_") if  b in domain_stopwords_set])))]
        top_tokens = []
        for t in top_tokens2:
          if t not in top_tokens:
            top_tokens.append(t)
        if top_tokens:
          if len(top_tokens) > 5: top_tokens = top_tokens[:5]
          label2 = ", ".join(top_tokens) 
          label2label[label] = label2
          
    #swap out the labels
    for old_label, new_label in label2label.items():
      if new_label != old_label:
        if old_label in tmp_clusters:
          a_cluster = tmp_span2batch[old_label]
          for item in a_cluster:
            span2cluster_label[item] = new_label
        label2tf[new_label] =  copy.copy(label2tf.get(old_label, {}))
        del label2tf[old_label] 
    for label, values in tmp_clusters.items():          
      spans = [span for span in values if span in span2idx]
      for span in spans:
        tmp_span2batch[span]['cluster_label'] = label
        
    # add before and after label as additional features
    prior_b = None
    for b in batch:
      if prior_b is not None:
        b['cluster_label_before'] = prior_b['cluster_label']
        prior_b['cluster_label_after'] = b['cluster_label']
      prior_b = b
      
    return batch, label2tf, df
  
  # similar to _create_token_embeds_and_synonyms, except for spans     
  #(1) compute features and embeddings in one batch for tokenized text.
  #(2) create clusters in an incremental fashion from batch
  #all leaf nodes are spans
  #spanf2idx is a mapping from the span to the actual underlying storage idx (e.g., a jsonl file or database)
  #span2cluster_label is like the synonym data-structure for tokens.
  def _create_span_embeds_and_span2cluster_label(self, project_name, curr_file_size, jsonl_file_idx, span2idx, batch, retained_batch, \
                                                      jsonl_file, batch_id_prefix, span_lfs,  span2cluster_label, \
                                                      text_span_size=1000, kmeans_batch_size=50000, epoch = 10, \
                                                      embed_batch_size=7000, min_prev_ids=10000, embedder="minilm", \
                                                      max_ontology_depth=4, max_top_parents=10000, do_ontology=True, \
                                                      running_features_per_label={}, ner_to_simplify=(), span_level_feature_extractors=default_span_level_feature_extractors, \
                                                      running_features_size=100, label2tf=None, df=None, domain_stopwords_set=stopwords_set,\
                                                      verbose_snrokel=False,  span_per_cluster=10, use_synonym_replacement=False, ):
    
    #transform a doc batch into a span batch, breaking up doc into spans
    batch = self._create_spans_batch(curr_file_size, batch, text_span_size=text_span_size, ner_to_simplify=ner_to_simplify, use_synonym_replacement=use_synonym_replacement)
    
    #create features, assuming linear spans.
    batch = self._create_span_features(batch, span_level_feature_extractors, running_features_per_label, running_features_size)
    
    #add the current back to the span2idx data structure
    start_idx_for_curr_batch = len(span2idx)
    tmp_span2batch = {}
    tmp_idx2span = {}
    tmp_batch_idx_in_span2cluster = []
    tmp_batch_idx_not_in_span2cluster = []
    for b in retained_batch + batch :
      span = (b['file_name'], b['lineno'], b['offset'])
      tmp_span2batch[span] = b
      if span not in span2idx:
        b['idx']= span2idx[span] = len(span2idx)
      else:
        b['idx']= span2idx[span]
      if b['idx'] in span2cluster_label:
        tmp_batch_idx_in_span2cluster.append(b['idx'])
      else:
        tmp_batch_idx_not_in_span2cluster.append(b['idx'])
      tmp_idx2span[b['idx']] = span
      
    if embedder == "clip":
      embed_dim = clip_model.config.text_config.hidden_size
    elif embedder == "minilm":
      embed_dim = minilm_model.config.hidden_size
    elif embedder == "labse":
      embed_dim = labse_model.config.hidden_size
    cluster_vecs = np_memmap(f"{project_name}/{project_name}.{embedder}_spans", shape=[len(span2idx), embed_dim])

    for rng in range(0, len(batch), embed_batch_size):
      max_rng = min(len(batch), rng+embed_batch_size)
      if embedder == "clip":
        toks = clip_processor([a['tokenized_text'].replace("_", " ") for a in batch[rng:max_rng]], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
          cluster_vecs = clip_model.get_text_features(**toks).cpu().numpy()
      elif embedder == "minilm":
        toks = minilm_tokenizer([a['tokenized_text'].replace("_", " ") for a in batch[rng:max_rng]], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
          cluster_vecs = minilm_model(**toks)
          cluster_vecs = mean_pooling(cluster_vecs, toks.attention_mask).cpu().numpy()
      elif embedder == "labse":
        toks = labse_tokenizer([a['tokenized_text'].replace("_", " ") for a in batch[rng:max_rng]], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
          cluster_vecs = labse_model(**toks).pooler_output.cpu().numpy()  
      cluster_vecs = np_memmap(f"{project_name}/{project_name}.{embedder}_spans", shape=[len(span2idx), embed_dim],  dat=cluster_vecs, idxs=range(len(span2idx)-len(batch)+rng, len(span2idx)-len(batch)+max_rng))  
    
    len_batch = len(tmp_batch_idx_not_in_span2cluster)
    for rng in range(0, len_batch, int(kmeans_batch_size*.7)):
        max_rng = min(len_batch, rng+int(kmeans_batch_size*.7))
        if rng > 0:
          prev_ids = [idx for idx in tmp_batch_idx_not_in_span2cluster[:rng] if tmp_idx2span[idx] not in span2cluster_label]
          tmp_batch_idx_in_span2cluster.extend( [idx for idx in tmp_batch_idx_not_in_span2cluster[:rng] if tmp_idx2span[idx] in span2cluster_label])
          tmp_batch_idx_in_span2cluster = list(set(tmp_batch_idx_in_span2cluster))
          if len(prev_ids) > kmeans_batch_size*.3: prev_ids.extend(random.sample(range(0, rng), (kmeans_batch_size*.3)-len(prev_ids)))
          #TODO: add some more stuff from tmp_batch_idx_in_span2cluster
        else:
          prev_ids = []
        idxs = prev_ids + [tmp_batch_idx_not_in_span2cluster[idx] for idx in range(rng, max_rng)]
        print (len(idxs))
        true_k=int((len(idxs)/span_per_cluster))
        spans2 = [tmp_idx2span[idx] or idx in idxs]
        tmp_clusters, span2cluster_label = self._create_cluster_for_spans(true_k, batch_id_prefix, spans2, cluster_vecs, tmp_clusters, idxs, span2cluster_label, span_per_cluster=span_per_cluster, domain_stopwords_set=domain_stopwords_set)
        # TODO: recluster
    
    # TODO: create_span_ontology
                   
    # create more informative labels                   
    batch, label2tf, df = self._create_informative_label_and_tfidf(batch, batch_id_prefix, tmp_clusters, span2idx, tmp_span2batch, span2cluster_label, label2tf, df)
    
    # at this point, batch should have enough data for all snorkel labeling functions
    if span_lfs:
      df_train = pd.DataFrom(batch)
      for span_label, lfs, snorkel_label_cardinality, snorkel_epochs in span_lfs:
        # we assume there is no shuffling, so we can tie back to the original batch
        applier = PandasLFApplier(lfs=fs)
        L_train = applier.apply(df=df_train)
        label_model = LabelModel(cardinality=snorkel_label_cardinality, verbose=verbose_snrokel)
        label_model.fit(L_train=L_train,n_epochs=snorkel_epochs)
        for idx, label in enumerate(label_model.predict(L=L_train,tie_break_policy="abstain")):
          batch[idx][span_label] = label
        # note, we only use these models once, since we are doing this in an incremental fashion.
        # we would want to create a final model by training on all re-labeled data from the jsonl file
    
    # all labeling and feature extraction is complete, and the batch has all the info. now save away the batch
    for b in batch:
      if b['idx'] >= start_idx_for_curr_batch:
        jsonl_file.write(json.dumps(b)+"\n")
        #TODO, replace with a datastore abstraction, such as sqlite
    
    # add stuff to the retained batches
                   
    return retained_batch, span2idx, span2cluster_label, label2tf, df   

  # the main method for processing documents and their spans. 
  def apply_span_feature_detect_and_labeling(self, project_name, files, text_span_size=1000, max_lines_per_section=10, max_len_for_prefix=100, \
                                                min_len_for_prefix=20, embed_batch_size=100, 
                                                features_batch_size = 10000000, kmeans_batch_size=1024, \
                                                span_per_cluster= 20, retained_spans_per_cluster=5, min_prev_ids=10000, \
                                                ner_to_simplify=(), span_level_feature_extractors=default_span_level_feature_extractors, running_features_size=100, \
                                                prefix_extractors = default_prefix_extractors, dedup=True, max_top_parents=10000, \
                                                span_lfs = [], verbose_snrokel=True, use_synonym_replacement=False, max_ontology_depth=4, \
                                                batch_id_prefix = 0, seen = None,  embedder="minilm", \
                                                auto_create_tokenizer_and_model=True, \
                                                ):
    global clip_model, minilm_model, labse_model
    init_models()
    model = self.model
    tokenizer = self.tokenizer
    os.system(f"mkdir -p {project_name}")
    span2idx = self.span2idx = OrderedDict() if not hasattr(self, 'span2idx') else self.span2idx
    span_clusters = self.span_clusters = {} if not hasattr(self, 'span_clusters') else self.span_clusters
    label2tf = self.label2tf = {} if not hasattr(self, 'label2tf') else self.label2tf
    df = self.df = {} if not hasattr(self, 'df') else self.df
    span2cluster_label = self.span2cluster_label = {} if not hasattr(self, 'span2cluster_label') else self.span2cluster_label
    label_models = self.label_models = {} if not hasattr(self, 'label_models') else self.label_models
    if (not hasattr(model, 'kenlm_model') or model.kenlm_model is not None) and auto_create_tokenizer_and_model:
      tokenizer, model = self.tokenizer, self.model = RiverbedModel.create_tokenizer_and_model(project_name, files, )
    kenlm_model = self.model.kenlm_model 
      
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

   
    running_features_per_label = {}
    file_name = files.pop()
    f = open(file_name) 
    domain_stopwords_set = set(list(stopwords_set) + list(stopwords.keys()))
    prior_line = ""
    batch = []
    retained_batch = []
    curr = ""
    cluster_vecs = None
    curr_date = ""
    curr_position = 0
    next_position = 0
    curr_file_size = os.path.getsize(file_name)
    position = 0
    line = ""
    lineno = -1
    curr_lineno = 0

    if seen is None: seen = {}
    
    with open(f"{project_name}/{project_name}.jsonl", "w", encoding="utf8") as jsonl_file:
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
          batch_id_prefix += 1
          retained_batch, span2idx, span2cluster_label, label2tf, df = self._create_span_embeds_and_span2cluster_label(project_name, curr_file_size, jsonl_file_idx, span2idx, batch, \
                                                      retained_batch, jsonl_file,  f"{batch_id_prefix}_", span_lfs,  span2cluster_label, text_span_size, \
                                                      kmeans_batch_size=kmeans_batch_size, epoch = epoch, embed_batch_size=embed_batch_size, min_prev_ids=min_prev_ids, \
                                                      max_ontology_depth=max_ontology_depth, max_top_parents=max_top_parents, do_ontology=True, embedder=embedder, \
                                                      running_features_per_label=running_features_per_label, ner_to_simplify=ner_to_simplify, span_level_feature_extractors=span_level_feature_extractors, \
                                                      running_features_size=running_features_size, label2tf=label2tf, df=df, domain_stopwords_set=domain_stopwords_set, \
                                                      verbose_snrokel=verbose_snrokel,  span_per_cluster=span_per_cluster, use_synonym_replacement=use_synonym_replacement, )  
          batch = []
      
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
          batch_id_prefix += 1
          retained_batch, span2idx, span2cluster_label, label2tf, df = self._create_span_embeds_and_span2cluster_label(project_name, curr_file_size, jsonl_file_idx, spanf2idx, batch, \
                                                      retained_batch, jsonl_file,  f"{batch_id_prefix}_", span_lfs,  span2cluster_label, text_span_size, \
                                                      kmeans_batch_size=kmeans_batch_size, epoch = epoch, embed_batch_size=embed_batch_size, min_prev_ids=min_prev_ids,  \
                                                      max_ontology_depth=max_ontology_depth, max_top_parents=max_top_parents, do_ontology=True, embedder=embedder,\
                                                      running_features_per_label=running_features_per_label, ner_to_simplify=ner_to_simplify, span_level_feature_extractors=span_level_feature_extractors, \
                                                      running_features_size=running_features_size, label2tf=label2tf, df=df, domain_stopwords_set=domain_stopwords_set, \
                                                      verbose_snrokel=verbose_snrokel)  
          batch = []
      
    #now create global labeling functions based on all the labeled data
    #have an option to use a different labeling function, such as regression trees. 
    #we don't necessarily need snorkel lfs after we have labeled the dataset.
    if span_lfs:
      df_train = pd.DataFrame(f"{project_name}/{project_name}.jsonl").shuffle()
      for span_label, lfs, snorkel_label_cardinality, snorkel_epochs in span_lfs:
        applier = PandasLFApplier(lfs=lfs)
        L_train = applier.apply(df=df_train)
        label_models.append(span_label, LabelModel(cardinality=snorkel_label_cardinality,verbose=verbose_snrokel))
    
    span2idx, span_clusters, label2tf, df, span2cluster_label, label_models = self.span2idx, self.span_clusters, self.label2tf, self.df, self.span2cluster_label, self.label_models                    
    return self

  def save_pretrained(self, project_name):
      os.system(f"mkdir -p {project_name}")
      pickle.dump(self, open(f"{project_name}/{project_name}.pickle", "wb"))
    
  @staticmethod
  def from_pretrained(project_name):
      self = pickle.load(open(f"{project_name}/{project_name}.pickle", "rb"))
      return self
