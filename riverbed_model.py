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
from .riverbed_tokenizer import RiverbedTokenizer

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


  
#################################################################################
#BASIC MODEL
class RiverbedModel(nn.Module):
 
  def __init__(self, project_name, embedder):
    super().__init__()
    global labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set
    self.project_name  = project_name 
    labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set = init_models()
    self.searcher = SearcherIndexer(idx_dir=project_name+"/word_searcher", embedder=embedder)
    self.synonyms = None
    self.kenlm_model = None
  
  def search(self, *args, **kwargs):
    self.searcher.search(*args, **kwargs)
  
  def forward(self, *args,**kwargs):
    return self.searcher(*args, **kwargs)

  #vector and perpexity guided generation      
  #generate text based simiarilty between vectors found through search and lowering the 
  #perplexity with respect to pretraining corpus trained with the kenlm model.
  def generate(self, gen_model, gen_tokenizer, target=None, do_perplexity=True,
                             prompt="", encoder_input=None, num_input_words=5, \
                             num_words_per_step=4, num_return_sequences=4, max_length=20, top_p=0.95,
                             top_k=10, update_target=True, *args, **kwargs):
    assert prompt or target is not None
    p = next(self.parameters())
    if encoder_input is not None:
      encoder_input = [encoder_input] * num_return_sequences
    if target is None and prompt and prompt != gen_tokenizer.bos_token:
      target = self.searcher.get_embeddings(prompt, model=gen_model, tokenizer=gen_tokenizer)
    if not prompt and target is not None:
        current_generated_sentences = [", ".join([a['text'] for a in self.search(target=target, limit=num_input_words)])]
    else:
      if not prompt: prompt = gen_tokenizer.bos_token+""
      current_generated_sentences = [prompt] * num_return_sequences

    end_length = max(max_length, len_words*num_words_per_step)
    embed_array = None
    for mlength in range(num_return_sequences, end_length, num_words_per_step):
      if not encoder_input:
        inputs = gen_tokenizer(current_generated_sentences, padding=True,  return_tensors="pt").to(device=p.device)
      else:
        inputs = gen_tokenizer(encoder_input, padding=True,  return_tensors="pt").to(device=p.device)
        inputs['decoder_input_ids'] = gen_tokenizer(current_generated_sentences, padding=True, return_tensors="pt", add_special_tokens=False).input_ids.to(p.device)
      out = gen_model.generate(num_return_sequences=num_return_sequences,
                                      top_p=top_p,
                                      top_k=top_k,
                                      do_sample=True, max_length=mlength, **input)
      #out will return num_return_sequences * num_return_sequences sequences                                    
      text_array = gen_tokenizer.batch_decode(out, skip_special_tokens=True)
      text_array = list(set(text_array))
      embedding_array = self.searcher.get_embedding(text_array)
      if target is None:
         # since we have no target, we assume we are doing seq2seq and starting with the bos token. 
         # we only have perplexity to guide us with extra info.
         assert do_perplexity
         text_array2 = [[(50/(min(self.get_perplexity(self.tokenize(text_array[idx])),self.get_perplexity(text_array[idx]))), text_array[idx], embedding_array[idx]] for idx in range(len(text_array))]
      else:
         
         dist = cosine_distance(target, embedding_array)
         # how to tie back to the original target??
         result = dist.top_k(num_return_sequences*2)
         text_array2 = []
         embed_array= []
         for score, idx in zip(result.values(), result.indices()):
           text_array2.append([score + (50/min(self.get_perplexity(self.tokenize(text_array[idx])),self.get_perplexity(text_array[idx]))), text_array[idx], embedding_array[idx]])
         text_array2.sort(key=lambda a: a[0], reverse=True)[:top_k]
      update_target_array =  [a[2] for a in text_array2]
      text_array2 = [a[1] for a in text_array2] 

      if mlength < end_length:
          for i in range(len(text_array2):
            sent  = text_array2[i]
            if target is not None:
              update_target_array[i] += torch.mean(self.get_embedding_by_idx([a['id'] for a in self.search(sent, limit=5)]))
  
    return

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
  
  
