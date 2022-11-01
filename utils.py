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
import itertools
import indexed_gzip as igzip
import pickle
import threading
import io
import os
import copy
from whoosh.analysis import StemmingAnalyzer
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser
import whoosh.index as whoosh_index
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from fast_pytorch_kmeans import KMeans
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
import random
import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel, BertTokenizerFast, CLIPProcessor, CLIPModel, \
                        BertModel, T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel
from nltk.corpus import stopwords as nltk_stopwords
from torch import nn
import spacy
from collections import OrderedDict
import multiprocessing
import math 
import json
from dateutil.parser import parse as dateutil_parse
import seaborn as sns
from tsnecuda import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

in_notebook = 'google.colab' in sys.modules
if not in_notebook:
    try:
        get_ipython()
    except:
      in_notebook = False
      
if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'
  
try:
  if minilm_model is not None: 
    pass
except:
   doc2query_tokenizer = doc2query_model = doc2query_encoder = codebert_tokenizer = codebert_model = labse_tokenizer= labse_model=  clip_processor = minilm_tokenizer= clip_model= minilm_model= spacy_nlp= stopwords_set = None

def get_spacy():
  global spacy_nlp
  if spacy_nlp is None: init_models()
  return spacy_nlp

def get_stopwords():
  global stopwords_set
  if stopwords_set is None: init_models()
  return stopwords_set

def get_doc2query_tokenizer_and_model():
  global doc2query_tokenizer, doc2query_model
  if doc2query_model is None:
    init_models()
  return doc2query_tokenizer, doc2query_model


def init_models(embedder=None):    
  global doc2query_tokenizer, doc2query_model, doc2query_encoder, codebert_tokenizer, codebert_model, labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set
  if doc2query_model is None:

    codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")   
    minilm_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    labse_tokenizer = BertTokenizerFast.from_pretrained("sentence-transformers/LaBSE")
    doc2query_tokenizer = T5Tokenizer.from_pretrained('doc2query/all-t5-base-v1')

    # we will always keep the doc2query model in gpu
    if device == 'cuda':
      doc2query_model = T5ForConditionalGeneration.from_pretrained('doc2query/all-t5-base-v1').half().eval().to(device)
      doc2query_encoder = T5EncoderModel.from_pretrained('doc2query/all-t5-base-v1').half().eval()
      #share the parameter so we don't waste memory
      doc2query_encoder.shared = doc2query_model.shared
      doc2query_encoder.encoder = doc2query_model.encoder
      doc2query_encoder = doc2query_encoder.to(device)
    else:
      doc2query_model = T5ForConditionalGeneration.from_pretrained('doc2query/all-t5-base-v1').eval()
      doc2query_encoder = T5EncoderModel.from_pretrained('doc2query/all-t5-base-v1').eval()
      #share the parameter so we don't waste memory
      doc2query_encoder.shared = doc2query_model.shared
      doc2query_encoder.encoder = doc2query_model.encoder
      doc2query_encoder = doc2query_encoder

    spacy_nlp = spacy.load('en_core_web_md')
    stopwords_set = set(nltk_stopwords.words('english') + ['...', 'could', 'should', 'shall', 'can', 'might', 'may', 'include', 'including'])
  if embedder is not None: use_model(embedder)
  return  doc2query_tokenizer, doc2query_model, doc2query_encoder, codebert_tokenizer, codebert_model, labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set

def use_model(embedder):
  global doc2query_tokenizer, doc2query_model, doc2query_encoder, codebert_tokenizer, codebert_model, labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set
  if embedder == "clip":
    if minilm_model is not None: minilm_model =  minilm_model.cpu()
    if labse_model is not None: labse_model =  labse_model.cpu()
    if codebert_model is not None: codebert_model =  codebert_model.cpu()
    if clip_model is None:
      if device == 'cuda':
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").half().eval()
      else:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    clip_model = clip_model.to(device)
  elif embedder == "minilm":
    if clip_model is not None: clip_model =  clip_model.cpu()
    if labse_model is not None: labse_model =  labse_model.cpu()
    if codebert_model is not None: codebert_model =  codebert_model.cpu()
    if minilm_model is None:
      if device == 'cuda':
        minilm_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').half().eval()
      else:
        minilm_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').eval()
    minilm_model = minilm_model.to(device)
  elif embedder == "codebert":
    if clip_model is not None: clip_model =  clip_model.cpu()
    if labse_model is not None: labse_model =  labse_model.cpu()
    if minilm_model is not None: minilm_model =  minilm_model.cpu()
    if codebert_model is None:
      if device == 'cuda':
        codebert_model = AutoModel.from_pretrained('microsoft/graphcodebert-base').half().eval()
      else:
        codebert_model = AutoModel.from_pretrained('microsoft/graphcodebert-base').eval()
    codebert_model = codebert_model.to(device)
  elif embedder == "labse":
    if clip_model is not None: clip_model =  clip_model.cpu()
    if codebert_model is not None: codebert_model =  labse_model.cpu()
    if minilm_model is not None: minilm_model =  minilm_model.cpu()
    if labse_model is None:
      if device == 'cuda':
        labse_model = BertModel.from_pretrained("sentence-transformers/LaBSE").half().eval()
      else:
        labse_model = BertModel.from_pretrained("sentence-transformers/LaBSE").eval()
    labse_model = labse_model.to(device)
  elif embedder == "doc2query":
    # we will always keep the doc2query model in gpu
    if clip_model is not None: clip_model =  clip_model.cpu()
    if codebert_model is not None: codebert_model =  labse_model.cpu()
    if minilm_model is not None: minilm_model =  minilm_model.cpu()
    if labse_model is not None: labse_model =  labse_model.cpu()
        

def apply_model(embedder, sent):  
  global doc2query_tokenizer, doc2query_model, doc2query_encoder, codebert_tokenizer, codebert_model, labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set
  def get_one_embed(sent, embedder):
    if embedder == "clip":
        toks = clip_processor(sent, padding=True, truncation=True, return_tensors="pt").to(device)
        dat = clip_model.get_text_features(**toks)
    elif embedder == "minilm":
        toks = minilm_tokenizer(sent, padding=True, truncation=True, return_tensors="pt").to(device)
        dat = minilm_model(**toks)
        dat = mean_pooling(dat, toks.attention_mask)
    elif embedder == "codebert":
        toks = codebert_tokenizer(sent, padding=True, truncation=True, return_tensors="pt").to(device)
        dat = codebert_model(**toks)
        dat = mean_pooling(dat, toks.attention_mask)
    elif embedder == "labse":
        toks = labse_tokenizer(sent, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        dat = labse_model(**toks).pooler_output 
    elif embedder == "doc2query":
        toks = doc2query_tokenizer(sent, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        dat = doc2query_encoder(**toks)
        dat = mean_pooling(dat, toks.attention_mask)  
    return dat
  ###
  if type(sent) is str:
    return get_one_embed(sent, embedder)
  else:
    #poor man length batched breaking by length 1000
    #print (len(sent))
    batch = []
    all_dat = []
    doing_800 = False
    for s in sent:
      if not doing_800 and len(s) >= 800:
        if batch:
          dat = get_one_embed(batch, embedder)
          all_dat.append(dat)
        batch = [s]
        doing_800 = True
      elif doing_800 and len(s) < 800:
        if batch:
          dat = get_one_embed(batch, embedder)
          all_dat.append(dat)
        batch = [s]
        doing_800 = False
      else:
        batch.append(s)
    if batch:
      dat = get_one_embed(batch, embedder)
      all_dat.append(dat)
    if len(all_dat) == 1: return all_dat[0]
    return torch.vstack(all_dat)    

def get_model_embed_dim(embedder):
  global doc2query_tokenizer, doc2query_model, doc2query_encoder, codebert_tokenizer, codebert_model, labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set
  if embedder == "clip":
    return clip_model.config.text_config.hidden_size
  elif embedder == "minilm":
    return minilm_model.config.hidden_size
  elif embedder == "codebert":
    return codebert_model.config.hidden_size  
  elif embedder == "labse":
    return labse_model.config.hidden_size   
  elif embedder == "doc2query":
    return doc2query_model.config.hidden_size 

def _dateutil_parse_ext(text):
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

def get_np_mmap_length(filename, shape, dtype=np.float16, ):
  if not os.path.exists(filename):
    return shape[0]
  else:
    size = np.dtype(dtype).itemsize*np.prod(shape[1:])
    return int(os.path.getsize(filename)/size)

def np_memmap(filename, dat=None, idxs=None, shape=None, dtype=np.float16, offset=0, order='C' ):
  if not filename.endswith(".mmap"):
    filename = filename+".mmap"
  if os.path.exists(filename):
    mode = "r+"
  else:
    mode = "w+"
  if shape is None and dat is not None: 
    shape = dat.shape
  if not shape: shape = [0, 1]
  memmap = np.memmap(filename, mode=mode, dtype=dtype, shape=tuple(shape), offset=offset, order=order)
  if dat is None:
    return memmap
  if tuple(shape) == tuple(dat.shape):
    memmap[:] = dat
  else:
    memmap[idxs] = dat
  return memmap


#Mean Pooling - Take attention mask into account for correct averaging
#TODO, mask out the prefix for data that isn't the first portion of a prefixed text.
def mean_pooling(model_output, attention_mask):
    with torch.no_grad():
      token_embeddings = model_output.last_hidden_state
      input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
      return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def _apply_temperature(dat,temperature):
  #print ('apply temp', temperature)
  dtype = dat.dtype
  return nn.functional.softmax(dat.float() / temperature, dim=1).to(dtype)

#helper function for getting embeddings
# for embedding saved into disk (embed_text), we don't use a temperature
# for search, we can vary the temperature  of the target vector change the focus of the search.
def _get_embeddings(sent, downsampler=None, embedder="minilm", universal_embed_mode=None, prototypes=None, \
                    temperature=None, universal_downsampler=None):
  dat = apply_model(embedder, sent)
  #print (dat.dtype)
  norm_dat = torch.nn.functional.normalize(dat, dim=1)
  #some embedders are very sensitive to running the data through softmax, so here
  #we maginfify the parts of the vector that have a high value, but not by as much as a normal softmax
  #in this way, we can control the softmax with a temperature but still have norm
  if temperature is not None:
    dat = (_apply_temperature(dat,1)+_apply_temperature(dat,temperature) + norm_dat)/3
  else:
    dat = (2*_apply_temperature(dat,1) + norm_dat)/3
  if downsampler is not None:
    p = next(downsampler.parameters())
    dat = downsampler(dat.to(device=p.device, dtype=p.dtype))
  if universal_embed_mode:
      dat = cosine_similarity(dat, prototypes)
      dat = torch.nn.functional.normalize(dat, dim=1)
      p = next(downsampler.parameters())
      dat = universal_downsampler(dat.to(device=p.device, dtype=p.dtype))

  return dat

#get the embeddings using the appropriate downsampling and temperrature
def get_embeddings(sent, downsampler, dtype=np.float16, embedder="minilm", universal_embed_mode=None, prototypes=None, temperature=None, universal_downsampler=None):
  global labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set
  init_models()
  use_model(embedder)
  with torch.no_grad():
    dat = _get_embeddings(sent, downsampler, embedder, universal_embed_mode, prototypes, temperature, \
                          universal_downsampler)
    if dtype == np.float16: 
      dat = dat.half()
    else:
      dat = dat.float()
    #dat = dat.cpu().numpy()
    return dat
  
#create embeddings for all text in dat_iter. data_iter can be an interable of just the text or a (idx, text) pair.
#saves to the mmap_file. returns downsampler, skip_idxs, dtype, mmap_len, embed_dim.
#skip_idxs are the lines/embeddings that are empty and should not be clustered search indexed.
def embed_text(dat_iter, mmap_file, start_idx=None, downsampler=None, skip_idxs=None,  dtype=np.float16, mmap_len=0, embed_dim=25, \
               embedder="minilm", chunk_size=500,  universal_embed_mode=None, prototypes=None, \
               temperature=None, universal_downsampler=None, use_tqdm=True):
    global device, labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set
    assert not universal_embed_mode or (prototypes is not None and universal_downsampler is not None)   
    init_models()
    use_model(embedder)
    if start_idx is None: start_idx = mmap_len
    if skip_idxs is None: skip_idxs = []
    if downsampler is None:
      embed_dim = get_model_embed_dim(embedder)
    else:
      if dtype == np.float16:
          downsampler = downsampler.half()
      downsampler = downsampler.to(device)
      assert embed_dim == downsampler.weights.shape[1]
    batch = []
    idxs = []
    if use_tqdm:
      dat_iter2 = tqdm.tqdm(dat_iter)
    else:
      dat_iter2 = dat_iter
    with torch.no_grad():
      idx = start_idx-1
      for l in dat_iter2:
          if type(l) is tuple:
            idx, l = l
          else:
            idx += 1
          mmap_len = max(mmap_len, idx+1)
          l = l.strip()
          if l: 
            batch.append(l)
            idxs.append(idx)
          if not l or len(batch) >= chunk_size:  
            if batch:
              dat = _get_embeddings(batch, downsampler, embedder, universal_embed_mode, prototypes, \
                                    temperature, universal_downsampler).cpu().numpy()
              np_memmap(mmap_file, shape=[mmap_len, embed_dim], dat=dat, idxs=idxs, dtype=dtype)  
              batch = []
              idxs = []
            if not l:
              skip_idxs.append(idx) 
      if batch:
        dat = _get_embeddings(batch, downsampler, embedder, universal_embed_mode, prototypes, \
                                    temperature, universal_downsampler).cpu().numpy()
        np_memmap(mmap_file, shape=[mmap_len, embed_dim], dat=dat, idxs=idxs, dtype=dtype)  
        batch = []
        idxs = []
    return mmap_len, skip_idxs
    
#cluster pruning based approximate nearest neightbor search. See https://nlp.stanford.edu/IR-book/html/htmledition/cluster-pruning-1.html
#assumes the embeddings are stored in the mmap_file and the clustered index has been created.
def embeddings_search(target, mmap_file, top_parents,  top_parent_idxs, parent2idx, parents, clusters, mmap_len=0,  embed_dim=25, dtype=np.float16, chunk_size=10000, k=5):
  curr_parents = top_parents
  embeddings = parents[top_parent_idxs]
  max_level = top_parents[0][0]
  #print (max_level, top_parents)

  #print (parent_levels, embeddings)
  for _ in range(max_level+1):
    results = cosine_similarity(target, embeddings)
    results = results.topk(k)
    #print (results)
    children = list(itertools.chain(*[clusters[curr_parents[idx]] for idx in results.indices]))
    if type(children[0]) is int: #we are at the leaf nodes
      idxs = []
      n_chunks = 0
      for child_id in children:
         idxs.append(child_id)
         n_chunks += 1
         if n_chunks > chunk_size:
            embeddings = torch.from_numpy(np_memmap(mmap_file, shape=[mmap_len, embed_dim], dtype=dtype)[idxs]).to(device)
            results = cosine_similarity(target, embeddings)
            results = results.sort(descending=True)
            for idx, score in zip(results.indices.tolist(), results.values.tolist()):
               idx = idxs[idx]
               yield (idx, score)
            idxs = []
            n_chunk = 0
      if idxs:
         embeddings = torch.from_numpy(np_memmap(mmap_file, shape=[mmap_len, embed_dim], dtype=dtype)[idxs]).to(device)
         results = cosine_similarity(target, embeddings)
         results = results.sort(descending=True)
         for idx, score in zip(results.indices.tolist(), results.values.tolist()):
            idx = idxs[idx]
            yield (idx, score)
      break
    else:
      curr_parents = children
      embeddings = parents[[parent2idx[parent] for parent in curr_parents]]
            

#internal function used to cluster one batch of embeddings
#spans are either int tuples of (level, embedding_idx) or just the embedding_idx. 
def _cluster_one_batch(true_k,  spans, embedding_idxs, clusters, span2cluster_label, level, cluster_embeddings, min_overlap_merge_cluster, grouping_fn=None, grouping_fn_callback_data=None):
    with torch.no_grad():
       embeddings = torch.from_numpy(cluster_embeddings[embedding_idxs])
       if device == 'cuda':
       
         km = KMeans(n_clusters=true_k, mode='cosine')
         km_labels = km.fit_predict(embeddings.to(device=device, dtype=torch.float32))
         km_labels = [l.item() for l in km_labels.cpu()]
       else:
         km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                                           init_size=max(true_k*3,1000), batch_size=1024).fit(embeddings.to_numpy())
         km_labels = km.labels_  
      
    #put into a temporary cluster
    tmp_cluster = {}  
    for span, label in zip(spans, km_labels):
      tmp_cluster[label] = tmp_cluster.get(label, [])+[span]
  
    #print (len(tmp_cluster), tmp_cluster)
    #create unique (level, id) labels and merge if necessary
    for a_cluster in tmp_cluster.values():
        need_labels = [span for span in a_cluster if span not in span2cluster_label]
        if grouping_fn is not None:
          labeled_groups_hash = grouping_fun(grouping_fn_callback_data, cluster_embeddings, [span for span in a_cluster if span in span2cluster_label])
          unlabeled_groups_hash = grouping_fun(grouping_fn_callback_data, cluster_embeddings, need_labels)
        else:
          labeled_groups_hash = {'*': [span for span in a_cluster if span in span2cluster_label]}
          unlabeled_groups_hash =  {'*': need_labels}
        if need_labels:
          for group_id, unlabeled_group in unlabeled_groups_hash.items():
            label = None
            if group_id in labeled_groups_hash:
              labeled_group = labeled_groups_hash[group_id]
              cluster_labels =[span2cluster_label[span] for span in labeled_group]
              if not cluster_labels: continue
              # merge with previous labeled clusters if there is an overlap
              most_common = Counter(cluster_labels).most_common(1)[0]
              if most_common[1] >= min_overlap_merge_cluster: 
                label = most_common[0]    
            if not label:
              # otherwise create a new cluster
              if type(unlabeled_group[0]) is int:
                label = (level, unlabeled_group[0])
              else:
                label = (level, unlabeled_group[0][1])
            for span in unlabeled_group:
              if span not in clusters.get(label, []):
                clusters[label] = clusters.get(label, []) + [span]
              span2cluster_label[span] = label



def create_hiearchical_clusters(clusters, span2cluster_label, mmap_file, mmap_len=0, embed_dim=25, dtype=np.float16, skip_idxs=None, idxs=None, max_level=4, \
                                max_cluster_size=200, min_overlap_merge_cluster=2, prefered_leaf_node_size=None, kmeans_batch_size=250000, \
                                recluster_start_iter=0.85, max_decluster_iter=0.95, use_tqdm=True, grouping_fn=None, grouping_fn_callback_data=None):
  """
  Incremental hiearchical clustering from embeddings stored in a mmap file. Can also used to create an index for searching.       
  with a max of 4 levels, and each node containing 200 items, we can have up to 1.6B items approximately
  span2cluster_label maps a child span to a parent span. spans can be of the form int|(int,int).
  leaf nodes are ints. non-leaf nodes are (int,int) tuples
  clusters maps cluster_label => list of spans  
  when we use the term 'idx', we normally refer to the index in an embedding file.

  :arg clusters:      the dict mapping parent span to list of child span
  :arg span2cluster_label: the inverse of the above.
  :arg mmap_file:     the name of the mmap file.
  :arg mmap_len:      the current length of the mmap file.
  :arg embed_dim:     the dimension of an embedding.
  :arg dtype:         the numpy dtype.
  :arg skip_idxs:     Optioal. the idx into the embeddings that will not be clustered or searched for.
  :arg idxs:          Optioal. if provided, the particular embedding idx that will be clustered in this call.
  :arg max_level:      the maximum level of the cluster hiearchy.
  :arg max_cluster_size:the maximum size of any particular cluster.
  :arg min_overlap_merge_cluster. When incremental clustering, the minimum overlap between one cluster and another before merging them.
  :arg kmeans_batch_size: the size of each batch of embeddings that are kmean batched.
  :arg use_tqdm:        whether to report the progress of the clustering.
  :arg grouping_fn:       Optional. a function that takes in a grouping_fn_callback_data, embeddings, and a list of spans, will return a hash of form {'group_X': [...], 'group_Y': [...], etc.}
  :arg grouping_fn_callback_data: Optional. arbitrary data to pass to the grouping_fn
  """
  global device
  if skip_idxs is None: 
    skip_idxs = set()
  else:
    skip_idxs = set(skip_idxs)
  if clusters is None: clusters = {}
  if span2cluster_label is None: 
    span2cluster_label = {}
    for label, a_cluster in clusters:
      for span in a_cluster:
        span2cluster_label[span] = label
  else:      
    #make sure clusters have the same data as span2cluster_label
    for span, label in span2cluster_label.items():
      if span not in clusters.get(label,[]):
        clusters[label] = clusters.get(label,[]) + [span]
  #we are not going to cluster idxs that should be skipped
  if idxs:
    idxs = [idx for idx in idxs if idx not in skip_idxs]
  #remove some idx from the clusters so we can re-compute the clusters
  remove_idxs = list(skip_idxs) + ([] if idxs is None else idxs)
  if remove_idxs is not None:
    need_recompute_clusters=False
    for idx in remove_idxs:
      label = span2cluster_label.get(idx)
      if label is not None:
        clusters[label].remove(idx)
        a_cluster = clusters[label]
        del span2cluster_label[idx]
        need_recompute_clusters=True
        # now re-create the label if the idx is the proto index.
        if idx == label[1]:
          new_idx = a_cluster[0]          
          for level in range(0, max_level):   
            new_label2 = (level, new_idx)       
            old_label2 = (level, idx)
            if old_label2 in span2cluster_label:
              #rename the label for the children
              clusters[new_label2] = clusters[old_label2]
              del clusters[old_label2]
              for span in clusters[new_label2]:
                span2cluster_label[span] = new_label2
              #make the parent refer to the new label
              parent_label = span2cluster_label[old_label2]
              clusters[parent_label].remove(old_label2)
              clusters[parent_label].append(new_label2)
              span2cluster_label[new_label2] = parent_label

    #belt and suspenders, let's just recreate the clusters                       
    if need_recompute_clusters:
      clusters.clear()
      for span, label in span2cluster_label.items():
        clusters[label] = clusters.get(label, []) + [span]
  #print (mmap_len, clusters, span2cluster_label)

  if prefered_leaf_node_size is None: prefered_leaf_node_size = max_cluster_size
  cluster_embeddings = np_memmap(mmap_file, shape=[mmap_len, embed_dim], dtype=dtype)
  # at level 0, the spans are the indexes themselves, so no need to map using all_spans
  all_spans = None
  for level in range(max_level):
    assert level == 0 or (all_spans is not None and idxs is not None)
    #print ("got here")
    if idxs is None: 
      len_spans = mmap_len
    else:
      len_spans = len(idxs)
    # we are going to do a minimum of 6 times in case there are not already clustered items
    # from previous iterations. 
    num_times = max(6,math.ceil(len_spans/int(.7*kmeans_batch_size)))
    recluster_at = max(0,num_times*recluster_start_iter)
    rng = 0
    if use_tqdm:
      num_times2 = tqdm.tqdm(range(num_times))
    else:
      num_times2 = range(num_times)
    for times in num_times2:
        max_rng = min(len_spans, rng+int(.7*kmeans_batch_size))
        #create the next batch to cluster
        if idxs is None:
          spans = list(range(rng, max_rng))
          not_already_clustered = [idx for idx in range(rng) if (all_spans is not None and all_spans[idx] not in span2cluster_label) or \
                        (all_spans is None and idx not in span2cluster_label)]
        else:
          spans = idxs[rng: max_rng] 
          not_already_clustered = [idx for idx in range(rng) if idxs[:rng] if (all_spans is not None and all_spans[idx] not in span2cluster_label) or \
                        (all_spans is None and idx not in span2cluster_label)]
        num_itmes_left = kmeans_batch_size - len(spans)
        if len(not_already_clustered) > int(.5*num_itmes_left):
          spans.extend(random.sample(not_already_clustered, int(.5*num_itmes_left)))
        else:
          spans.extend(not_already_clustered)
        if len(spans) == 0: continue
        if level == 0:
          already_clustered = [idx for idx in range(mmap_len) if idx in span2cluster_label]
        else:
          already_clustered = [idx for idx, span in enumerate(all_spans) if span in span2cluster_label]
        if len(already_clustered)  > int(.5*num_itmes_left):
          spans.extend(random.sample(already_clustered, int(.5*num_itmes_left)))
        else:
          spans.extend(already_clustered)
        # get the embedding indexs for the cluster
        if level == 0:
          spans = [span for span in spans if span not in skip_idxs]
          embedding_idxs = spans
        else:
          spans = [all_spans[idx] for idx in spans]
          spans = [span for span in spans  if span[1] not in skip_idxs] 
          embedding_idxs = [span[1] for span in spans]
        #print (spans)
        #do kmeans clustering in batches with the embedding indexes
        if level == 0:
          true_k = int(len(embedding_idxs)/prefered_leaf_node_size)
        else:
          true_k = int(len(embedding_idxs)/max_cluster_size)
        _cluster_one_batch(true_k,  spans, embedding_idxs, clusters, span2cluster_label, level, cluster_embeddings, min_overlap_merge_cluster, grouping_fn, grouping_fn_callback_data)
        # re-cluster any small clusters or break up large clusters   
        if times >= recluster_at:  
            need_recompute_clusters = False   
            for parent, spans in list(clusters.items()): 
              if  times < max(0, num_times*max_decluster_iter) and \
                ((level == 0 and len(spans) < prefered_leaf_node_size*.5) or
                 (level != 0 and len(spans) < max_cluster_size*.5)):
                need_recompute_clusters = True
                for span in spans:
                  del span2cluster_label[span]  
              elif len(spans) > max_cluster_size:
                need_recompute_clusters = True
                for token in spans:
                  del span2cluster_label[token]
                embedding_idxs = [span if type(span) is int else span[1] for span in spans]
                if level == 0:
                  true_k = int(len(embedding_idxs)/prefered_leaf_node_size)
                else:
                  true_k = int(len(embedding_idxs)/max_cluster_size)
                _cluster_one_batch(true_k,  spans, embedding_idxs, clusters, span2cluster_label, level, cluster_embeddings,  min_overlap_merge_cluster, grouping_fn, grouping_fn_callback_data)
        
            if need_recompute_clusters:
              clusters.clear()
              for span, label in span2cluster_label.items():
                clusters[label] = clusters.get(label, []) + [span]
        rng = max_rng

    # prepare data for next level clustering
    all_spans = [label for label in clusters.keys() if label[0] == level]
    if len(all_spans) < max_cluster_size: break
    idxs = [idx for idx, label in enumerate(all_spans) if label not in span2cluster_label]
  
  return clusters, span2cluster_label

#level 0 means the bottom most parents. level -1 means the children.         
def plot_clusters(clusters, span2cluster_label, mmap_file, mmap_len=0, embed_dim=25, dtype=np.float16, level=0, additional_idxs=None, plot_width=11.7, plot_heigth=8.27, \
        tsne_perplexity=15, tsne_learning_rate=10):
  cluster_embeddings = np_memmap(mmap_file, shape=[mmap_len, embed_dim], dtype=dtype)
  num_colors = 0
  all_children=[]
  colors = []
  for parent, children in clusters.items():
    if parent[1] == level+1:
      all_children.extend(children)
      colors.extend([num_colors]*len(children))
      num_colors += 1
  X = cluster_embeddings[all_children].astype(np.float32)
  sns.set(rc={'figure.figsize':(plot_width,plot_heigth)})
  palette = sns.color_palette("bright", num_colors)
  with torch.no_grad():
    X_emb =  TSNE(n_components=2, perplexity=tsne_perplexity, learning_rate=tsne_learning_rate).fit_transform(X)
    sns.scatterplot(X_emb[:,0], X_emb[:,1],  legend='full',  hue=colors, palette=palette)
    plt.show()

def _is_contiguous(arr):
        start = None
        prev = None
        contiguous=True
        for i in arr:
          if start is None:
            start = i
          if prev is None or i == prev+1:
            prev = i
            continue
          contiguous = False
          break
        return contiguous, start, i+1


class FileByLineIdx:
    """ A class for accessing a file by line numbers. Requires  fobj that provides a seek, and tell method.
    Optionally, the dat representing the line seek points can also be passed as dat. """
    def __init__(self, fobj, dat=None):
      self.dat = dat
      self.fobj = fobj
      pos = fobj.tell()
      fobj.seek(0, os.SEEK_END)
      self.file_size = file_size = fobj.tell() 
      if self.dat is not None:
        fobj.seek(pos,0)
      else:
        def reader(fobj, rng, max_rng, ret):
          fobj.seek(rng,0)
          pos = fobj.tell()
          while rng < max_rng:
            fobj.readline()
            pos = fobj.tell() 
            if pos < max_rng:
              ret.append(pos)
            else:
              break
          rng = pos
        workers=[]
        line_nums = []
        for rng in range(0, file_size, 10000000):                    
          max_rng = min(rng + 10000000, file_size)
          line_nums.append([])
          worker = threading.Thread(target=reader, args=(copy.copy(fobj), rng, max_rng, line_nums[-1]))
          workers.append(worker)
          worker.start()
        for worker in workers:
          worker.join()
        self.dat = [0]+list(itertools.chain(*line_nums))
        fobj.seek(pos,0)
  

    def __iter__(self):
        fobj = self.fobj
        len_self = len(self)
        for start in range(0, len_self, 1000):
          end = min(len_self, start+1000)
          start = self.dat[start]
          if end == len_self:
            end = self.file_size
          else:
            end= self.dat[end]-1
          ret = []
          pos = self.tell()
          fobj.seek(start, 0)
          ret= fobj.read(end-start).split(b'\n')
          fobj.seek(pos, 0)
          for line in ret:
            yield line

    def __len__(self):
        return len(self.dat)

    def __getitem__(self, keys):
        fobj = self.fobj
        start, end = None, None
        if isinstance(keys, int):
          contiguous = False
        else:
          contiguous, start, end = _is_contiguous(keys)
        if isinstance(keys, slice):
          contiguous = True
          start = 0 if keys.start is None else keys.start
          end = len(self) if keys.stop is None else keys.stop

        if contiguous:
          start = self.dat[start]
          if end >= len(self.dat):
            end = self.file_size
          else:
            end= self.dat[end+1]-1
          pos = fobj.tell()
          fobj.seek(start, 0)
          ret= fobj.read(end-start).split(b'\n')
          fobj.seek(pos, 0)
          return ret
        elif isinstance(keys, int):
          start = self.dat[keys]
          pos = fobj.tell()
          fobj.seek(start, 0)
          ret= fobj.readline()
          fobj.seek(pos, 0)
          return ret
        else:
          return [self[idx] for idx in keys]


def _unpickle_gzip_by_line(state):
    """Create a new ``IndexedGzipFile`` from a pickled state.
    :arg state: State of a pickled object, as returned by the
                ``IndexedGzipFile.__reduce__`` method.
    :returns:   A new ``IndexedGzipFile`` object.
    """
    tell  = state.pop('tell')
    index = state.pop('index')
    gzobj = GzipByLineIdx(**state)

    if index is not None:
        gzobj.import_index(fileobj=io.BytesIO(index))

    gzobj.seek(tell)

    return gzobj

  
class GzipByLineIdx(igzip.IndexedGzipFile):
  #TODO: refactor to use FileByLineIdx as a member obj.
  
    """This class inheriets from `` ingdex_gzip.IndexedGzipFile``. This class allows in addition to the functionality 
    of IndexedGzipFile, access to a specific line based on the seek point of the line, using the __getitem__ method.
    Additionally, a (conginguous) list or slice can be used, which will be more efficient then doing line by line access. 
    
    The base IndexedGzipFile class allows for fast random access of a gzip
    file by using the ``zran`` library to build and maintain an index of seek
    points into the file.
    ``IndexedGzipFile`` is an ``io.BufferedReader`` which wraps an
    :class:`_IndexedGzipFile` instance. By accessing the ``_IndexedGzipFile``
    instance through an ``io.BufferedReader``, read performance is improved
    through buffering, and access to the I/O methods is made thread-safe.
    A :meth:`pread` method is also implemented, as it is not implemented by
    the ``io.BufferedReader``.
    """


    def __init__(self, *args, **kwargs):
        """Create an ``LineIndexGzipFileExt``. The file may be specified either
        with an open file handle (``fileobj``), or with a ``filename``. If the
        former, the file must have been opened in ``'rb'`` mode.
        .. note:: The ``auto_build`` behaviour only takes place on calls to
                  :meth:`seek`.
        :arg filename:         File name or open file handle.
        :arg fileobj:          Open file handle.
        :arg mode:             Opening mode. Must be either ``'r'`` or ``'rb``.
        :arg auto_build:       If ``True`` (the default), the index is
                               automatically built on calls to :meth:`seek`.
        :arg skip_crc_check:   Defaults to ``False``. If ``True``, CRC/size
                               validation of the uncompressed data is not
                               performed.
        :arg spacing:          Number of bytes between index seek points.
        :arg window_size:      Number of bytes of uncompressed data stored with
                               each seek point.
        :arg readbuf_size:     Size of buffer in bytes for storing compressed
                               data read in from the file.
        :arg readall_buf_size: Size of buffer in bytes used by :meth:`read`
                               when reading until EOF.
        :arg drop_handles:     Has no effect if an open ``fid`` is specified,
                               rather than a ``filename``.  If ``True`` (the
                               default), a handle to the file is opened and
                               closed on every access. Otherwise the file is
                               opened at ``__cinit__``, and kept open until
                               this ``_IndexedGzipFile`` is destroyed.
        :arg index_file:       Pre-generated index for this ``gz`` file -
                               if provided, passed through to
                               :meth:`import_index`.
        :arg buffer_size:      Optional, must be passed as a keyword argument.
                               Passed through to
                               ``io.BufferedReader.__init__``. If not provided,
                               a default value of 1048576 is used.
        :arg line2seekpoint:      Optional, must be passed as a keyword argument.
                               If not passed, this will automatically be created.                               
        """
        filename = kwargs.get("filename") 
        if args and not filename:
          filename = args[0]
        need_export_index = False
        if filename:
          if not os.path.exists(filename+"_idx"):
            need_export_index = True
            os.makedirs(filename+"_idx")
          if not os.path.exists(filename+"_idx/igzip.pickle"):
            need_export_index = True
          else:
            kwargs['index_file'] = kwargs.pop('index_file', filename+"_idx/igzip.pickle")
        
        if 'file_size' in kwargs:
          file_size = self.file_size = kwargs.pop('file_size', None)
          need_export_index = False
        self.line2seekpoint  = kwargs.pop('line2seekpoint', None)
        if need_export_index and 'auto_build' not in kwargs: kwargs['auto_build'] = True
        super(GzipByLineIdx, self).__init__(*args, **kwargs)
        if not hasattr(self, 'file_size'):
          self.build_full_index()
          pos = self.tell()
          self.seek(0, os.SEEK_END)
          self.file_size = file_size = self.tell() 
          if self.line2seekpoint is None:
            def reader(fobj, rng, max_rng, ret):
              fobj.seek(rng,0)
              pos = fobj.tell()
              while rng < max_rng:
                fobj.readline()
                pos = fobj.tell() 
                if pos < max_rng:
                  ret.append(pos)
                else:
                  break
                rng = pos

            workers=[]
            line_nums = []
            for rng in range(0, file_size, 10000000):                    
              max_rng = min(rng + 10000000, file_size)
              line_nums.append([])
              worker = threading.Thread(target=reader, args=(copy.copy(self), rng, max_rng, line_nums[-1]))
              workers.append(worker)
              worker.start()
            for worker in workers:
              worker.join()
            self.line2seekpoint = [0]+list(itertools.chain(*line_nums))
        if filename and need_export_index: 
          self.export_index(filename+"_idx/igzip.pickle")

    def __reduce__(self):
        """Used to pickle an ``GzipByLineIdx``.
        Returns a tuple containing:
          - a reference to the ``unpickle`` function
          - a tuple containing a "state" object, which can be passed
            to ``unpickle``.
        """
        
        fobj = self._IndexedGzipFile__igz_fobj

        if (not fobj.drop_handles) or (not fobj.own_file):
            raise pickle.PicklingError(
                'Cannot pickle GzipByLineIdx that has been created '
                'with an open file object, or that has been created '
                'with drop_handles=False')

        # export and serialise the index if
        # any index points have been created.
        # The index data is serialised as a
        # bytes object.
        if fobj.npoints == 0:
            index = None

        else:
            index = io.BytesIO()
            self.export_index(fileobj=index)
            index = index.getvalue()

        state = {
            'filename'         : fobj.filename,
            'auto_build'       : fobj.auto_build,
            'spacing'          : fobj.spacing,
            'window_size'      : fobj.window_size,
            'readbuf_size'     : fobj.readbuf_size,
            'readall_buf_size' : fobj.readall_buf_size,
            'buffer_size'      : self._IndexedGzipFile__buffer_size,
            'line2seekpoint'   : self.line2seekpoint,
            'file_size'   : self.file_size,
            'tell'             : self.tell(),
            'index'            : index}

        return (_unpickle_gzip_by_line, (state, ))

    
    def __iter__(self):
        len_self = len(self)
        for start in range(0, len_self, 1000):
          end = min(len_self, start+1000)
          start = self.line2seekpoint[start]
          if end == len_self:
            end = self.file_size
          else:
            end= self.line2seekpoint[end]-1
          ret = []
          with self._IndexedGzipFile__file_lock:
            pos = self.tell()
            self.seek(start, 0)
            ret= self.read(end-start).split(b'\n')
            self.seek(pos, 0)
          for line in ret:
            yield line

    def __len__(self):
        return len(self.line2seekpoint)

    def __getitem__(self, keys):
        start, end = None, None
        if isinstance(keys, int):
          contiguous = False
        else:
          contiguous, start, end = _is_contiguous(keys)
        if isinstance(keys, slice):
          contiguous = True
          start = 0 if keys.start is None else keys.start
          end = len(self) if keys.stop is None else keys.stop

        if contiguous:
          start = self.line2seekpoint[start]
          if end >= len(self.line2seekpoint):
            end = self.file_size
          else:
            end= self.line2seekpoint[end+1]-1
          with self._IndexedGzipFile__file_lock:
            pos = self.tell()
            self.seek(start, 0)
            ret= self.read(end-start).split(b'\n')
            self.seek(pos, 0)
            return ret
        elif isinstance(keys, int):
          start = self.line2seekpoint[keys]
          with self._IndexedGzipFile__file_lock:
            pos = self.tell()
            self.seek(start, 0)
            ret= self.readline()
            self.seek(pos, 0)
            return ret
        else:
          return [self[idx] for idx in keys]
    
    @staticmethod
    def open(filename):
       if os.path.exists(filename+"_idx/index.pickle"):
          return GzipByLineIdx(filename, index_file=filename+"_idx/index.pickle")
       else:
          return GzipByLineIdx(filename) 
