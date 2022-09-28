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
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast, CLIPProcessor, CLIPModel, BertModel
from nltk.corpus import stopwords as nltk_stopwords
from torch import nn
import spacy
from collections import OrderedDict
import multiprocessing
import math 
import json
if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'
  
try:
  if minilm_model is not None: 
    pass
except:
   labse_tokenizer= labse_model=  clip_processor = minilm_tokenizer= clip_model= minilm_model= spacy_nlp= stopwords_set = None

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

def get_embeddings(sent, downsampler, dtype=np.float16, embedder="minilm", do_universal_embed=False, prototypes=None, universal_downsampler=None):
  global labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set
  init_models()
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
  with torch.no_grad():
    if embedder == "clip":
      toks = clip_processor(sent, padding=True, truncation=True, return_tensors="pt").to(device)
      dat = clip_model.get_text_features(**toks)
    elif embedder == "minilm":
      toks = minilm_tokenizer(sent, padding=True, truncation=True, return_tensors="pt").to(device)
      dat = minilm_model(**toks)
      dat = mean_pooling(dat, toks.attention_mask)
    elif embedder == "labse":
      toks = labse_tokenizer(sent, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
      dat = labse_model(**toks).pooler_output   
    dat = torch.nn.functional.normalize(dat, dim=1)
    dat = downsampler(dat)
    if do_universal_embed:
      dat = cosine_similarity(dat, prototypes)
      dat = torch.nn.functional.normalize(dat, dim=1)
      dat = universal_downsampler(dat)
    if dtype == np.float16: 
      dat = dat.half()
    else:
      dat = dat.float()
    #dat = dat.cpu().numpy()
    return dat

def np_memmap(f, dat=None, idxs=None, shape=None, dtype=np.float16, ):
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


#Mean Pooling - Take attention mask into account for correct averaging
#TODO, mask out the prefix for data that isn't the first portion of a prefixed text.
def mean_pooling(model_output, attention_mask):
    with torch.no_grad():
      token_embeddings = model_output.last_hidden_state
      input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
      return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#create embeddings for all text in dat_iter. data_iter can be an interable of just the text or a (idx, text) pair.
#saves to the mmap_file. returns downsampler, skip_idxs, dtype, mmap_len, embed_dim.
#skip_idxs are the lines/embeddings that are empty and should not be clustered search indexed.
def embed_text(dat_iter, mmap_file, start_idx=None, downsampler=None, skip_idxs=None,  dtype=np.float16, mmap_len=0, embed_dim=25,  embedder="minilm", chunk_size=1000,  do_universal_embed=False, prototypes=None, universal_downsampler=None, use_tqdm=True):
    global device, labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set
    assert not do_universal_embed or (prototypes is not None and universal_downsampler is not None)   
    assert downsampler is not None
    init_models()
    if start_idx is None: start_idx = mmap_len
    if skip_idxs is None: skip_idxs = []
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
    if dtype == np.float16:
        downsampler = downsampler.half()
    downsampler = downsampler.to(device)
    batch = []
    idxs = []
    if use_tqdm:
      dat_iter2 = tqdm.tqdm(dat_iter)
    else:
      dat_iter2 = dat_iter
    idx = start_idx-1
    for l in dat_iter2:
        if type(l) is tuple:
          idx, l = l
        else:
          idx += 1
        mmap_len = max(mmap_len, idx+1)
        try:
          l = l.decode()
        except:
          pass
        l = l.replace("\\n", "\n").replace("\\t", "\t").replace("_", " ").strip()
        if l: 
          batch.append(l)
          idxs.append(idx)
        if not l or len(batch) >= chunk_size:  
          if batch:
            dat = get_embeddings(batch, downsampler, dtype=dtype, embedder=embedder, \
                                 do_universal_embed=do_universal_embed, prototypes=prototypes, universal_downsampler=universal_downsampler)
            cluster_vecs = np_memmap(mmap_file, shape=[mmap_len, embed_dim], dat=dat, idxs=idxs)  
            batch = []
            idxs = []
          if not l:
            skip_idxs.append(idx) 
    if batch:
      dat = get_embeddings(batch, downsampler, dtype=dtype, embedder=embedder, \
                                 do_universal_embed=do_universal_embed, prototypes=prototypes, universal_downsampler=universal_downsampler)
      cluster_vecs = np_memmap(mmap_file, shape=[mmap_len, embed_dim], dat=dat, idxs=idxs)  
      batch = []
      idxs = []
    return downsampler, skip_idxs, dtype, mmap_len, embed_dim 
    
#cluster pruning based approximate nearest neightbor search. See https://nlp.stanford.edu/IR-book/html/htmledition/cluster-pruning-1.html
#assumes the embeddings are stored in the mmap_file and the clustered index has been created.
def embeddings_search(vec, mmap_file, top_parents,  top_parent_idxs, parent2idx, parents, clusters, mmap_len=0, embed_dim=25, dtype=np.float16, chunk_size=10000, k=5):
  curr_parents = top_parents
  vecs = parents[top_parent_idxs] 
  max_level = top_parents[0][0]
  #print (max_level, top_parents)

  #print (parent_levels, vecs)
  for _ in range(max_level+1):
    results = cosine_similarity(vec, vecs)
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
            vecs = torch.from_numpy(np_memmap(mmap_file, shape=[mmap_len, embed_dim], dtype=dtype)[idxs]).to(device)
            results = cosine_similarity(vec, vecs)
            results = results.sort(descending=True)
            for idx, score in zip(results.indices.tolist(), results.values.tolist()):
               idx = idxs[idx]
               yield (idx, score)
            idxs = []
            n_chunk = 0
      if idxs:
         vecs = torch.from_numpy(np_memmap(mmap_file, shape=[mmap_len, embed_dim], dtype=dtype)[idxs]).to(device)
         results = cosine_similarity(vec, vecs)
         results = results.sort(descending=True)
         for idx, score in zip(results.indices.tolist(), results.values.tolist()):
            idx = idxs[idx]
            yield (idx, score)
      break
    else:
      curr_parents = children
      vecs = parents[[parent2idx[parent] for parent in curr_parents]] 


#internal function used to cluster one batch of embeddings
def _cluster_one_batch(true_k,  spans, vector_idxs, clusters, span2cluster_label, level, cluster_vecs, min_overlap_merge_cluster):
    if device == 'cuda':
      km = KMeans(n_clusters=true_k, mode='cosine')
      km_labels = km.fit_predict(torch.from_numpy(cluster_vecs[vector_idxs]).to(device=device, dtype=torch.float32))
      km_labels = [l.item() for l in km_labels.cpu()]
    else:
      km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                                        init_size=max(true_k*3,1000), batch_size=1024)
      km_labels = km.labels_  
      
    #put into a temporary cluster
    tmp_cluster = {}  
    for span, label in zip(spans, km_labels):
      tmp_cluster[label] = tmp_cluster.get(label, [])+[span]
  
    #print (len(tmp_cluster), tmp_cluster)
    #create unique (level, id) labels and merge if necessary
    for a_cluster in tmp_cluster.values():
        label = None
        need_labels = [span for span in a_cluster if span not in span2cluster_label]
        if need_labels:
          cluster_labels = [span2cluster_label[span] for span in a_cluster if span in span2cluster_label]
          if cluster_labels:
            # merge with previous clusters if there is an overlap
            most_common = Counter(cluster_labels).most_common(1)[0]
            if most_common[1] >= min_overlap_merge_cluster: 
              label = most_common[0]
          if not label:
            # otherwise create a new cluster
            if type(need_labels[0]) is int:
              label = (level, need_labels[0])
            else:
              label = (level, need_labels[0][1])
          for span in need_labels:
            if span not in clusters.get(label, []):
              clusters[label] = clusters.get(label, []) + [span]
            span2cluster_label[span] = label


# Incremental hiearchical clustering from vectors in the mmap file. Can also used to create an index for searching.       
# with a max of 4 levels, and each node containing 200 items, we can have up to 1.6B items approximately
# span2cluster_label maps a span to a parent span. spans can be of the form int|(int,int).
# leaf nodes are ints. non-leaf nodes are (int,int) tuples
# clusters maps cluster_label => list of spans  
def create_hiearchical_clusters(clusters, span2cluster_label, mmap_file, mmap_len=0, embed_dim=25, dtype=np.float16, skip_idxs=None, cluster_idxs=None, max_level=4, max_cluster_size=200, \
                               min_overlap_merge_cluster=2, prefered_leaf_node_size=None, kmeans_batch_size=250000, use_tqdm=True):
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
  if cluster_idxs:
    cluster_idxs = [idx for idx in cluster_idxs if idx not in skip_idxs]
  #remove some idx from the clusters so we can re-compute the clusters
  remove_idxs = list(skip_idxs) + ([] if cluster_idxs is None else cluster_idxs)
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

  if prefered_leaf_node_size is None: prefered_leaf_node_size = max_cluster_size
  cluster_vecs = np_memmap(mmap_file, shape=[mmap_len, embed_dim], dtype=dtype)
  # at level 0, the spans are the indexes themselves, so no need to map using all_spans
  all_spans = None
  for level in range(max_level):
    assert level == 0 or (all_spans is not None and cluster_idxs is not None)
    if cluster_idxs is None: 
      len_spans = mmap_len
    else:
      len_spans = len(cluster_idxs)
    # we are going to do a minimum of 6 times in case there are not already clustered items
    # from previous iterations. 
    num_times = max(6,math.ceil(len_spans/int(.7*kmeans_batch_size)))
    recluster_at = max(0,num_times*0.65)
    rng = 0
    if use_tqdm:
      num_times2 = tqdm.tqdm(range(num_times))
    else:
      num_times2 = range(num_times)
    for times in num_times2:
        max_rng = min(len_spans, rng+int(.7*kmeans_batch_size))
        #create the next batch to cluster
        if cluster_idxs is None:
          spans = list(range(rng, max_rng))
          not_already_clustered = [idx for idx in range(rng) if (all_spans is not None and all_spans[idx] not in span2cluster_label) or \
                        (all_spans is None and idx not in span2cluster_label)]
        else:
          spans = cluster_idxs[rng: max_rng] 
          not_already_clustered = [idx for idx in range(rng) if cluster_idxs[:rng] if (all_spans is not None and all_spans[idx] not in span2cluster_label) or \
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
        # get the vector indexs for the cluster
        if level == 0:
          spans = [span for span in spans if span not in skip_idxs]
          vector_idxs = spans
        else:
          spans = [all_spans[idx] for idx in spans]
          spans = [span for span in spans  if span[1] not in skip_idxs] 
          vector_idxs = [span[1] for span in spans]
        #do kmeans clustering in batches with the vector indexes
        if level == 0:
          true_k = int(len(vector_idxs)/prefered_leaf_node_size)
        else:
          true_k = int(len(vector_idxs)/max_cluster_size)
        _cluster_one_batch(true_k,  spans, vector_idxs, clusters, span2cluster_label, level, cluster_vecs, min_overlap_merge_cluster)
        # re-cluster any small clusters or break up large clusters   
        if times >= recluster_at:  
            need_recompute_clusters = False   
            for parent, spans in list(clusters.items()): 
              if  times < num_times-2 and \
                ((level == 0 and len(spans) < prefered_leaf_node_size*.5) or
                 (level != 0 and len(spans) < max_cluster_size*.5)):
                need_recompute_clusters = True
                for span in spans:
                  del span2cluster_label[span]  
              elif len(spans) > max_cluster_size:
                need_recompute_clusters = True
                for token in spans:
                  del span2cluster_label[token]
                vector_idxs = [span if type(span) is int else span[1] for span in spans]
                if level == 0:
                  true_k = int(len(vector_idxs)/prefered_leaf_node_size)
                else:
                  true_k = int(len(vector_idxs)/max_cluster_size)
                _cluster_one_batch(true_k,  spans, vector_idxs, clusters, span2cluster_label, level, cluster_vecs,  min_overlap_merge_cluster)
        
            if need_recompute_clusters:
              clusters.clear()
              for span, label in span2cluster_label.items():
                clusters[label] = clusters.get(label, []) + [span]
        rng = max_rng

    # prepare data for next level clustering
    all_spans = [label for label in clusters.keys() if label[0] == level]
    if len(all_spans) < max_cluster_size: break
    cluster_idxs = [idx for idx, label in enumerate(all_spans) if label not in span2cluster_label]
  
  return clusters, span2cluster_label
        
    
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

