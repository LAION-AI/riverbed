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
from .utils import *

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'
  
try:
  if minilm_model is not None: 
    pass
except:
   labse_tokenizer= labse_model=  clip_processor = minilm_tokenizer= clip_model= minilm_model= spacy_nlp= stopwords_set = None


def get_content_from_line(l, search_field="text"):
    l =l.decode().replace("\\n", "\n").replace("\\t", "\t").strip()
    if not l: return ''
    if l[0] == "{" and l[-1] == "}":
      content = l.split(search_field+'": "')[1]
      content = content.split('", "')[0].replace("_", " ")
    else:
      content = l.replace("_", " ")
    return content
    
class PreprocessorMixin:
  def __init__(self, start_idx = 0, embed_search_field="text", bm25_field="text"):
    raise NotImplementedError

  def reset_embed_search_idx(self, start_idx):
    raise NotImplementedError
    
  def reset_bm25_idx(self, start_idx):
    raise NotImplementedError
  
  #perform both bm25 and embed_search processing/transmation of the text, and yield data.
  def process(self, lines_iterator, *args, **kwargs):
    raise NotImplementedError
    
  def process_embed_search_field(self, lines_iterator, *args, **kwargs):
    raise NotImplementedError

  def process_bm25_field(self, lines_iterator, *args, **kwargs):
    raise NotImplementedError  
  
  def process_one_line_for_embed_search_field(self, line, *args, **kwargs):
    raise NotImplementedError

  def process_one_line_for_bm25_field(self, line, *args, **kwargs):
    raise NotImplementedError  

#TODO. modify the embed_text function to store away a tuple idx -> embedding idx. The tuple idx corresponds to a unique id generated 
#by the preprocessor, e.g., (lineno, offset) or (file name, lineno, offset), etc. 
class BasicLinePeprocessor(PreprocessorMixin):
  def __init__(self, start_idx = 0, embed_search_field="text", bm25_field="text", searcher=None):
    self.reset_embed_search_idx(start_idx)
    self.reset_bm25_idx(start_idx)
    self.embed_search_field = embed_search_field
    self.bm25_field = bm25_field
    self.searcher = searcher
    
  def reset_embed_search_idx(self, start_idx):
     self.embed_search_idx = start_idx
      
  def reset_bm25_idx(self, start_idx):
     self.bm25_idx = start_idx
  
  # gets in a lines iterator and outputs subsequent dict generator
  def process(self, lines_iterator, *args, **kwargs):
    for line in lines_iterator:
      line1, line2 =  get_content_from_line(line, self.embed_search_field, self.bm25_field)
      if not l: 
        yield None
        continue
      try:
        line = line.decode()
      except:
        pass
      offset = 0
      for text, key_words in zip(line1.split("\\n"), line2.split("\\n")):
        offset = line.index(text, offset)
        yield {'idx': self.embed_search_idx, 'offset': offset, 'text': text, 'key_words': key_words}
      self.embed_search_idx += 1
  
  # gets in a lines iterator and outputs subsequent dict generator
  def process_embed_search_field(self, lines_iterator, *args, **kwargs):
    for line in lines_iterator:
      l =  get_content_from_line(line, self.embed_search_field)
      if not l: 
        yield None
        continue
      try:
        line = line.decode()
      except:
        pass
      offset = 0
      for l2 in l.split("\\n"):
        offset = line.index(l2, offset)
        yield {'idx': self.embed_search_idx, 'offset': offset, 'text': l2}
      self.embed_search_idx += 1
      
  # gets in a lines iterator and outputs subsequent dict generator
  def process_bm25_field(self, lines_iterator, *args, **kwargs):
    for line in lines_iterator:
      l =  get_content_from_line(line, self.bm25_field)
      if not l: 
        yield None
        continue
      try:
        line = line.decode()
      except:
        pass
      offset = 0
      for l2 in l.split("\\n"):
        offset = line.index(l2, offset)
        yield {'idx': self.bm25_idx, 'offset': offset, 'text': l2}
      self.bm25_idx += 1

#TODO. Change this to inherit from a transformers.PretrainedModel.
class Searcher(nn.Module):
  
  def __init__(self,  filename=None, idx_dir=None, content_data_store=None, mmap_file=None, mmap_len=0, embed_dim=25, dtype=np.float16, \
               parents=None, parent_levels=None, parent_labels=None, skip_idxs=None, \
               parent2idx=None, top_parents=None, top_parent_idxs=None, clusters=None,  embedder="minilm", chunk_size=500, \
               embed_search_field="text", bm25_field=None, downsampler=None, auto_embed_text=False, \
               auto_create_embeddings_idx=False, auto_create_bm25_idx=False,  \
               span2cluster_label=None, idxs=None, max_level=4, max_cluster_size=200, \
               min_overlap_merge_cluster=2, prefered_leaf_node_size=None, kmeans_batch_size=250000, \
               universal_embed_mode = None, prototype_sentences=None,  prototypes=None, universal_downsampler =None, min_num_prorotypes=50000, \
               use_tqdm=True, preprocessor=None
              ):
    #TODO, add a embedding_preprocessor. Given a batch of sentences, and an embedding, create additional embeddings corresponding to the batch. 
    """
        Cluster indexes and performs approximate nearest neighbor search on a memmap file. 
        Also provides a wrapper for Whoosh BM25.
        :arg filename:   Optional. The name of the file that is to be indexed and searched. 
                              Can be a txt or jsonl file or a gzip of the foregoing. 
        :arg idx_dir:    Optional. If not passed then it will be "filename_idx". If no filename is passed, then it will be the current directory. 
        :arg content_data_store:      Optional. The data store object, which can be a GzipFileByLineIdx, FileByLineIdx, or 
                             anything accessible by indexing content_data_store[i] and exposing len() which returns the number of items/lines.  
                             If filename is passed byt content_data_store is not passed, it will be created. 
        :arg  mmap_file:      Optional, must be passed as a keyword argument.
                                This is the file name for the embeddings representing 
                                each line in the gzip file. Used for embeddings search.
        :arg mmap_len         Optional, must be passed as a keyword argument if mmap_file is passed.
                                This is the shape of the mmap_file.     
        :arg embed_dim        Optional, must be passed as a keyword argument if mmap_file is passed.
                                This is the shape of the mmap_file.                       
        :arg dtype            Optional, must be passed as a keyword argument.
                                This is the dtype of the mmap_file.                              
        :arg  parents:        Optional, must be passed as a keyword argument.
                                This is a numpy or pytorch embedding of all the parents of the clusters]
                                Where level 4 parents are the top level parents. 
                                This structure is used for approximage nearest neighbor search.
        :arg parent2idx:      Optional, must be passed as a keyword argument. If parents
                                are passed, this param must be also passed. 
                                It is a dict that maps the parent tuple to the index into the parents tensor
        :arg top_parents:     Optional. The list of tuples representing the top parents.
        :arg top_parents_idxs: Optional. The index into the parents embedding for the top_parents.  
        :arg clusters:         Optional. A dictionary representing parent label -> [child indexes]
        :arg auto_create_embeddings_idx. Optional. Will create a cluster index from the contents of the mmap file. 
                                Assumes the mmap_file is populated.
        :arg auto_embed_text. Optional. Will populate the mmap_file from the data from filename/content_data_store. 
        :arg auto_create_bm25_idx: Optional. Will do BM25 indexing of the contents of the file using whoosh, with stemming.
        :arg content_data_store           Optional. The access for a file by lines.
        :arg embed_search_field:      Optional. Defaults to "text". If the data is in jsonl format,
                              this is the field that is Whoosh/bm25 indexed.
        :arg bm25_field:        Optional. Can be different than the embed_search_field. If none, then will be set to the embed_search_field.
        :arg idxs:                Optional. Only these idxs should be indexed and searched.
        :arg skip_idxs:           Optional. The indexes that are empty and should not be searched or clustered.
        :arg content_data_store:           Optional. 
        :arg downsampler:          Optional. The pythorch downsampler for mapping the output of the embedder to a lower dimension.
        :arg universal_embed_mode:  Optional. Either None, "assigned", "random", or "clusters". If we should do universal embedding as described below, this will control
                                    how the prototypes are assigned. 
        :arg prototype_sentences:     Optional. A sorted list of sentences that represents the protoypes for embeddings space. If universal_embed_mode is set and prototypes
                                  are not provided,then this will be the level 0 parents sentences of the current clustering.
                                  To get universal embedding, we do cosine(target, prototypes), then normalize and then run through a universial_downsampler
        :arg protoypes:         Optional. The embeddings in the embeddeing or (downsampled embedding) space that corresponds to the prototype_sentences.
        :arg min_num_prorotypes Optional. Will control the number of prototypes.
        :arg universal_downsampler Optional. The pythorch downsampler for mapping the output described above to a lower dimension that works across embedders
                                  and concept drift in the same embedder. maps from # of prototypes -> embed_dim. 
        :arg preprocessor:      Optional. If not set, then the BasicLineProcessor will be used.
        
      NOTE: Either pass in the parents, parent_levels, parent_labels, and parent2idx data is pased or clusters is passed. 
          If none of these are passed and auto_create_embeddings_idx is set, then the data in the mmap file will be clustered and the 
          data structure will be created.

      USAGE:
      
        for r in obj.search("test"): print (r)

        for r in obj.search(numpy_or_pytorch_tensor): print (r)

        for r in obj.search("test", numpy_or_pytorch_tensor): print (r)

      """
    global device
    global  labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set
    super().__init__()
    if preprocessor is None: preprocessor = BasicLinePeprocessor(embed_search_field=embed_search_field, bm25_field=bm25_field)
    self.embedder, self.preprocessor = embedder, preprocessor
    if idx_dir is None and filename is not None:
      idx_dir = f"{filename}_idx"
    elif idx_dir is None: idx_dir = "./"
      
    self.idx_dir = idx_dir
    if not os.path.exists(self.idx_dir):
      os.makedirs(self.idx_dir)   
    if mmap_file is None:
      mmap_file = f"{self.idx_dir}/search_index_{embed_search_field}_{embedder}_{embed_dim}.mmap"
    if content_data_store is None:
      if filename is not None:
        if filename.endswith(".gz"):
          content_data_store = GzipByLineIdx.open(filename)
        else:
          content_data_store =  FileByLineIdx(fobj=open(filename, "rb"))  
    self.content_data_store = content_data_store 
    labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set = init_models()
    if downsampler is None:
      model_embed_dim = get_model_embed_dim(embedder)
      downsampler = nn.Linear(model_embed_dim, embed_dim, bias=False).eval() 
    if bm25_field is None: bm25_field = embed_search_field
    self.universal_embed_mode, self.mmap_file, self.mmap_len, self.embed_dim, self.dtype, self.clusters, self.parent2idx,  self.parents, self.top_parents, self.top_parent_idxs, self.embed_search_field, self.bm25_field, self.downsampler  = \
             universal_embed_mode, mmap_file, mmap_len, embed_dim, dtype, clusters, parent2idx, parents, top_parents, top_parent_idxs, embed_search_field, bm25_field, downsampler
    self.prototype_sentences,  self.prototypes, self.universal_downsampler = prototype_sentences,  prototypes, universal_downsampler
    if self.downsampler is not None: 
      if self.dtype == np.float16:
        self.downsampler.eval().to(device)
      else:
        self.downsampler.half().eval().to(device)
    if self.parents is not None: 
      if self.dtype == np.float16:
        self.parents = self.parents.half().to(device)
      else:
        self.parents = self.parents.to(device)
    if skip_idxs is None: skip_idxs = []
    self.skip_idxs = set(list(self.skip_idxs if hasattr(self, 'skip_idxs') and self.skip_idxs else []) + list(skip_idxs))
    if universal_embed_mode not in (None, "assigned"):
      auto_embed_text = True
    if auto_embed_text and self.content_data_store is not None:
      self.embed_text(chunk_size=chunk_size, use_tqdm=use_tqdm)
    if universal_embed_mode not in (None, "assigned") and clusters is None:
      auto_create_embeddings_idx = True
    if os.path.exists(self.mmap_file) and (idxs is not None or auto_create_embeddings_idx):
      self.recreate_embeddings_idx(clusters=self.clusters, span2cluster_label=span2cluster_label, idxs=idxs, max_level=max_level, max_cluster_size=max_cluster_size, \
                               min_overlap_merge_cluster=min_overlap_merge_cluster, prefered_leaf_node_size=prefered_leaf_node_size, kmeans_batch_size=kmeans_batch_size)
    elif self.clusters:
      self.recreate_parents_data()
    if auto_create_bm25_idx and self.content_data_store:
       self.recreate_bm25_idx(auto_create_bm25_idx=auto_create_bm25_idx, idxs=idxs, use_tqdm=use_tqdm)
    setattr(self,f'downsampler_{self.embed_search_field}_{embedder}_{self.embed_dim}', self.downsampler)
    setattr(self,f'clusters_{self.embed_search_field}_{self.embedder}_{self.embed_dim}', self.clusters)
    
    ## experimental universal embedding code
    self.universal_embed_mode = universal_embed_mode
    if universal_embed_mode:
      assert (prototypes is None and prototype_sentences is None and universal_downsampler is None) or universal_embed_mode == "assigned"
      if universal_embed_mode == "random":
        prototype_sentences = [get_content_from_line(self.content_data_store[i], embed_search_field) for i in random.sample(list(range(len(self.content_data_store)), min_num_prorotypes))]
      elif universal_embed_mode == "cluster":
        level_0_parents = [span[1] for span in self.parent2idx.keys() if span[0] == 0]
        prototype_sentences = [get_content_from_line(self.content_data_store[span[1]], embed_search_field) for span in level_0_parents]
      assert prototype_sentences
      if len(prototype_senences) > min_num_prorotypes:
         assert universal_embed_mode != "assigned"
         prototype_senences = random.sample(prototype_senences,min_num_prorotypes)
      elif len(prototype_senences) < min_num_prorotypes:
         assert universal_embed_mode != "assigned"
         prototype_sentences.extend([get_content_from_line(self.content_data_store[i], embed_search_field) for i in random.sample(list(range(len(self.content_data_store)), min_num_prorotypes-len(prorotype_senences)))])
      prototypes = self.get_embeddings(prototype_sentences)
      universal_downsampler = nn.Linear(len(prototype_sentences), embed_dim, bias=False)
      self.prototype_sentences,  self.prototypes, self.universal_downsampler = prototype_sentences,  prototypes, universal_downsampler
      if self.universal_downsampler is not None: 
        if self.dtype == np.float16:
          self.universal_downsampler.eval().to(device)
        else:
          self.universal_downsampler.half().eval().to(device)
      if self.prototypes is not None: 
        if self.dtype == np.float16:
          self.prototypes = self.prototypes.half().to(device)
        else:
          self.prototypes = self.prototypes.to(device)
      #now re-create the embeddings, and remove the old embedder based embeddings since we won't use those anymore.
      os.system(f"rm -rf {self.mmap_file}")
      self.mmap_file = f"{self.idx_dir}/search_index_{embed_search_field}_universal_{embed_dim}.mmap"
      if auto_embed_text and self.content_data_store is not None:
        self.embed_text(chunk_size=chunk_size, use_tqdm=use_tqdm)
      self.recreate_parents_data()
    parents = self.parents
    del self.parents
    self.register_buffer('parents', parents)
    prototypes = self.prototypes
    del self.prototypes
    self.register_buffer('prototypes', prototypes)
      
  
  # get the downsampled sentence embeddings. can be used to train the downsampler(s).
  def forward(self, *args, **kwargs):
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
      dat = cosine_similarity(dat, self.prototypes)
      dat = torch.nn.functional.normalize(dat, dim=1)
      dat = self.universal_downsampler(dat)
    return dat
  
  #switch to a different embedder for the same data.
  def switch_search_context(self, downsampler = None, mmap_file=None, embedder="minilm", clusters=None, \
                            span2cluster_label=None, idxs=None, max_level=4, max_cluster_size=200, chunk_size=500,  \
                            parent2idx=None, parents=None, top_parents=None, top_parent_idxs=None, skip_idxs=None, \
                            auto_embed_text=False,auto_create_embeddings_idx=False, auto_create_bm25_idx=False,  \
                            reuse_clusters=False, min_overlap_merge_cluster=2, prefered_leaf_node_size=None, kmeans_batch_size=250000, use_tqdm=True
                          ):
    global device
    global  labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set
    if hasattr(self,f'downsampler_{self.embed_search_field}_{self.embedder}_{self.embed_dim}'): getattr(self,f'downsampler_{self.embed_search_field}_{self.embedder}_{self.embed_dim}').cpu()
    if hasattr(self, 'downsampler') and self.downsampler is not None: self.downsampler.cpu()
    content_data_store = self.content_data_store
    if self.universal_embed_mode == "clustered":
      clusters = self.clusters
    elif reuse_clusters: 
      assert clusters is None
      clusters = self.clusters
    if mmap_file is None:
      if  self.universal_embed_mode:
        mmap_file = f"{self.idx_dir}/search_index_{self.embed_search_field}_universal_{self.embed_dim}.mmap"
        auto_embed_text=not os.path.exists(self.mmap_file) # the universal embeddings are created once. 
      else:
        mmap_file = f"{self.idx_dir}/search_index_{self.embed_search_field}_{embedder}_{self.embed_dim}.mmap"
    labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set = init_models()
    if downsampler is None:
      if hasattr(self,f'downsampler_{self.embed_search_field}_{embedder}_{self.embed_dim}'):
        downsampler = getattr(self,f'downsampler_{self.embed_search_field}_{embedder}_{self.embed_dim}')
      else:
        model_embed_dim = get_model_embed_dim(embedder)  
        downsampler = nn.Linear(model_embed_dim, self.embed_dim, bias=False).eval() 
    if clusters is None:
      if hasattr(self,f'clusters_{self.embed_search_field}_{embedder}_{self.embed_dim}'):
        clusters = getattr(self,f'clusters_{self.embed_search_field}_{embedder}_{self.embed_dim}')
    self.embedder, self.mmap_file, self.clusters, self.parent2idx,  self.parents, self.top_parents, self.top_parent_idxs,  self.downsampler  = \
             embedder, mmap_file, clusters, parent2idx, parents, top_parents, top_parent_idxs, downsampler
    if skip_idxs is None: skip_idxs = []
    self.skip_idxs = set(list(self.skip_idxs if hasattr(self, 'skip_idxs') and self.skip_idxs else []) + list(skip_idxs))
    if auto_embed_text and self.content_data_store is not None:
      self.embed_text(chunk_size=chunk_size, use_tqdm=use_tqdm)
    if os.path.exists(self.mmap_file) and (idxs is not None or auto_create_embeddings_idx):
      self.recreate_embeddings_idx(clusters=self.clusters, span2cluster_label=span2cluster_label, idxs=idxs, max_level=max_level, max_cluster_size=max_cluster_size, \
                               min_overlap_merge_cluster=min_overlap_merge_cluster, prefered_leaf_node_size=prefered_leaf_node_size, kmeans_batch_size=kmeans_batch_size)
    elif self.clusters: 
      self.recreate_parents_data()
    if auto_create_bm25_idx and self.content_data_store:
       self.recreate_bm25_idx(auto_create_bm25_idx=auto_create_bm25_idx, idxs=idxs, use_tqdm=use_tqdm)
        
    #experimental universal embedding code
    if self.universal_embed_mode is not None and self.prototype_sentences:
      self.prototypes = self.get_embeddings(self.prototype_sentences)
    if self.downsampler is not None: 
      if self.dtype == np.float16:
        self.downsampler.eval().to(device)
      else:
        self.downsampler.half().eval().to(device)
    if self.parents is not None: 
      if self.dtype == np.float16:
        self.parents = self.parents.half().to(device)
      else:
        self.parents = self.parents.to(device)
    if self.prototypes is not None: 
      if self.dtype == np.float16:
        self.prototypes = self.prototypes.half().to(device)
      else:
        self.prototypes = self.prototypes.to(device)
    setattr(self,f'downsampler_{self.embed_search_field}_{embedder}_{self.embed_dim}', self.downsampler)
    setattr(self,f'clusters_{self.embed_search_field}_{self.embedder}_{self.embed_dim}', self.clusters)
    parents = self.parents
    del self.parents
    self.register_buffer('parents', parents)
    prototypes = self.prototypes
    del self.prototypes
    self.register_buffer('prototypes', prototypes)
    return self
  
  #get the sentence embedding for the sent or batch of sentences
  def get_embeddings(self, sent_or_batch):
    return get_embeddings(sent_or_batch, downsampler=self.downsampler, dtype=self.dtype, embedder=self.embedder, \
                          universal_embed_mode=self.universal_embed_mode, prototypes=self.prototypes, \
                          universal_downsampler=self.universal_downsampler)
              
  #embed all of self.content_data_store or (idx, content) for idx in idxs for the row/content from content_data_store
  def embed_text(self, start_idx=None, chunk_size=500, idxs=None, use_tqdm=True, auto_create_bm25_idx=False, **kwargs):
    assert self.content_data_store is not None
    if start_idx is None: start_idx = 0
    embed_search_field = self.embed_search_field 
    ###
    def content_data_store_reader():
      content_data_store = self.content_data_store
      if hasattr(content_data_store, 'tell'):
        pos = content_data_store.tell()
        content_data_store.seek(0, 0)
                                      
      for l in content_data_store:
        yield get_content_from_line(l, embed_search_field)
      if hasattr(content_data_store, 'tell'):
        content_data_store.seek(pos,0)
    ###  
    
    if idxs is not None:
      #TODO:
      #data_iterator = [(idx, self.preprocessor.process_one_line_for_embed_search(self.content_data_store[idx])) for idx in idxs]
      data_iterator = [(idx, get_content_from_line(self.content_data_store[idx], embed_search_field)) for idx in idxs]
    else:
      # TODO: 
      # self.preprocessor.reset_embed_search_idx(0)
      # data_iterator = self.preprocessor.process_embed_search_field(data_iterator, **kwargs)
      data_iterator = content_data_store_reader()  
    
    self.mmap_len, skip_idxs =  embed_text(data_iterator, self.mmap_file, start_idx=start_idx, downsampler=self.downsampler, \
                          mmap_len=self.mmap_len, embed_dim=self.embed_dim, embedder=self.embedder, chunk_size=chunk_size, use_tqdm=use_tqdm, \
                          universal_embed_mode=self.universal_embed_mode, prototypes=self.prototypes, universal_downsampler=self.universal_downsampler)
    setattr(self,f'downsampler_{self.embed_search_field}_{self.embedder}_{self.embed_dim}', self.downsampler)
    self.skip_idxs = set(list(self.skip_idxs)+skip_idxs)
      

  #the below is probably in-efficient
  def recreate_parents_data(self):
    global device
    
    assert self.clusters
    all_parents = list(self.clusters.keys())
    all_parents.sort(key=lambda a: a[0], reverse=True)
    max_level = all_parents[0][0]
    self.top_parents =  [a for a in all_parents if a[0] == max_level]
    self.top_parent_idxs = [idx for idx, a in enumerate(all_parents) if a[0] == max_level]
    self.parent2idx = dict([(a,idx) for idx, a in enumerate(all_parents)])
    self.parents = torch.from_numpy(np_memmap(self.mmap_file, shape=[self.mmap_len, self.embed_dim], dtype=self.dtype)[[a[1] for a in all_parents]]).to(device)
             
  def recreate_embeddings_idx(self,  clusters=None, span2cluster_label=None, idxs=None, max_level=4, max_cluster_size=200, \
                               min_overlap_merge_cluster=2, prefered_leaf_node_size=None, kmeans_batch_size=250000,):
    global device
    #print (clusters, idxs)
    
    if clusters is None or idxs is not None:
      if clusters is None and idxs is not None: clusters = self.clusters
      clusters, _ = self.cluster(clusters=clusters, span2cluster_label=span2cluster_label, idxs=idxs, max_level=max_level, max_cluster_size=max_cluster_size, \
                               min_overlap_merge_cluster=min_overlap_merge_cluster, prefered_leaf_node_size=prefered_leaf_node_size, kmeans_batch_size=kmeans_batch_size)
    #print (clusters)
    self.clusters = clusters
    self.recreate_parents_data()
              
  def get_cluster_and_span2cluster_label(self):
    span2cluster_label = {}
    for label, a_cluster in self.clusters:
      for span in a_cluster:
        span2cluster_label[span] = label
    return self.clusters, span2cluster_label

  def get_all_parents(self): 
    return self.parent2idx.keys()
  

  def cluster(self, clusters=None, span2cluster_label=None, idxs=None, max_level=4, max_cluster_size=200, \
                               min_overlap_merge_cluster=2, prefered_leaf_node_size=None, kmeans_batch_size=250000, use_tqdm=True):
    return create_hiearchical_clusters(clusters=clusters, span2cluster_label=span2cluster_label, mmap_file=self.mmap_file, mmap_len=self.mmap_len, embed_dim=self.embed_dim, dtype=self.dtype, \
                                      skip_idxs=self.skip_idxs, idxs=idxs, max_level=max_level, \
                                      max_cluster_size=max_cluster_size, min_overlap_merge_cluster=min_overlap_merge_cluster, \
                                      prefered_leaf_node_size=prefered_leaf_node_size, kmeans_batch_size=kmeans_batch_size, use_tqdm=use_tqdm)
  
  
  
  def recreate_bm25_idx(self, auto_create_bm25_idx=False, idxs=None, use_tqdm=True):
    assert self.content_data_store is not None
    content_data_store = self.content_data_store
    bm25_field = self.bm25_field 
    schema = Schema(id=ID(stored=True), content=TEXT(analyzer=StemmingAnalyzer()))
    #TODO determine how to clear out the whoosh index besides rm -rf _M* MAIN*
    idx_dir = self.idx_dir
    os.system(f"mkdir -p {idx_dir}/bm25_{bm25_field}")
    need_reindex = auto_create_bm25_idx or not os.path.exists(f"{idx_dir}/bm25_{bm25_field}/_MAIN_1.toc") #CHECK IF THIS IS RIGHT 
    if not need_reindex:
      self.whoosh_ix = whoosh_index.open_dir(f"{idx_dir}/bm25_{bm25_field}")
    else:
      self.whoosh_ix = create_in(f"{idx_dir}/bm25_{bm25_field}", schema)
      writer = self.whoosh_ix.writer(multisegment=True, limitmb=1024, procs=multiprocessing.cpu_count())      
      #writer = self.whoosh_ix.writer(multisegment=True,  procs=multiprocessing.cpu_count())      
      if hasattr(content_data_store, 'tell'):
        pos = content_data_store.tell()
        content_data_store.seek(0, 0)
      if idxs is not None:
        idx_text_pairs = [(idx, self.content_data_store[idx]) for idx in idxs]
        if use_tqdm:
          data_iterator =  tqdm.tqdm(idx_text_pairs)
        else:
          data_iterator = idx_text_pairs
      else:
        if use_tqdm:
          data_iterator = tqdm.tqdm(enumerate(content_data_store))
        else:
          data_iterator = enumerate(content_data_store)
      # TODO: 
      #self.preprocessor.reset_bm25_idx(0)
      #data_iterator = self.preprocessor.process_bm25_field(content_data_store, **kwargs)
      for idx, l in data_iterator:
          content= get_content_from_line(l, bm25_field)
          if not content: continue
          writer.add_document(id=str(idx), content=content)  
      writer.commit()
      if hasattr(content_data_store, 'tell'):
        content_data_store.seek(pos,0)

               
    
  #search using embedding based and/or bm25 search. returns generator of a dict obj containing the results in 'id', 'text',and 'score. 
  #if the underlying data is jsonl, then the result of the data will also be returned in the dict.
  #WARNING: we overwrite the 'id', 'score', and 'text' field, so we might want to use a different field name like f'{field_prefix}_score'
  #key_terms. TODO: See https://whoosh.readthedocs.io/en/latest/keywords.html
  def search(self, query=None, embedding=None, do_bm25_only=False, k=5, chunk_size=100, limit=None):
    def _get_data(idx):
      l  = self.content_data_store[idx]
      if type(l) is not str:
        dat = l.decode().replace("\\n", "\n").replace("\\t", "\t").strip()
      if dat[0] == "{" and dat[-1] == "}":
        try:
          dat = json.loads(l)
        except:
          pass
      return dat
    
    embedder = self.embedder
    if type(query) in (np.array, torch.Tensor):
      embedding = query
      query = None
    assert embedding is None or self.parents is not None
    if embedding is None and query is not None and hasattr(self, 'downsampler') and self.downsampler is not None:
      embedding = self.get_embeddings(query)
      if not hasattr(self, 'whoosh_ix') or self.whoosh_ix is None:
        query = None
    embedding_search_results = embeddings_search(embedding, mmap_file= self.mmap_file, mmap_len=self.mmap_len, embed_dim=self.embed_dim,  dtype=self.dtype, \
                                  parents=self.parents, clusters=self.clusters, top_parent_idxs=self.top_parent_idxs, \
                                  top_parents=self.top_parents, parent2idx=self.parent2idx, k=k)
    if limit is None: 
      cnt = 10^6
    else:
      cnt = limit
    if query is not None:        
      assert hasattr(self, 'whoosh_ix'), "must be created with bm25 indexing"
      with self.whoosh_ix.searcher() as searcher:
        if type(query) is str:
           query = QueryParser("content", self.whoosh_ix.schema).parse(query)
        results = searcher.search(query, limit=limit)
        if embedding is None or do_bm25_only:
          for r in results:
            data = _get_data(int(r['id']))
            if type(data) is dict:
              data['id'] = int(r['id'])
              yield data
            else:
              yield {'id': int(r['id']), 'text': data}
            cnt -= 1
            if cnt <= 0: return
        else:
          idxs = []
          key_terms = []
          n_chunks = 0
          for r in results:
             idxs.append(int(r['id']))
             key_terms.append([]) # r.key_terms())
             n_chunks += 1
             if n_chunks > chunk_size:
                embedding_results = {}
                for _, r in zip(range(chunk_size), embedding_search_results):
                  embedding_results[r[0]] = ([], r[1])
                idxs = [idx for idx in idxs if idx not in embedding_results]
                embeddings = torch.from_numpy(np_memmap(self.mmap_file, shape=[self.mmap_len, self.embed_dim], dtype=self.dtype)[idxs]).to(device)
                results = cosine_similarity(embedding, embeddings)
                for idx, score, key_term in zip(idxs, results, key_terms):
                   embedding_results[idx] = (key_term, score.item())
                embedding_results = list(embedding_results.items())
                embedding_results.sort(key=lambda a: a[1][1], reverse=True)
                for idx, score_keyterm in embedding_results:
                  data = _get_data(idx)
                  if type(data) is dict:
                    data['id'] = idx
                    data['score'] = score_keyterm[1]
                    yield data
                  else:
                    yield {'id': idx, 'text': data, 'score': score_keyterm[1]}
                  cnt -= 1
                  if cnt <= 0: return
                idxs = []
                key_terms = []
                n_chunk = 0
          if idxs:
            embedding_results = {}
            for _, r in zip(range(chunk_size), embedding_search_results):
              embedding_results[r[0]] =  ([], r[1])
            idxs = [idx for idx in idxs if idx not in embedding_results]
            embeddings = torch.from_numpy(np_memmap(self.mmap_file, shape=[self.mmap_len, self.embed_dim], dtype=self.dtype)[idxs]).to(device)
            results = cosine_similarity(embedding, embeddings)
            for idx, score, key_term in zip(idxs, results, key_terms):
               embedding_results[idx] = (key_term, score.item())
            embedding_results = list(embedding_results.items())
            embedding_results.sort(key=lambda a: a[1][1], reverse=True)
            for idx, score_keyterm in embedding_results:
               data = _get_data(idx)
               if type(data) is dict:
                 data['id'] = idx
                 data['score'] = score_keyterm[1]
                 yield data
               else:
                 yield {'id': idx, 'text': data, 'score': score_keyterm[1]}
               cnt -= 1
               if cnt <= 0: return
    #return any stragglers         
    for r in embedding_search_results:
       data = _get_data(r[0])
       if type(data) is dict:
         data['id'] = r[0]
         data['score'] = r[1]
         yield data
       else:
         yield {'id': r[0], 'text': data, 'score': r[1]}
       cnt -= 1
       if cnt <= 0: return
    
        
  def save_pretrained(self, idx_dir=None):
    if idx_dir is not None:
      if self.idx_dir != idx_dir:
        os.system(f"cp -rf {self.idx_dir} {idx_dir}")
    else:
      idx_dir = self.idx_dir
    
    content_data_store = self.content_data_store
    mmap_file = self.mmap_file
    old_idx_dir = self.idx_dir 
    self.idx_dir = None
    if self.mmap_file.startswith(old_idx_dir):
      self.mmap_file = self.mmap_file.split("/")[-1]
    if hasattr(self, 'content_data_store') and self.content_data_store is not None:
      if type(self.content_data_store) is GzipByLineIdx:
        self.content_data_store = None
      elif type(self.content_data_store) is FileByLineIdx:
        fobj = self.content_data_store.fobj
        self.content_data_store.fobj = None  
    device2 = "cpu"
    if self.downsampler is not None:
      device2 = next(self.downsampler.parameters()).device
      self.downsampler.cpu()
    for field in dir(self):
      if field.startswith("downsampler_"):
            downsampler = getattr(self, field)
            if downsampler is not None:
              setattr(self, field, downsampler.cpu())
    parents = self.parents
    self.parents = None
    torch.save(self, open(f"{idx_dir}/search_index.pickle", "wb"))
    self.mmap_file = mmap_file
    self.idx_dir = old_idx_dir
    self.content_data_store = content_data_store
    if self.downsampler is not None:
      self.downsampler.to(device2)
    self.parents = parents
    if type(self.content_data_store) is FileByLineIdx:
      self.content_data_store.fobj = fobj
      
  @staticmethod
  def from_pretrained(filename=None, idx_dir=None):
      global device
      assert idx_dir is not None or filename is not None
      if idx_dir is None: idx_dir = f"{filename}_idx"
      self = torch.load(open(f"{idx_dir}/search_index.pickle", "rb"))
      self.idx_dir = idx_dir
      if os.path.exists(f"{idx_dir}/{self.mmap_file}"):
        self.mmap_file = f"{idx_dir}/{self.mmap_file}"
      if filename:
        if filename.endswith(".gz"):
          self.content_data_store = GzipByLineIdx.open(filename)
        elif type(self.content_data_store) is FileByLineIdx:
          self.content_data_store.fobj=open(filename, "rb")
      self.downsampler.eval().to(device)
      if self.clusters: self.recreate_parents_data()
      if self.prototype_sentences and self.prototypes is None: 
        self.prototypes = self.get_embeddings(self.prototype_sentences)
      parents = self.parents
      del self.parents
      self.register_buffer('parents', parents)
      prototypes = self.prototypes
      del self.prototypes
      self.register_buffer('prototypes', prototypes)
    
      return self
        
