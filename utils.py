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
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from fast_pytorch_kmeans import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import random

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

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
         
  
#cluster pruning based approximate nearest neightbor search. See https://nlp.stanford.edu/IR-book/html/htmledition/cluster-pruning-1.html
def pytorch_ann_search(vec, mmap_file, shape, dtype, parents, num_top_level_parents, parent_levels, parent2idx, chunk_size=10000):
  vecs = parents[:num_top_level_parents]
  idx2idx = list(range(num_top_level_parents))
  for _ in range(parent_levels[0]):
    results = cosine_similarity(vec.unsqueeze(0), vecs)
    results = results.top_k(k)
    children = itertools.chain(*[parent2idx[idx2idx[idx]] for idx in results.indices])
    idx = results.indices[0]
    if parent_levels[idx2idx[idx]] == 0: #we are at the leaf nodes
      idxs = []
      n_chunks = 0
      for child_id in children:
         idxs.append(child_id)
         n_chunks += 1
         if n_chunks > chunk_size:
            vecs = torch.from_numpy(np_memmap(mmap_file, shape=shape, dtype=dtype)[idxs]).to(device)
            results = cosine_similarity(vec.unsqueeze(0), vecs)
            results = results.sort()
            for idx, score in zip(results.indexes, results.values):
               idx = idxs[idx]
               yield (idx, score)
            idxs = []
            n_chunk = 0
      if idxs:
         vecs = torch.from_numpy(np_memmap(mmap_file, shape=shape, dtype=dtype)[idxs]).to(device)
         results = cosine_similarity(vec.unsqueeze(0), vecs)
         results = results.sort()
         for idx, score in zip(results.indexes, results.values):
            idx = idxs[idx]
            yield (idx, score)
    else:
      vecs = parents[children]
      idx2idx = children   
      
# incremental hiearchical clustering from vectors in the mmap file.       
# with a max of 4 levels, and each node containing 200 items, we can have up to 1.6B items approximately
# span2cluster_label maps a span to a parent span. spans can be of the form int|(int,int).
# leaf nodes are ints. non-leaf nodes are (int,int) tuples
# clusters maps cluster_label => list of spans  
def create_hiearchical_clusters(clusters, span2cluster_label, mmap_file, shape, dtype, cluster_idxs=None, max_level=4, max_cluster_size=200, \
                               min_overlap_merge_cluster=2, prefered_leaf_node_size=None, kmeans_batch_size=10000):
  global device
  for span, label in span2cluster_label.items():
    clusters[label] = clusters.get(label,[]) + [span]
  if prefered_leaf_node_size is None: prefered_leaf_node_size = max_cluster_size
  cluster_vecs = np_memmap(mmap_file, shape=shape, dtype=dtype)
  # first cluster leaves at level 0. 
  # the spans are the indexes themselves, so no need to map using all_spans
  all_spans = None
  for level in range(max_level):
    assert level == 0 or (all_spans is not None and cluster_idxs is not None)
    if cluster_idxs is None: 
      len_spans = cluster_vecs.shape[0]
    else:
      len_spans = len(cluster_idxs)
    recluster_at = max(0,len_spans-4*int(.7*kmeans_batch_size))
    for rng in range(0, len_spans, int(.7*kmeans_batch_size)):
        max_rng = min(len_spans, rng+int(.7*kmeans_batch_size))
        #create the next batch to cluster
        if cluster_idxs is None:
          spans = list(range(rng, max_rng))
          spans.extend([idx for idx in range(rng) if (all_spans is not None and all_spans[idx] not in span2cluster_label) or \
                        (all_spans is None and idx not in span2cluster_label)])
        else:
          spans = cluster_idxs[rng: max_rng] 
          spans.extend([idx for idx in range(rng) if cluster_idxs[:rng] if (all_spans is not None and all_spans[idx] not in span2cluster_label) or \
                        (all_spans is None and idx not in span2cluster_label)])
        if level == 0:
          already_clustered = [idx for idx in range(cluster_vecs.shape[0]) if idx in span2cluster_label]
        else:
          already_clustered = [idx for idx, span in enumerate(all_spans) if span in span2cluster_label]
        if len(already_clustered) > int(.3*kmeans_batch_size):
          spans.extend(random.sample(already_clustered, int(.3*kmeans_batch_size)))
        else:
          spans.extend(already_clustered)
        # get the vector indexs for the cluster
        if level == 0:
          vector_idxs = spans
        else:
          spans = [all_spans[idx] for idx in spans] 
          vector_idxs = [span[1] for span in spans]
        #do kmeans clustering in batches with the vector indexes
        if level == 0:
          true_k = len(vector_idxs)/prefered_leaf_node_size
        else:
          true_k = len(vector_idxs)/max_cluster_size
  
        if device == 'cuda':
          kmeans = KMeans(n_clusters=true_k, mode='cosine')
          km_labels = kmeans.fit_predict(torch.from_numpy(cluster_vecs[vector_idxs]).to(device))
          km_labels = [l.item() for l in km_labels.cpu()]
        else:
          km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                                        init_size=max(true_k*3,1000), batch_size=1024).fit(cluster_vecs[vector_idxs])
          km_labels = km.labels_
          
        #put into a temporary cluster
        tmp_cluster = {}  
        for span, label in zip(spans, km_labels):
          tmp_cluster[label] = tmp_cluster.get(label, [])+[span]
          
        #create unique (level, id) labels and merge if necessary
        for a_cluster in tmp_cluster.values():
            cluster_labels = [span2cluster_label[span] for span in a_cluster if span in span2cluster_label]
            # merge with previous clusters if there is an overlap
            if cluster_labels:
              most_common = Counter(cluster_labels).most_common(1)[0]
              if most_common[1] >= min_overlap_merge_cluster: 
                label = most_common[0]
                need_labels = [span for span in a_cluster if span2cluster_label.get(span) in (label, None)]
                for span in need_labels:
                   if span not in clusters.get(label, []):
                      clusters[label] = tmp_cluster.get(label, []) + [span]
                   span2cluster_label[span] = label
                    
            #label the rest of the cluster that hasn't been merged        
            need_labels = [span for span in a_cluster if span not in span2cluster_label]
            if need_labels:
              if type(need_labels[0]) is int:
                label = (level, need_labels[0])
              else:
                label = (level, need_labels[0][1])
              for span in  need_labels:
                 if span not in clusters.get(label, []):
                    clusters[label] = tmp_cluster.get(label, []) + [span]
                 span2cluster_label[span] = label

          # re-cluster any small clusters or break up large clusters   
        if rng >= recluster_at and max_rng != len_spans:     
            for parent, spans in clusters.items(): 
              if len(spans) < prefered_leaf_node_size*.5:
                for span in spans:
                  del span2cluster_label[span]
              elif len(spans) > max_cluster_size:
                  for token in spans[int(max_cluster_size*.75):]:
                    del span2cluster_label[token]
                  
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
    """ A class for accessing a file by line numbers. Requires  fobj that provides a seek, and tell method must be passed.
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

           
class SearcherIdx:
  def __init__(self,  mmap_file, shape, dtype=np.float32, parents=None, parent_levels=None, parent2idx=None, clusters=None, auto_create_ann_idx=True, \
               bm25_filename=None, bm25_fobj=None,  bm25_field="text", filebyline=None):
    """
        Clusters and performs approximate nearest neighbor search on a memmap file. Also provides a wrapper for Whoosh BM25.
        :arg  mmap_file:      Optional, must be passed as a keyword argument.
                              This is the file name for the vectors representing 
                              each line in the gzip file. Used for ANN search.
        :arg shape             Optional, must be passed as a keyword argument.
                              This is the shape of the mmap_file.                              
        :arg dtype             Optional, must be passed as a keyword argument.
                              This is the dtype of the mmap_file.                              
        :arg  parents:      Optional, must be passed as a keyword argument.
                                This is a numpy or pytorch vector of the form,
                                assuming a 4 level hiearcical structure:
                                [level 3 parents ... level 1 parents ... level 0 parents]
                                Where level 4 parents are the top level parents. 
                                This structure is used for approximage nearest neighbor search.
        :arg parent_levels:  Optional, must be passed as a keyword argument. If parents
                              are passed, this param must be also passed. It is an array repreesnting the
                              paent level. e.g., 
                              [4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, \
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]
        :arg parent2idx:      Optional, must be passed as a keyword argument. If parents
                              are passed, this param must be also passed. It is a list of lists which 
                              represents the mapping between one parents level to the next, and where 
                              the level 0 parents points to the actual leaf vectors in the mmap_file. e.g., 
                              [[3,...5], [6, ... 9], [10, ... 14], # level 4 parents points to level 3 parents
                               [15... 25], .....                   # level 3 parents point to level 2 parents.
                               ...
                               [46, 10020, ....], [10, 1, ... 100], ... # level 0 parents point to vecs in the mmap_file.
                               
                               If mmap_file, shape, parents, parent_levels and parent2idx are not provided,
                               vector based search will not be available. 
        NOTE: Either pass in the parents, parent_levels, parent2idx data or pass in clusters to create the prior three data. 
          if none of these are passed and auto_create_idx is set, then the data in the mmap file will be clustered and the 
          data structure will be created.
        :filebyline     Optional. The access for a file by lines.
        :bm25_filename:     Optional. If the filename and fobj are both None, then no bm25 search is provided.
        :bm25_fobj:         Optional. File object for bm25 indexing and searching.
        :arg bm25_field:     Optional. Defaults to "text". If the data is in jsonl format,
                             this is the field that is Whoosh/bm25 indexed.
      USAGE:
      
        for r in obj.search("test"): print (r)

        for r in obj.search(numpy_or_pytorch_tensor): print (r)

        for r in obj.search("test", numpy_or_pytorch_tensor): print (r)

      """
    global device
    if clusters is not None:
      self.recreate_idx(mmap_file, shape, dtype, clusters)
    else:       
     if parents is None and auto_create_ann_idx:
      self.recreate_idx(mmap_file, shape, dtype)
     else:
      self.mmap_file, self.shape, self.dtype, self.parents, self.parent_levels, self.parent2idx = \
             mmap_file, shape, dtype, parents, parent_levels, parent2idx
     if self.parent_levels is not None:
       self.num_top_parents = len([a for a in self.parent_levels if a == max(self.parent_levels)])
     else:
       self.num_top_parents = 0
    if bm25_filename or bm25_fobj:
       self.recreate_whoosh_idx(bm25_field, bm25_filename, bm25_fobj, filebyline)
    
  def recreate_whoosh_idx(self, bm25_field="text", bm25_filename=None, bm25_fobj=None, filebyline=None):
      assert bm25_filename is not None or bm25_fobj is not None
      if bm25_filename is None:
        if not hasattr(self, 'bm25_filename'): 
          bm25_filename = self.bm25_filename = f"__tmp__{random.randint(0,1000000)}"
        else:
          bm25_filename = self.bm25_filename
      self.bm25_field = bm25_field
      self.bm25_filename = bm25_filename
      if bm25_fobj is None:
        bm25_fobj = open(bm25_filename, "rb")
      self.bm25_fobj = bm25_fobj  
      schema = Schema(id=ID(stored=True), content=TEXT)
      #TODO determine how to clear out the whoosh index besides rm -rf _M* MAIN*
      bm25_fobj.seek(0, os.SEEK_END)
      self.whoosh_ix = create_in(bm25_filename+"_idx", schema)  
      writer = self.whoosh_ix.writer()
      bm25_fobj.seek(0, os.SEEK_END)
      for idx, l in enumerate(bm25_fobj):
          l =l.decode().replace("\\n", "\n").strip()
          if l[0] == "{" and l[-1] == "}":
            content = l.split(self.bm25_field+'": "')[1]
            content = content.split('", "')[0]
          else:
            content = l
            writer.add_document(id=str(idx),
                                    content=content)  
      writer.commit()
      bm25_fobj.seek(0, os.SEEK_END)
      self.filebyline = filebyline
      if self.filebyline is None: self.filebyline = FileByLineIdx(fobj=bm25_fobj)
               
  def whoosh_searcher(self):
      return self.whoosh_ix.searcher()

  def search(self, query=None, vec=None, lookahead_cutoff=100, k=5, chunk_size=10000):
        if type(query) in (np.array, torch.Tensor):
          vec = query
          query = None
        assert vec is None or self.parents is not None
        if query is not None:        
          assert hasattr(self, 'whoosh_ix'), "must be created with bm25 search set"
          with self.whoosh_searcher() as searcher:
            if type(query) is str:
               query = QueryParser("content", self.whoosh_ix.schema).parse(query)
            results = searcher.search(query)
            if vec is None:
              for r in results:
               yield (int(r['id']), self.filebyline[int(r['id'])].decode().strip())
            else:
              idxs = []
              n_chunks = 0
              for r in results:
                 idxs.append(int(r['id']))
                 n_chunks += 1
                 if n_chunks > chunk_size:
                    vecs = torch.from_numpy(np_memmap(self.mmap_file, shape=self.shape, dtype=self.dtype)[idxs]).to(device)
                    results = cosine_similarity(vec.unsqueeze(0), vecs)
                    results = results.sort()
                    for idx, score in zip(results.indexes, results.values):
                       idx = idxs[idx]
                       yield (idx, self.filebyline[idx].decode().strip(), score)
                    idxs = []
                    n_chunk = 0
              if idxs:
                  vecs = torch.from_numpy(np_memmap(self.mmap_file, shape=self.shape, dtype=self.dtype)[idxs]).to(device)
                  results = cosine_similarity(vec.unsqueeze(0), vecs)
                  results = results.sort()
                  for idx, score in zip(results.indexes, results.values):
                     idx = idxs[idx]
                     yield (idx, self[idx].decode().strip(), score)
        else:
                assert vec is not None        
                #do ANN search
                if type(self.parents) is not torch.Tensor:
                   self.parents = torch.from_numpy(self.parents).to(device)
                if hasattr(self, 'filebyline'):
                  for r in pytorch_ann_search(vec, self.mmap_file, self.shape,  self.dtype, \
                                            self.parents, self.num_top_level_parents, self.parent_levels, self.parent2idx):
                    yield (r[0], self.filebyline[r[0]].decode().strip(), r[1])
                else:
                  for r in pytorch_ann_search(vec, self.mmap_file, self.shape,  self.dtype, \
                                            self.parents, self.num_top_level_parents, self.parent_levels, self.parent2idx):
                    yield (r[0], r[1])
                  
  def recreate_ann_idx(self,  mmap_file, shape, dtype, clusters=None):
    global device
    self.mmap_file, self.shape, self.dtype = mmap_file, shape, dtype
    if clusters is None:
      clusters, _ = self.cluster()
    cluster_info = list(clusters.items())
    cluster_info.sort(key=lambda a: a[0][0], reverse=True)
    self.parents = torch.from_numpy(np_memmap(mmap_file, shape=shape, dtype=dtype)[[a[0][1] for a in cluster_info]]).to(device)
    self.parent_levels = [a[0][0] for a in cluster_info]
    self.num_top_parents = len([a for a in self.parent_levels if a == max(self.parent_levels)])
    label2idx = dict([(a[0], idx) for idx, a in enumerate(cluster_info)]) 
    self.parent2idx = [[a if type(a) is int  else label2idx[a] for a in a_cluster] for _, a_cluster in cluster_info]

  
  def cluster(self, clusters=None, span2cluster_label=None, cluster_idxs=None, max_level=4, max_cluster_size=200, \
                               min_overlap_merge_cluster=2, prefered_leaf_node_size=None, kmeans_batch_size=10000):
    if clusters is None: cluster = {}
    if span2cluster_label is None: span2cluster_label = {}
    return create_hiearchical_clusters(clusters, span2cluster_label, self.mmap_file, self.shape, self.dtype, cluster_idxs=cluster_idxs, max_level=max_level, \
                                      max_cluster_size=max_cluster_size, min_overlap_merge_cluster=min_overlap_merge_cluster, \
                                      prefered_leaf_node_size=prefered_leaf_node_size, kmeans_batch_size=kmeans_batch_size)
  
def _unpickle(state):
    """Create a new ``IndexedGzipFile`` from a pickled state.
    :arg state: State of a pickled object, as returned by the
                ``IndexedGzipFile.__reduce__`` method.
    :returns:   A new ``IndexedGzipFile`` object.
    """
    tell  = state.pop('tell')
    index = state.pop('index')
    gzobj = IndexedGzipFileExt(**state)

    if index is not None:
        gzobj.import_index(fileobj=io.BytesIO(index))

    gzobj.seek(tell)

    return gzobj

  
class IndexedGzipFileExt(igzip.IndexedGzipFile):
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
        """Create an ``LineIndexGzipFile``. The file may be specified either
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
          if not os.path.exists(filename+"_idx/index.pickle"):
            need_export_index = True
          else:
            kwargs['index_file'] = kwargs.pop('index_file', filename+"_idx/index.pickle")
        
        if 'file_size' in kwargs:
          file_size = self.file_size = kwargs.pop('file_size', None)
          need_export_index = False
        self.line2seekpoint  = kwargs.pop('line2seekpoint', None)
        if need_export_index and 'auto_build' not in kwargs: kwargs['auto_build'] = True
        super(IndexedGzipFileExt, self).__init__(*args, **kwargs)
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
          self.export_index(filename+"_idx/index.pickle")

    def __reduce__(self):
        """Used to pickle an ``LineIndexGzipFile``.
        Returns a tuple containing:
          - a reference to the ``unpickle`` function
          - a tuple containing a "state" object, which can be passed
            to ``unpickle``.
        """
        
        fobj = self._IndexedGzipFile__igz_fobj

        if (not fobj.drop_handles) or (not fobj.own_file):
            raise pickle.PicklingError(
                'Cannot pickle IndexedGzipFile that has been created '
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

        return (_unpickle, (state, ))

    
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
    
