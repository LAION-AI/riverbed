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

def _unpickle(state):
    """Create a new ``IndexedGzipFile`` from a pickled state.
    :arg state: State of a pickled object, as returned by the
                ``IndexedGzipFile.__reduce__`` method.
    :returns:   A new ``IndexedGzipFile`` object.
    """
    tell  = state.pop('tell')
    index = state.pop('index')
    gzobj = Gush(**state)

    if index is not None:
        gzobj.import_index(fileobj=io.BytesIO(index))

    gzobj.seek(tell)

    return gzobj

class Gush(igzip.IndexedGzipFile):
    """A Gush file is a gzip file with added funcitonalites directed at information retreival.
    It is a single gzip file that can be searched via BM25, vector based search and accessed 
    by line number.
    
      for r in obj.search("test"): print (r)

      for r in obj.search(numpy_or_pytorch_tensor): print (r)
      
      for r in obj.search("test", numpy_or_pytorch_tensor): print (r)
    
    This class allows in addition to the functionality of IndexedGzipFile, indexing 
    each line using whoosh, and access to a specific line based on the seek point of 
    the line, using the __getitem__ method.
    
      obj[5]
    
    Additionally, a (conginguous) list or slice can be used, which will be more efficient 
    then doing line by line access. 
    
      obj[5:10]
    
    The underlying vectors can also be accessed using line numbers.
    
      obj.vecs[5]

      objs.vecs[5:10]
    
    
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
        :arg index_whoosh:     Whether to index the file using whoose for 
                               BM25 searching
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
        :arg  mmap_file:      Optional, must be passed as a keyword argument.
                              This is the file name for the vectors representing 
                              each line in the gzip file.
        :arg  parent_vecs:      Optional, must be passed as a keyword argument.
                                This is a numpy or pytorch vector of the form,
                                assuming a 4 level hiearcical structure:
                                [level 3 parents ... level 1 parents ... level 0 parents]
                                Where level 4 parents are the top level parents. 
                                This structure is used for approximage nearest neighbor search.
        :arg parent_levels:  Optional, must be passed as a keyword argument. If parent_vecs
                              are passed, this param must be also passed. It is an array repreesnting the
                              paent level. e.g., 
                              [4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, \
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]
        :arg parent2idx:      Optional, must be passed as a keyword argument. If parent_vecs
                              are passed, this param must be also passed. It is a list of lists which 
                              represents the mapping between one parents level to the next, and where 
                              the level 0 parents points to the actual leaf vectors in the mmap_file. e.g., 
                              [[3,...5], [6, ... 9], [10, ... 14], # level 4 parents points to level 3 parents
                               [15... 25], .....                   # level 3 parents point to level 2 parents.
                               ...
                               [46, 10020, ....], [10, 1, ... 100], ... # level 0 parents point to vecs in the mmap_file.
                              
                              
                              
                                
        """
        f = kwargs.get("filename") 
        if args and not f:
          f = args[0]
        need_export_index = False
        if f:
          if not os.path.exists(f+"idx"):
            need_export_index = True
            os.makedirs(f+"idx")
          if not os.path.exists(f+"idx/index.pickle"):
            need_export_index = True
          else:
            kwargs['index_file'] = kwargs.pop('index_file', f+"idx/index.pickle")
        if 'file_size' in kwargs:
          file_size = self.file_size = kwargs.pop('file_size', None)
          need_export_index = False
        self.line2seekpoint  = kwargs.pop('line2seekpoint', None)
        self.parent_vecs  = kwargs.pop('parent_vecs', None)
        self.parent2idx  = kwargs.pop('parent2idx', None)
        self.mmap_file = kwargs.pop('mmap_file', f+"idx/index.mmap")
        if need_export_index and 'auto_build' not in kwargs: kwargs['auto_build'] = True
        super(Gush, self).__init__(*args, **kwargs)
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
        if f and need_export_index: 
            self.export_index(f+"idx/index.pickle")
        if 'index_whoosh' in kwargs and kwargs['index_whoosh']:
          assert f, "need a index name to store the whoosh index"
          schema = Schema(id=ID(stored=True), content=TEXT)
          #TODO determine how to clear out the whoosh index besides rm -rf _M* MAIN*
          self.whoosh_ix = create_in(f+"idx", schema)  
          if need_export_index: 
            writer self.whoosh_ix.writer()
            for idx, l in enumerate(self):
              writer.add_document(id=str(id),
                                  content=l.decode().strip())  
            writer.commit()
            self.seek(0, os.SEEK_END)
    
    def whoosh_searcher(self):
        assert hasattr(self, 'whoosh_ix'), "must be created with whoosh_index set"
        return self.whoosh_ix.searcher()

    def search(self, query=None, vec=None, lookahead_cutoff=100, k=5):
        if type(query) in (nd.array, torch.Tensor):
          vec = query
          query = None
        if query is not None:        
          assert hasattr(self, 'whoosh_ix'), "must be created with whoosh_index set"
          with self.whoosh_searcher():
            if type(query) is str:
               query = QueryParser("content", self.whoosh_ix.schema).parse(query)
            results = searcher.search(query)
            idxs = []
            n_chunks = 0
            if vec is None:
              for r in results:
               yield (int(r['id']), self[int(r['id'])].decode().strip())
            else:
              for r in results:
               idxs.append(int(r['id']))
               n_chunks += 1
               if n_chunks > chunk_size:
                  vecs = torch.from_numpy(np_memmap(self.mmap_file, shape=self.vec_shape)[idxs]).to(device)
                  results = cosine_similarity(vec.unsqueeze(0), vecs)
                  results = results.sort()
                  for idx, score in zip(results.indexes, results.values):
                     idx = idxs[idx]
                     yield (idx, self[idx].decode().strip()), score)
                  idxs = []
                  n_chunk = 0
            if idxs:
                vecs = torch.from_numpy(np_memmap(self.mmap_file, shape=self.vec_shape)[idxs]).to(device)
                results = cosine_similarity(vec.unsqueeze(0), vecs)
                results = results.sort()
                for idx, score in zip(results.indexes, results.values):
                   idx = idxs[idx]
                   yield (idx, self[idx].decode().strip()), score)
        else:
                assert vec is not None        
                #do ANN search
                if type(self.parents) is not torch.Tensor:
                   self.parents = torch.from_numpy(self.parents).to(device)
                vecs = self.parents[:self.num_top_level_parents]
                idx2idx = list(range(self.num_top_level_parents))
                for _ in range(self.parent_level[0]):
                   results = cosine_similarity(vec.unsqueeze(0), vecs)
                   results = results.top_k(k)
                   children = itertools.chain(*[self.parent2child[idx2idx[idx]] for idx in results.indices])
                   idx = results.indices[0]
                   if self.parent_level[idx2idx[idx]] == 0: #we are at the leaf nodes
                        idxs = []
                        n_chunks = 0
                        for child_id in children:
                           idxs.append(child_id)
                           n_chunks += 1
                           if n_chunks > chunk_size:
                              vecs = torch.from_numpy(np_memmap(self.mmap_file, shape=self.vec_shape)[idxs]).to(device)
                              results = cosine_similarity(vec.unsqueeze(0), vecs)
                              results = results.sort()
                              for idx, score in zip(results.indexes, results.values):
                                 idx = idxs[idx]
                                 yield (idx, self[idx].decode().strip()), score)
                              idxs = []
                              n_chunk = 0
                        if idxs:
                              vecs = torch.from_numpy(np_memmap(self.mmap_file, shape=self.vec_shape)[idxs]).to(device)
                              results = cosine_similarity(vec.unsqueeze(0), vecs)
                              results = results.sort()
                              for idx, score in zip(results.indexes, results.values):
                                 idx = idxs[idx]
                                 yield (idx, self[idx].decode().strip()), score)
                   else:
                     vecs = self.parents[children]
                     idx2idx = children    

                
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
        if type(self.parents) is torch.Tensor:
           self.parents = self.parents.cpu()
        state = {
            'filename'         : fobj.filename,
            'auto_build'       : fobj.auto_build,
            'spacing'          : fobj.spacing,
            'window_size'      : fobj.window_size,
            'readbuf_size'     : fobj.readbuf_size,
            'readall_buf_size' : fobj.readall_buf_size,
            'buffer_size'      : self._IndexedGzipFile__buffer_size,
            'line2seekpoint'   : self.line2seekpoint,
            'parent_vecs'   : self.parent_vecs,                
            'parent2line'   : self.parent2line,                      
            'file_size'   : self.file_size,
            'tell'             : self.tell(),
            'index'            : index}

        if type(self.parents) is torch.Tensor:
           self.parents = self.parents.to(device)
        
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
