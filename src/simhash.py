#@title Simhash Code
#simhash hashing and clustering based on Chenghao Mou's awesome: https://github.com/bigscience-workshop/data_tooling/blob/master/ac_dc/deduplicate/ which is under Apache 2
from typing import Dict
import numpy as np
import simhash
import regex as re
from itertools import product
import os, tqdm
from collections import Counter, defaultdict, deque
from typing import Dict, Set
import tqdm 
import random
import itertools
import faiss

PUNCTUATION_REGEX = re.compile(r"\p{P}")
DIGIT_REGEX = re.compile(r"\d")

def hashing(
    document: str,
    tokenization: str = "character",
    window_size: int = 20,
    ignore_punctuation: bool = True,
    lowercase: bool = True
) -> Dict[str, int]:
    """Hashing a document with SimHash.
    spanmeters
    ----------
    document : str
        The text to use for hashing, by default "text"
    tokenization : str, optional
        Method to use for tokenization, by default "character"
    window_size : int, optional
        The size of the token window, by default 6
    ignore_punctuation : bool, optional
        To ignore punctuation or not, by default True
    lowercase : bool, optional
        To lowercase the text or not, by default True
    Returns
    -------
    int: The hash code

    Raises
    ------
    Exception
        Unrecognized tokenization spanmeter
    """
    if lowercase:
        document = document.lower()

    if ignore_punctuation:
        document = PUNCTUATION_REGEX.sub("", document)

    if tokenization == "character":
        document = " ".join(document.split())
        tokens = [
            str.encode(document[i : i + window_size])
            for i in range(len(document) - window_size)
        ]
        if not tokens: tokens = [str.encode(document)]
    elif tokenization == "punctuation":
        tokens0 = PUNCTUATION_REGEX.split(document)
        tokens = [
            str.encode(" ".join(tokens0[i : i + window_size]))
            for i in range(len(tokens0) - window_size)
        ]
        if not tokens: tokens = [str.encode(t) for t in tokens0]
    elif tokenization == "space":
        tokens0 = document.split(" ") #consider whether we want to just use .split() to match \n and \t
        tokens = [
            str.encode(" ".join(tokens0[i : i + window_size]))
            for i in range(len(tokens0) - window_size)
        ]
        if not tokens: tokens = [str.encode(t) for t in tokens0]
    # we could try other types of tokenizations such as stemming and removal of stopwords
    else:
        raise Exception(f"Unrecognized tokenization spanmeter {tokenization}")
    assert tokens
    #TODO: the hash code is actually a 64bit int. Check sys.maxsize. 
    #Was having a problem with serialzing np.int64 in json so i casted to int. 
    #might not be an issue in parquet in which case we should revert back to np.int64.
    return int(simhash.compute(map(simhash.unsigned_hash, tokens)))


def index_clusters_batch_python(visited, hash2cluster, cluster2hash, hashes, num_blocks, hamming_distance):
    """
    Create clusters within hamming distance. 
    Collapses a->b, b->c to all be in the same cluster.
    NOTE: this isn't always true that a and c are within hamming_distance. 
    NOTE: The cluster_id is the hashcode of the first item in the cluster and thus can be used to do further clustering and hamming distance matching.
    """
    matches = simhash.find_all(hashes, num_blocks, hamming_distance)
    graph = defaultdict(dict)
    for x, y in matches:
      graph[x][y] = True
      graph[y][x] = True
    hashes = set(hashes)
    
    while hashes:
        hash = hashes.pop()
        if hash in visited:
            continue

        # BFS to find the cluster
        if hash not in graph:
            hash2cluster[hash] = -1
            continue

        q = deque([hash])
        visited.add(hash)
        cluster_id = hash
        hash2cluster[hash] = cluster_id
        cluster2hash[cluster_id] = cluster2hash.get(cluster_id, []) + [hash]

        while q:
            node = q.popleft()
            for neighbor in graph[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                q.append(neighbor)
                hash2cluster[neighbor] = cluster_id
                cluster2hash[cluster_id] = cluster2hash.get(cluster_id, []) + [neighbor]

    return visited, hash2cluster, cluster2hash,

def index_clusters_python(hashes, num_blocks, hamming_distance, do_sort=True, batch_size=900000, verbose=False):
  """ Incrementally find clusters of int64 bit hashes of *around* the same hamming distance from each other. 
  Returns hash2cluster and cluster2hash dicts, where the ids are all int64 bit hashes.
  """
  if do_sort: 
    hashes.sort()
  # we are assuming no exact duplicates. if we want to deal with exact duplicates, we can easily just collapse them in sequence
  # since this is a sorted list
  cluster2hash = {}
  hash2cluster = {}
  visited: Set[int] = set()
  if len(hashes) <= batch_size:
    visited, hash2cluster, cluster2hash = find_clusters_batch(visited, hash2cluster, cluster2hash, hashes, num_blocks, hamming_distance)
    return hash2cluster, cluster2hash
  batch_size2 = int(batch_size/2)
  if verbose:
    a_iter = tqdm.tqdm(range(0, len(hashes), batch_size2))
  else:
    a_iter = range(0, len(hashes), batch_size2)
  for rng in a_iter:
    max_rng = min(len(hashes), rng+batch_size2)
    hashes2 = hashes[rng:max_rng]
    hashes3 = []
    if cluster2hash:
      iterms_per_clusters = int(max(1, batch_size2/len(cluster2hash)))
      hashes3 = list(itertools.chain(*[val[:iterms_per_clusters] for val in cluster2hash.values()]))
      if len(hashes3) > int(batch_size2/2):
        hashes3 = random.sample(hashes3, batch_size2)
    if rng > 0 and len(hashes3) < batch_size2:
        hashes3 = list(set(hashes3+random.sample(hashes[:rng], batch_size2-len(hashes3))))
    #print (len(hashes3))
    hashes2.extend(hashes3)
    #print (len(hashes2))
    visited, hash2cluster, cluster2hash = index_clusters_batch_python(visited, hash2cluster, cluster2hash, hashes2, num_blocks, hamming_distance)
  return hash2cluster, cluster2hash

def search_python_only(queries, num_blocks, hamming_distance):
    """
    Create clusters within hamming distance. 
    Collapses a->b, b->c to all be in the same cluster.
    NOTE: this isn't always true that a and c are within hamming_distance. 
    NOTE: The cluster_id is the hashcode of the first item in the cluster and thus can be used to do further clustering and hamming distance matching.
    """
    return simhash.find_all(queries, num_blocks, hamming_distance)
    
    
def index_faiss(hashes, d=16):
    """ 
    hashes: the array of ints representing the simhash
    d: Dimension of the ints.
    
    """
    sqrt_size = int(math.sqrt(len(hashes)))
    
    # Vectors to train the quantizer.
    training = [hashes[i] for i in random.sample(range(len(hashes)), 2*sqrt_size)]


    # Initializing the quantizer.
    quantizer = faiss.IndexBinaryFlat(d)

    sqrt_size = int(math.sqrt(len(hashes)))
    
    # Number of clusters.
    nlist = sqrt_size

    # Initializing index.
    index = faiss.IndexBinaryIVF(quantizer, d, nlist)
    index.nprobe = 4 # Number of nearest clusters to be searched per query. 

    # Training the quantizer.
    index.train(training)

    # Adding the database vectors.
    index.add(hashes)

    return index


def search_faiss(queries, qindices, hamming_distance, k=500, index=None):
    """
    k: Number of nearest neighbors to retrieve per query vector.
    """
    if index is None:
        index = index_faiss(queries)
    ret = []
    # Querying the index.
    D, I = index.search(queries, k)
    for i, matches, indices in zip(qindices, D, I):
        for score, j in zip(matches, indices):
            if score <= hamming_distance:
                ret.append((i, j))
    return ret
    
def incremental_span_and_document_neardedup( dup_span, dup_doc, unformatted_text, formatted_text=None, shingle_size = 5, cleanup_dup_span_limit=1000000, cleanup_dup_doc_limit=1000000, normalize_text=True, keep_first_dup_in_unformatted_text=False, keep_first_dup_in_formatted_text=True, replace_char='*'):
    """
    Given a document text and a dict representing any near duplicate spans and duplicate docs, remove duplicate spans of shingle size from the text.
    The text can be in the form of clean unformatted text, e.g., removed formatting and any extraneous tags, and the corresponding formatted text, 
    Assumes that double spaces denote sentence break in the text, and formatted_text.
    normalize_text will add double spaces between common punctuations and quotes. 
    Return:
    
      doc_is_dup, deduped unformatted_text, deduped formatted_text
        where doc_is_dup is 0 if there is no duplicates, 1 if there are partial span dups, and 2 if the whole document is a near dup.
        text is the original text with any duplicate spans replaced with the replace_char, collapsing multiple replace chars into one char.
      NOTE: the formatted_text are not guaranted to be deduped b/c there may be formatting in between spans that affects deduplication. 
      
    """
    is_dup_chunk={}
    if normalize_text:
      #simple normalize and add double spaces after sentences. TODO, add other lang punc.
      unformatted_text = unformatted_text.replace("! ", "!  ").replace("? ", "?  ").replace(". ", ".  ").replace("．", "．  ").replace("。", "。  ").replace("？", "？  ")\
        .replace("!\" ", "!\"  ").replace("?\" ", "?\"  ").replace(".\" ", ".\"  ").replace("．\"", "．\"  ").replace("。\"", "。\"  ").replace("？\"", "？\"  ")\
        .replace("!” ", "!”  ").replace("?” ", "?”  ").replace(".” ", ".”  ").replace("．”", "．”  ").replace("。”", "。”  ").replace("？”", "？”  ")\
        .replace("!》 ", "!》  ").replace("?》 ", "?》  ").replace(".》 ", ".》  ").replace("．》", "．》  ").replace("。》", "。》  ").replace("？》", "？》  ")\
        .replace("、", "、 ").replace("’s", " 's").replace("`s", " 's").replace("'s", " 's")
    if formatted_text is None: formatted_text = unformatted_text
    text_arr = [a.strip() for a in unformatted_text.split("  ") if a.strip()]
    
    #chunkify into sentences
    chunks = []
    for sent in text_arr:
      if not sent: continue
      if " " not in sent and len(sent) > 20:
          while sent:
            chunks.append(sent[:20])
            sent = sent[20:]
      else:
          chunks.append(sent)
    
    replace_text = " "+replace_char+" "
    shingles = [" ".join(chunks[i : i + shingle_size]) for i in range(len(chunks) - shingle_size)]
    is_dup_within_doc = {}
    unformatted_text = " ".join(unformatted_text.split())
    
    #dedup spans other than the first matching span using shingle_size of sentences (e.g., a span) 
    for ch_idx in range(len(chunks) - shingle_size):
      orig_shingle= " ".join(chunks[ch_idx : ch_idx + shingle_size])
      shingle = DIGIT_REGEX.sub('1', orig_shingle).strip()
      if not shingle: continue
      hashcode = hashing(shingle)
      if hashcode in is_dup_within_doc:
        prev_ch_idx = is_dup_within_doc[hashcode][0]
        prev_chunk = chunks[prev_ch_idx]
        clean_position = unformatted_text.find(prev_chunk)
        formatted_text_position = formatted_text.find(prev_chunk)
        if clean_position >= 0 and formatted_text_position >= 0:
          clean_position += len(shingle)
          formatted_text_position += len(shingle)
          unformatted_text2 = unformatted_text[clean_position+1:]
          formatted_text2 = formatted_text[formatted_text_position+1:]
          if shingle in unformatted_text2:
            unformatted_text2 = unformatted_text2.replace(shingle, replace_text)
          else:
            for chunk in chunks[ch_idx : ch_idx + shingle_size]:
              if len(chunk) > 3: unformatted_text2 = unformatted_text2.replace(chunk, replace_text)
          if shingle in formatted_text2:
            formatted_text2 = formatted_text2.replace(shingle, replace_text)
          else:
            for chunk in chunks[ch_idx : ch_idx + shingle_size]:
              if len(chunk) > 3: formatted_text2 = formatted_text2.replace(chunk, replace_text)
          unformatted_text = unformatted_text[:clean_position+1] + unformatted_text2
          formatted_text = formatted_text[:formatted_text_position+1] + formatted_text2
      
      is_dup_within_doc[hashcode] = is_dup_within_doc.get(hashcode, []) + [ch_idx]
        
      if hashcode in dup_span:
        dup_span[hashcode] += 1
      else:
        dup_span[hashcode] = 1
        
    if not keep_first_dup_in_formatted_text:      
      for hashcode, ch_idx in is_dup_within_doc.items():  
        if hashcode in dup_span and dup_span.get(hashcode, len(ch_idx)) > len(ch_idx): #this item is a duplicate across documents
          ch_idx = ch_idx[0]
          shingle= " ".join(chunks[ch_idx : ch_idx + shingle_size])
          if shingle in formatted_text: 
            formatted_text = formatted_text.replace(shingle, replace_text)
          else:
            for chunk in chunks[ch_idx : ch_idx + shingle_size]:
                formatted_text = formatted_text.replace(chunk, replace_text)
    
    if not keep_first_dup_in_unformatted_text:      
      for hashcode, ch_idx in is_dup_within_doc.items():  
        if hashcode in dup_span and dup_span.get(hashcode,0) > len(ch_idx): #this item is a duplicate across documents
          ch_idx = ch_idx[0]
          shingle= " ".join(chunks[ch_idx : ch_idx + shingle_size])
          if shingle in unformatted_text: 
            unformatted_text = unformatted_text.replace(shingle, replace_text)
          else:
            for chunk in chunks[ch_idx : ch_idx + shingle_size]:
                unformatted_text = unformatted_text.replace(chunk, replace_text)
    
    unformatted_text = unformatted_text.replace(replace_char+" .", replace_text).\
        replace(replace_char+" !", replace_text).\
        replace(replace_char+" ?", replace_text).\
        replace(replace_char+" .", replace_text).\
        replace(replace_char+" ．", replace_text).\
        replace(replace_char+" 。", replace_text).\
        replace(replace_char+" ？", replace_text).\
        replace("  ", " ").\
        replace(' '+replace_char+' '+replace_char, " "+replace_char).\
        replace(' '+replace_char+' '+replace_char, " "+replace_char).\
        replace(' '+replace_char+' '+replace_char, " "+replace_char)

    unformatted_text = " ".join(unformatted_text.split())



    formatted_text = formatted_text.replace(replace_char+" .", replace_text).\
        replace(replace_char+" !", replace_text).\
        replace(replace_char+" ?", replace_text).\
        replace(replace_char+" .", replace_text).\
        replace(replace_char+" ．", replace_text).\
        replace(replace_char+" 。", replace_text).\
        replace(replace_char+" ？", replace_text).\
        replace("  ", " ").\
        replace(' '+replace_char+' '+replace_char, " "+replace_char).\
        replace(' '+replace_char+' '+replace_char, " "+replace_char)
    formatted_text = " ".join(formatted_text.split())


    #TODO: improve this so we cleaup by value until we reach the limit
    if len(dup_span) > cleanup_dup_span_limit:
      for key, val in list(dup_span.items()):
        if val <= 1: del dup_span[key]
          
    if len(dup_doc) > cleanup_dup_doc_limit:
      for key, val in list(dup_doc.items()):
        if val <= 1: del dup_doc[key]
          
    doc_is_dup = 0
    if any([a for h, a in is_dup_within_doc.items() if len(a) > 1 or len(a) < dup_span.get(h,len(a))]):
      hashcode = " ".join(unformatted_text.replace("*", "").split())
      hashcode = hashcode.strip(' '+replace_char).lower()
      hashcode = DIGIT_REGEX.sub('1', hashcode)
      hashcode = hashing(hashcode)
      if hashcode in dup_doc:
        dup_doc[hashcode] += 1
        doc_is_dup=2
      else:
        dup_doc[hashcode] = 1
        doc_is_dup=1
        
    return doc_is_dup, unformatted_text, formatted_text

    
def test_simhash(): # test clustering on a medium size dataset
  num = 40000*1000
  import numpy as np
  import sys
  arr = np.random.randint( sys.maxsize, size=num, dtype=np.int64)
  arr.sort()
  arr = arr.tolist()
  find_clusters(arr, 5, 4, do_sort=False, verbose=True)
