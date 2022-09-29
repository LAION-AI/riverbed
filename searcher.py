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

def _get_content_from_line(l, search_field="text"):
    l =l.decode().replace("\\n", "\n").replace("\\t", "\t").strip()
    if not l: return ''
    if l[0] == "{" and l[-1] == "}":
      content = l.split(search_field+'": "')[1]
      content = content.split('", "')[0].replace("_", " ")
    else:
      content = l.replace("_", " ")
    return content

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
# and we use that data to simplify the sentence here.  
# TODO: have an option NOT to simplify the prefix. 
def _simplify_text(self, text, tokenizer, ents, ner_to_simplify=(), use_synonym_replacement=False):
    if not ner_to_simplify and not synonyms and not ents: return text, ents
    # assumes the text has already been tokenized and replacing NER with @#@{idx}@#@ 
    tokenized_text = text
    #do a second tokenize if we want to do synonym replacement.
    if use_synonym_replacement:
      tokenized_text = tokenizer.tokenize(text, use_synonym_replacement=True)  
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
  

def _create_informative_parent_label_from_tfidf(clusters, span2idx, tmp_span2batch, span2cluster_label, \
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
  
class RiverbedPreprocessor:
       pass
  
#################################################################################
# SPAN AND DOCUMENT PROCESSOR
# includes labeling of spans of text with different features, including clustering
# assumes each batch is NOT shuffeled.    
class SpansPeprocessor(RiverbedPreprocessor):
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
  #tf-idf weight
  
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
  
  def __init__(self, project_name, *args, **kwargs):
    super().__init__()
    
    self.searcher = SearcherIdx(project_name, *args, **kwargs)
  
  def __init__(self, project_name, start_idx = 0, search_field="text",  curr_file_size, jsonl_file_idx, span2idx, batch, retained_batch, \
                jsonl_file, batch_id_prefix, span_lfs,  span2cluster_label, \
                text_span_size=1000, kmeans_batch_size=50000, epoch = 10, \
                embed_batch_size=7000, min_prev_ids=10000, embedder="minilm", \
                max_ontology_depth=4, max_top_parents=10000, do_ontology=True, \
                running_features_per_label={}, ner_to_simplify=(), span_level_feature_extractors=default_span_level_feature_extractors, \
                running_features_size=100, label2tf=None, df=None, domain_stopwords_set=stopwords_set,\
                verbose_snrokel=False,  span_per_cluster=10, use_synonym_replacement=False, ):
    self.idx = start_idx
    self.search_field = search_field

  # gets in a lines iterator and outputs subsequent dict iterator
  def process(self, lines_iterator, *kwargs):
    for line in lines_iterator:
      l =  _get_content_from_line(line, self.search_field)
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
        yield {'idx': self.idx, 'offset': offset, 'text': l2}
      self.idx += 1
      
  def tokenize(self, *args, **kwargs):
    return self.tokenizer.tokenize(*args, **kwargs)

  
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
   # pass

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
  #     - cosine distance to nearest neighbor cluster head

  # similar to _create_token_embeds_and_synonyms, except for spans     
  #(1) compute features and embeddings in one batch for tokenized text.
  #(2) create clusters in an incremental fashion from batch
  #all leaf nodes are spans
  #spanf2idx is a mapping from the span to the actual underlying storage idx (e.g., a jsonl file or database)
  #span2cluster_label is like the synonym data-structure for tokens.
  def index(self):
    
  def _create_span_embeds_and_span2cluster_label(self):
    
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
    
    # all labeling and feature extraction is complete, and the batch has all the info. now save away the batch
    for b in batch:
      if b['idx'] >= start_idx_for_curr_batch:
        jsonl_file.write(json.dumps(b)+"\n")
        #TODO, replace with a datastore abstraction, such as sqlite
    
    # add stuff to the retained batches
                   
    return retained_batch, span2idx, span2cluster_label, label2tf, df   

  # the main method for processing documents and their spans. 
  def index(self):
    global clip_model, minilm_model, labse_model
    model = self.model
    tokenizer = self.tokenizer
    searcher = self.searcher = SearcherIdx(f"{project_name}.jsonl")
    os.system(f"mkdir -p {project_name}.jsonl_idx")
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
    
    with open(f"{project_name}.jsonl", "w", encoding="utf8") as jsonl_file:
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
          
    span2idx, span_clusters, label2tf, df, span2cluster_label, label_models = self.span2idx, self.span_clusters, self.label2tf, self.df, self.span2cluster_label, self.label_models                    
    self.searcher = Searcher(f"{project_name}.jsonl", search_field="tokenized_text", bm25_field="text", embedder="minilm", \
                             auto_embed_text=True, auto_create_bm25_idx=True, auto_create_embeddings_idx=True)
    return self

  def gzip_jsonl_file(self):
    os.system(f"gzip {project_name}/spans.jsonl")
    GzipFileByLine(f"{project_name}/spans.jsonl")
    
  def save_pretrained(self, project_name):
      os.system(f"mkdir -p {project_name}")
      pickle.dump(self, open(f"{project_name}/{project_name}.pickle", "wb"))
    
  @staticmethod
  def from_pretrained(project_name):
      self = pickle.load(open(f"{project_name}/{project_name}.pickle", "rb"))
      return self

    
class BasicLinePrepocessor(RiverbedPreprocessor):
  def __init__(self, start_idx = 0, search_field="text"):
    self.idx = start_idx
    self.search_field = search_field

  # gets in a lines iterator and outputs subsequent dict iterator
  def process(self, lines_iterator, *kwargs):
    for line in lines_iterator:
      l =  _get_content_from_line(line, self.search_field)
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
        yield {'idx': self.idx, 'offset': offset, 'text': l2}
      self.idx += 1

#TODO. Change this to inherit from a transformers.PretrainedModel.
class Searcher(nn.Module):
  
  def __init__(self,  filename, fobj=None, mmap_file=None, mmap_len=0, embed_dim=25, dtype=np.float16, \
               parents=None, parent_levels=None, parent_labels=None, skip_idxs=None, \
               parent2idx=None, top_parents=None, top_parent_idxs=None, clusters=None,  embedder="minilm", chunk_size=1000, \
               search_field="text", bm25_field=None, filebyline=None, downsampler=None, auto_embed_text=False, \
               auto_create_embeddings_idx=False, auto_create_bm25_idx=False,  \
               span2cluster_label=None, idxs=None, max_level=4, max_cluster_size=200, \
               min_overlap_merge_cluster=2, prefered_leaf_node_size=None, kmeans_batch_size=250000, \
               universal_embed_mode = None, prototype_sentences=None,  prototypes=None, universal_downsampler =None, min_num_prorotypes=50000, \
               use_tqdm=True, search_field_preprocessor=None, bm25_field_preprocessor=None
              ):
    #TODO, add a vector_preprocessor. Given a batch of sentences, and an embedding, create additional embeddings corresponding to the batch. 
    """
        Cluster indexes and performs approximate nearest neighbor search on a memmap file. 
        Also provides a wrapper for Whoosh BM25.
        :arg filename:        The name of the file that is to be indexed and searched. 
                              Can be a txt, jsonl or gzip of the foregoing. 
        :arg fobj:            Optional. The file object 
        :arg  mmap_file:      Optional, must be passed as a keyword argument.
                                This is the file name for the vectors representing 
                                each line in the gzip file. Used for embeddings search.
        :arg mmap_len         Optional, must be passed as a keyword argument if mmap_file is passed.
                                This is the shape of the mmap_file.     
        :arg embed_dim        Optional, must be passed as a keyword argument if mmap_file is passed.
                                This is the shape of the mmap_file.                       
        :arg dtype            Optional, must be passed as a keyword argument.
                                This is the dtype of the mmap_file.                              
        :arg  parents:        Optional, must be passed as a keyword argument.
                                This is a numpy or pytorch vector of all the parents of the clusters]
                                Where level 4 parents are the top level parents. 
                                This structure is used for approximage nearest neighbor search.
        :arg parent2idx:      Optional, must be passed as a keyword argument. If parents
                                are passed, this param must be also passed. 
                                It is a dict that maps the parent tuple to the index into the parents tensor
        :arg top_parents:     Optional. The list of tuples representing the top parents.
        :arg top_parents_idxs: Optional. The index into the parents vector for the top_parents.  
        :arg clusters:         Optional. A dictionary representing parent label -> [child indexes]
        :arg auto_create_embeddings_idx. Optional. Will create a cluster index from the contents of the mmap file. 
                                Assumes the mmap_file is populated.
        :arg auto_embed_text. Optional. Will populate the mmap_file from the data from filename/fobj. 
        :arg auto_create_bm25_idx: Optional. Will do BM25 indexing of the contents of the file using whoosh, with stemming.
        :arg filebyline           Optional. The access for a file by lines.
        :arg search_field:      Optional. Defaults to "text". If the data is in jsonl format,
                              this is the field that is Whoosh/bm25 indexed.
        :arg bm25_field:        Optional. Can be different than the search_field. If none, then will be set to the search_field.
        :arg idxs:                Optional. Only these idxs should be indexed and searched.
        :arg skip_idxs:           Optional. The indexes that are empty and should not be searched or clustered.
        :arg filebyline:           Optional. If not passed, will be created. Used to random access the file by line number.
        :arg downsampler:          Optional. The pythorch downsampler for mapping the output of the embedder to a lower dimension.
        :arg universal_embed_mode:  Optional. Either None, "assigned", "random", or "clusters". If we should do universal embedding as described below, this will control
                                    how the prototypes are assigned. 
        :arg prototype_sentences:     Optional. A sorted list of sentences that represents the protoypes for embeddings space. If universal_embed_mode is set and prototypes
                                  are not provided,then this will be the level 0 parents sentences of the current clustering.
                                  To get universal embedding, we do cosine(target, prototypes_vec), then normalize and then run through a universial_downsampler
        :arg protoypes:         Optional. The vectors in the embeddeing or (downsampled embedding) space that corresponds to the prototype_sentences.
        :arg min_num_prorotypes Optional. Will control the number of prototypes.
        :arg universal_downsampler Optional. The pythorch downsampler for mapping the output described above to a lower dimension that works across embedders
                                  and concept drift in the same embedder. maps from # of prototypes -> embed_dim. 
        :arg search_field_preprocessor:    Optional. If not set, then the BasicLineProcessor will be used.
        :arg bm25_field_preprocessor:      Optional. If not set, then the BasicLineProcessor will be used.
        
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
    if search_field_preprocessor is None: search_field_preprocessor = BasicLineProcessor(search_field=search_field)
    if bm25_field_preprocessor is None: 
       if bm25_field is None:
          bm25_field_preprocessor = search_field_preprocessor
        else:
          bm25_field_preprocessor = = BasicLinePreprocessor(search_field=bm25_field_preprocessor)
    self.embedder, self.search_field_preprocessor, self.bm25_field_preprocessor = embedder, search_field_preprocessor, bm25_field_preprocessor
    assert filename is not None
    self.idx_dir = f"{filename}_idx"
    if mmap_file is None:
      mmap_file = f"{self.idx_dir}/search_index_{search_field}_{embedder}_{embed_dim}.mmap"
    if fobj is None:
      if filename.endswith(".gz"):
        fobj = self.fobj = GzipByLineIdx.open(filename)
      else:
        fobj = self.fobj = open(filename, "rb")  
    else:
      self.fobj = fobj
    if not os.path.exists(filename+"_idx"):
      os.makedirs(filename+"_idx")   
    self.filebyline = filebyline
    if self.filebyline is None: 
      if type(self.fobj) is GzipByLineIdx:
        self.filebyline = self.fobj 
      else:   
        self.filebyline = FileByLineIdx(fobj=fobj) 
    labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set = init_models()
    if downsampler is None:
      if embedder == "clip":
        model_embed_dim = clip_model.config.text_config.hidden_size
      elif embedder == "minilm":
        model_embed_dim = minilm_model.config.hidden_size
      elif embedder == "labse":
        model_embed_dim = labse_model.config.hidden_size   
      downsampler = nn.Linear(model_embed_dim, embed_dim, bias=False).eval() 
    if bm25_field is None: bm25_field = search_field
    self.universal_embed_mode, self.mmap_file, self.mmap_len, self.embed_dim, self.dtype, self.clusters, self.parent2idx,  self.parents, self.top_parents, self.top_parent_idxs, self.search_field, self.bm25_field, self.downsampler  = \
             universal_embed_mode, mmap_file, mmap_len, embed_dim, dtype, clusters, parent2idx, parents, top_parents, top_parent_idxs, search_field, bm25_field, downsampler
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
    if auto_embed_text and self.fobj is not None:
      self.embed_text(chunk_size=chunk_size, use_tqdm=use_tqdm)
    if universal_embed_mode not in (None, "assigned") and clusters is None:
      auto_create_embeddings_idx = True
    if os.path.exists(self.mmap_file) and (idxs is not None or auto_create_embeddings_idx):
      self.recreate_embeddings_idx(clusters=self.clusters, span2cluster_label=span2cluster_label, idxs=idxs, max_level=max_level, max_cluster_size=max_cluster_size, \
                               min_overlap_merge_cluster=min_overlap_merge_cluster, prefered_leaf_node_size=prefered_leaf_node_size, kmeans_batch_size=kmeans_batch_size)
    else:
      self.recreate_parents_data()
    if auto_create_bm25_idx and self.fobj:
       self.recreate_whoosh_idx(auto_create_bm25_idx=auto_create_bm25_idx, idxs=idxs, use_tqdm=use_tqdm)
    setattr(self,f'downsampler_{self.search_field}_{embedder}_{self.embed_dim}', self.downsampler)
    setattr(self,f'clusters_{self.search_field}_{self.embedder}_{self.embed_dim}', self.clusters)
    self.universal_embed_mode = universal_embed_mode
    if universal_embed_mode:
      assert (prototypes is None and prototype_sentences is None and universal_downsampler is None) or universal_embed_mode == "assigned"
      if universal_embed_mode == "random":
        prototype_sentences = [_get_content_from_line(self.filebyline[i], search_field) for i in random.sample(list(range(len(self.filebyline)), min_num_prorotypes))]
      elif universal_embed_mode == "cluster":
        level_0_parents = [span[1] for span in self.parent2idx.keys() if span[0] == 0]
        prototype_sentences = [_get_content_from_line(self.filebyline[span[1]], search_field) for span in level_0_parents]
      assert prototype_sentences
      if len(prototype_senences) > min_num_prorotypes:
         assert universal_embed_mode != "assigned"
         prototype_senences = random.sample(prototype_senences,min_num_prorotypes)
      elif len(prototype_senences) < min_num_prorotypes:
         assert universal_embed_mode != "assigned"
         prototype_sentences.extend([_get_content_from_line(self.filebyline[i], search_field) for i in random.sample(list(range(len(self.filebyline)), min_num_prorotypes-len(prorotype_senences)))])
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
      self.mmap_file = f"{self.idx_dir}/search_index_{search_field}_universal_{embed_dim}.mmap"
      if auto_embed_text and self.fobj is not None:
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
      dat = cosine_similarity(dat, prototypes)
      dat = torch.nn.functional.normalize(dat, dim=1)
      dat = self.universal_downsampler(dat)
    return dat
  
  def switch_search_context(self, downsampler = None, mmap_file=None, embedder="minilm", clusters=None, \
                            span2cluster_label=None, idxs=None, max_level=4, max_cluster_size=200, chunk_size=1000,  \
                            parent2idx=None, parents=None, top_parents=None, top_parent_idxs=None, skip_idxs=None, \
                            auto_embed_text=False,auto_create_embeddings_idx=False, auto_create_bm25_idx=False,  \
                            reuse_clusters=False, min_overlap_merge_cluster=2, prefered_leaf_node_size=None, kmeans_batch_size=250000, use_tqdm=True
                          ):
    global device
    if hasattr(self,f'downsampler_{self.search_field}_{self.embedder}_{self.embed_dim}'): getattr(self,f'downsampler_{self.search_field}_{self.embedder}_{self.embed_dim}').cpu()
    if hasattr(self, 'downsampler') and self.downsampler is not None: self.downsampler.cpu()
    fobj = self.fobj
    if self.universal_embed_mode == "clustered":
      clusters = self.clusters
    elif reuse_clusters: 
      assert clusters is None
      clusters = self.clusters
    if mmap_file is None:
      if  self.universal_embed_mode:
        mmap_file = f"{self.idx_dir}/search_index_{self.search_field}_universal_{self.embed_dim}.mmap"
        auto_embed_text=not os.path.exists(self.mmap_file) # the universal embeddings are created once. 
      else:
        mmap_file = f"{self.idx_dir}/search_index_{self.search_field}_{embedder}_{self.embed_dim}.mmap"
    if downsampler is None:
      if hasattr(self,f'downsampler_{self.search_field}_{embedder}_{self.embed_dim}'):
        downsampler = getattr(self,f'downsampler_{self.search_field}_{embedder}_{self.embed_dim}')
      else:
        if embedder == "clip":
          model_embed_dim = clip_model.config.text_config.hidden_size
        elif embedder == "minilm":
          model_embed_dim = minilm_model.config.hidden_size
        elif embedder == "labse":
          model_embed_dim = labse_model.config.hidden_size   
        downsampler = nn.Linear(model_embed_dim, self.embed_dim, bias=False).eval() 
    if clusters is None:
      if hasattr(self,f'clusters_{self.search_field}_{embedder}_{self.embed_dim}'):
        clusters = getattr(self,f'clusters_{self.search_field}_{embedder}_{self.embed_dim}')
    self.embedder, self.mmap_file, self.clusters, self.parent2idx,  self.parents, self.top_parents, self.top_parent_idxs,  self.downsampler  = \
             embedder, mmap_file, clusters, parent2idx, parents, top_parents, top_parent_idxs, downsampler
    if skip_idxs is None: skip_idxs = []
    self.skip_idxs = set(list(self.skip_idxs if hasattr(self, 'skip_idxs') and self.skip_idxs else []) + list(skip_idxs))
    if auto_embed_text and self.fobj is not None:
      self.embed_text(chunk_size=chunk_size, use_tqdm=use_tqdm)
    if os.path.exists(self.mmap_file) and (idxs is not None or auto_create_embeddings_idx):
      self.recreate_embeddings_idx(clusters=self.clusters, span2cluster_label=span2cluster_label, idxs=idxs, max_level=max_level, max_cluster_size=max_cluster_size, \
                               min_overlap_merge_cluster=min_overlap_merge_cluster, prefered_leaf_node_size=prefered_leaf_node_size, kmeans_batch_size=kmeans_batch_size)
    else:
      self.recreate_parents_data()
    if auto_create_bm25_idx and self.fobj:
       self.recreate_whoosh_idx(auto_create_bm25_idx=auto_create_bm25_idx, idxs=idxs, use_tqdm=use_tqdm)
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
    setattr(self,f'downsampler_{self.search_field}_{embedder}_{self.embed_dim}', self.downsampler)
    setattr(self,f'clusters_{self.search_field}_{self.embedder}_{self.embed_dim}', self.clusters)
    parents = self.parents
    del self.parents
    self.register_buffer('parents', parents)
    prototypes = self.prototypes
    del self.prototypes
    self.register_buffer('prototypes', prototypes)
    
  #get the sentence embedding for the sent or batch
  def get_embeddings(self, sent_or_batch):
    return get_embeddings(sent_or_batch, downsampler=self.downsampler, dtype=self.dtype, embedder=self.embedder, \
                          universal_embed_mode=self.universal_embed_mode, prototypes=self.prototypes, universal_downsampler=self.universal_downsampler)
              
  #embed all of self.fobj or (idx, content) for idx in idxs for the row/content from fobj
  def embed_text(self, start_idx=None, chunk_size=1000, idxs=None, use_tqdm=True, auto_create_bm25_idx=False, **kwargs):
    assert self.fobj is not None
    if start_idx is None: start_idx = 0
    search_field = self.search_field 
    ###
    def fobj_data_reader():
      fobj = self.fobj
      pos = fobj.tell()
      fobj.seek(0, 0)
      for l in fobj:
        yield _get_content_from_line(l, search_field)
      fobj.seek(pos,0)
    ###  
    
    if idxs is not None:
      dat_iter = [(idx, _get_content_from_line(self.filebyline[idx], search_field)) for idx in idxs]
    else:
      dat_iter = fobj_data_reader()  
    # we can feed the dat_iter = self.search_field_preprocessor(data_iter, **kwargs)
    self.mmap_len, skip_idxs =  embed_text(dat_iter, self.mmap_file, start_idx=start_idx, downsampler=self.downsampler, \
                          mmap_len=self.mmap_len, embed_dim=self.embed_dim, embedder=self.embedder, chunk_size=chunk_size, use_tqdm=use_tqdm, \
                          universal_embed_mode=self.universal_embed_mode, prototypes=self.prototypes, universal_downsampler=self.universal_downsampler)
    setattr(self,f'downsampler_{self.search_field}_{self.embedder}_{self.embed_dim}', self.downsampler)
    self.skip_idxs = set(list(self.skip_idxs)+skip_idxs)
      

  #the below is probably in-efficient
  def recreate_parents_data(self):
    global device          
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
      clusters, _ = self.cluster(clusters=clusters, span2cluster_label=span2cluster_label, cluster_idxs=idxs, max_level=max_level, max_cluster_size=max_cluster_size, \
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
  

  def cluster(self, clusters=None, span2cluster_label=None, cluster_idxs=None, max_level=4, max_cluster_size=200, \
                               min_overlap_merge_cluster=2, prefered_leaf_node_size=None, kmeans_batch_size=250000, use_tqdm=True):
    return create_hiearchical_clusters(clusters=clusters, span2cluster_label=span2cluster_label, mmap_file=self.mmap_file, mmap_len=self.mmap_len, embed_dim=self.embed_dim, dtype=self.dtype, \
                                      skip_idxs=self.skip_idxs, cluster_idxs=cluster_idxs, max_level=max_level, \
                                      max_cluster_size=max_cluster_size, min_overlap_merge_cluster=min_overlap_merge_cluster, \
                                      prefered_leaf_node_size=prefered_leaf_node_size, kmeans_batch_size=kmeans_batch_size, use_tqdm=use_tqdm)
  
  
  
  def recreate_whoosh_idx(self, auto_create_bm25_idx=False, idxs=None, use_tqdm=True):
    assert self.fobj is not None
    fobj = self.fobj
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
      pos = fobj.tell()
      fobj.seek(0, 0)
      if idxs is not None:
        idx_text_pairs = [(idx, self.filebyline[idx]) for idx in idxs]
        if use_tqdm:
          dat_iter =  tqdm.tqdm(idx_text_pairs)
        else:
          dat_iter = idx_text_pairs
      else:
        if use_tqdm:
          dat_iter = tqdm.tqdm(enumerate(fobj))
        else:
          dat_iter = enumerate(fobj)
      for idx, l in dat_iter:
          content= _get_content_from_line(l, bm25_field)
          if not content: continue
          writer.add_document(id=str(idx), content=content)  
      writer.commit()
      fobj.seek(pos,0)

               
  def whoosh_searcher(self):
      return self.whoosh_ix.searcher()
    
  #search using vector based and/or bm25 search. returns generator of a dict obj containing the results in 'id', 'text',and 'score. 
  #if the underlying data is jsonl, then the result of the data will also be returned in the dict.
  #WARNING: we overwrite the 'id', 'score', and 'text' field, so we might want to use a different field name like f'{field_prefix}_score'
  #key_terms. TODO: See https://whoosh.readthedocs.io/en/latest/keywords.html
  def search(self, query=None, vec=None, do_bm25_only=False, k=5, chunk_size=100, limit=None):
    def _get_data(idx):
      l  = self.filebyline[idx]
      dat = l.decode().replace("\\n", "\n").replace("\\t", "\t").strip()
      if dat[0] == "{" and dat[-1] == "}":
        try:
          dat = json.loads(l)
        except:
          pass
      return dat
    
    embedder = self.embedder
    if type(query) in (np.array, torch.Tensor):
      vec = query
      query = None
    assert vec is None or self.parents is not None
    if vec is None and query is not None and hasattr(self, 'downsampler') and self.downsampler is not None:
      vec = self.get_embeddings(query)
      if not hasattr(self, 'whoosh_ix') or self.whoosh_ix is None:
        query = None
    vec_search_results = embeddings_search(vec, mmap_file= self.mmap_file, mmap_len=self.mmap_len, embed_dim=self.embed_dim,  dtype=self.dtype, \
                                  parents=self.parents, clusters=self.clusters, top_parent_idxs=self.top_parent_idxs, \
                                  top_parents=self.top_parents, parent2idx=self.parent2idx, k=k)
    if limit is None: 
      cnt = 10^6
    else:
      cnt = limit
    if query is not None:        
      assert hasattr(self, 'whoosh_ix'), "must be created with bm25 indexing"
      with self.whoosh_searcher() as searcher:
        if type(query) is str:
           query = QueryParser("content", self.whoosh_ix.schema).parse(query)
        results = searcher.search(query, limit=limit)
        if vec is None or do_bm25_only:
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
                vec_results = {}
                for _, r in zip(range(chunk_size), vec_search_results):
                  vec_results[r[0]] = ([], r[1])
                idxs = [idx for idx in idxs if idx not in vec_results]
                vecs = torch.from_numpy(np_memmap(self.mmap_file, shape=[self.mmap_len, self.embed_dim], dtype=self.dtype)[idxs]).to(device)
                results = cosine_similarity(vec, vecs)
                for idx, score, key_term in zip(idxs, results, key_terms):
                   vec_results[idx] = (key_term, score.item())
                vec_results = list(vec_results.items())
                vec_results.sort(key=lambda a: a[1][1], reverse=True)
                for idx, score_keyterm in vec_results:
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
            vec_results = {}
            for _, r in zip(range(chunk_size), vec_search_results):
              vec_results[r[0]] =  ([], r[1])
            idxs = [idx for idx in idxs if idx not in vec_results]
            vecs = torch.from_numpy(np_memmap(self.mmap_file, shape=[self.mmap_len, self.embed_dim], dtype=self.dtype)[idxs]).to(device)
            results = cosine_similarity(vec, vecs)
            for idx, score, key_term in zip(idxs, results, key_terms):
               vec_results[idx] = (key_term, score.item())
            vec_results = list(vec_results.items())
            vec_results.sort(key=lambda a: a[1][1], reverse=True)
            for idx, score_keyterm in vec_results:
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
    for r in vec_search_results:
       data = _get_data(r[0])
       if type(data) is dict:
         data['id'] = r[0]
         data['score'] = r[1]
         yield data
       else:
         yield {'id': r[0], 'text': data, 'score': r[1]}
       cnt -= 1
       if cnt <= 0: return
    
        
  def save_pretrained(self, filename):
      os.system(f"mkdir -p {filename}_idx")
      fobj = self.fobj
      mmap_file = self.mmap_file
      idx_dir = self.idx_dir 
      self.idx_dir = None
      if self.mmap_file.startswith(idx_dir):
        self.mmap_file = self.mmap_file.split("/")[-1]
      if hasattr(self, 'filebyline') and self.filebyline is not None:
        if type(self.filebyline) is GzipByLineIdx:
          self.filebyline = None
        else:
          self.filebyline.fobj = None
      self.fobj = None
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
      pickle.dump(self, open(f"{filename}_idx/search_index.pickle", "wb"))
      self.mmap_file = mmap_file
      self.idx_dir = idx_dir
      self.fobj = fobj
      if self.downsampler is not None:
        self.downsampler.to(device2)
      self.parents = parents
      if type(self.fobj) is GzipByLineIdx:
        self.filebyline = self.fobj
      else:
        if hasattr(self, 'filebyline') and self.filebyline is not None: self.filebyline.fobj = self.fobj

  @staticmethod
  def from_pretrained(filename):
      global device
      idx_dir = f"{filename}_idx"
      self = pickle.load(open(f"{idx_dir}/search_index.pickle", "rb"))
      self.idx_dir = idx_dir
      if os.path.exists(f"{idx_dir}/{self.mmap_file}"):
        self.mmap_file = f"{idx_dir}/{self.mmap_file}"
      if filename.endswith(".gz"):
        self.filebyline = self.fobj = GzipByLineIdx.open(filename)
      else:
        self.fobj = open(filename, "rb")
        if hasattr(self, 'filebyline') and self.filebyline is not None: self.filebyline.fobj = self.fobj
      self.downsampler.eval().to(device)
      self.recreate_parents_data()
      if self.prototype_sentences: 
        self.prototypes = self.get_embeddings(self.prototype_sentences)
      parents = self.parents
      del self.parents
      self.register_buffer('parents', parents)
      prototypes = self.prototypes
      del self.prototypes
      self.register_buffer('prototypes', prototypes)
    
      return self
        
