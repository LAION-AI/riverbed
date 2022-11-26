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
from ..utils import *
from .searcher_indexer import *

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
# the model stores a copy of synonyms and the kenlm model and a searcher to search token embeddings/ontology.
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

#################################################################################
#ANALYZER MODEL CODE
#this class is used to label data, create prediction models,and analyze the words including compound words in a corpus.
#it can be used to generate text controlled by the kenlm perplixty and retrieved 
#embeddings (might need to put generation in a different model)
# Adds new features and trains labeling models
class RiverbedSpanAnalyzerModel(nn.Module):

  
 def __init__(self, basic_analyzer, project_name, searcher_indexer, span_indexer, embedder):
   super().__init__()
   global labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set 
   self.basic_analyzer, self.project_name, self.searcher_indexer, self.processor = basic_analyzer, project_name, searcher_indexer, processor
   labse_tokenizer, labse_model,  clip_processor, minilm_tokenizer, clip_model, minilm_model, spacy_nlp, stopwords_set = init_models()
   self.span_searcher = SearcherIndexer(idx_dir=project_name+"/span_searcher", embedder=embedder, indexer=span_indexer)
   self.indexer = span_indexer
                 
  # code more informative labels for the span clusters
  def _create_informative_parent_label_from_tfidf(self, clusters, span2idx, span2data, span2cluster_label, span_label2user_label=None, \
                                            label2term_frequency=None, document_frequency=None, domain_stopwords_set=stopwords_set, max_levels=4):
      if label2term_frequency is None: label2term_frequency = {}
      if document_frequency is None: document_frequency = {}
      if span_label2user_label is None: span_label2user_label = {}
      #we gather info for term_frequency-inverse_document_frequency with respect to each token in each clusters
      for label, values in clusters.items(): 
        if label[0] == 0 and label not in span_label2user_label:
          for item in values:
            if span in span2idx:
              data = span2data[span]
              text = data['tokenized_text']
              #we don't want the artificial labels to skew the tfidf calculations
              #assumes we don't have more than 10 of the same label
              text = text.replace('The Organization','').replace('The_Organization','')
              text = text.replace('The Person','').replace('The_Person','')
              text = text.replace('The Facility','').replace('The_Facility','')
              text = text.replace('The Location','').replace('The_Location','')          
              text = text.replace('The Date','').replace('The_Date','')
              text = text.replace('The Law','').replace('The_Law','')
              text = text.replace('The Amount','').replace('The_Amount','')
              text = text.replace('The Event','').replace('The_Event','')

              #we add back the entities we had replaced with the artificial labels into the term_frequency-inverse_document_frequency calculations
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
              aHash = label2term_frequency[label] =  label2term_frequency.get(label, {})
              for token, cnt in cnts.items():
                aHash[token] = cnt/len_text
              for token in cnts.keys():
                document_frequency[token] = document_frequency.get(token,0) + 1

      #Now, acually create the new label from the tfidf of the tokens in this cluster
      #TODO, see how we might save away the tfidf or get the tfidf from the bm25 indexer. 
      for label, term_frequency in label2term_frequency.items():
          tfidf = copy.copy(term_frequency)    
          for token in list(tfidf.keys()):
            tfidf[token]  = tfidf[token] * min(1.5, tokenizer.token2weight.get(token, 1)) * math.log(1.0/(1+document_frequency[token]))
          top_tokens2 = [a[0].lower().strip("~!@#$%^&*()<>,.:;")  for a in Counter(tfidf).most_common(min(len(tfidf), 40))]
          top_tokens2 = [a for a in top_tokens2 if a not in domain_stopwords_set and ("_" not in a or (a.count("_")+1 != len([b for b in a.split("_") if  b in domain_stopwords_set])))]
          top_tokens = []
          for t in top_tokens2:
            if t not in top_tokens:
              top_tokens.append(t)
          if top_tokens:
            if len(top_tokens) > 5: top_tokens = top_tokens[:5]
            new_label = ", ".join(top_tokens) 
            span_label2user_label[label] = new_label

      #create parent labels
      for old_label, new_label in span_label2user_label.items():
        for parent_old_label in [(level, old_label[1]) for level in range(1, max_levels)]:
          if parent_old_label clusters:
            span_label2user_label[parent_old_label]= ("¶"*parent_old_label[0])+new_label

      return span_label2user_label, label2term_frequency, document_frequency
                                    
  def search_and_label(self, positive_query_set, negative_query_set):
    pass
  
  #returns a model - could be any transformer w/ classification head or scikitlearn supervised learning system
  #labeled content datastore is any dict iterator (or df iterator?). the predicted fields is the field to predict. 
  #e.g., Is the span about high risk, medium risk or low risk?
  def fit(self, labled_content_data_store, predicted_fields):
    pass
  
  #returns a tag
  def predict(self, model, example):
    pass
  
  #label all entries in the jsonl spans and emits a labeled_file_name.jsonl.gz file. Fills in the prediction in the label_field.
  #will not label the iterm if the confidence score is below filter_below_confidence_score
  def label_all_data_in_project(self, model, label_file_name, label_field, filter_below_confidence_score=0.0):
    pass

    
  def save_pretrained(self, project_name=None):
      os.system(f"mkdir -p {project_name}")
      self.span_indexer.gzip_jsonl_file()
      if project_name is not None and self.project_name != project_name:
        os.system(f"cp -rf {self.project_name} {project_name}")  
      word_searcher = self.word_searcher
      self.word_searcher = None
      span_searcher = self.span_searcher
      self.span_searcher = None     
      torch.save(self, open(f"{project_name}/riverbed_analyzer.pickle", "wb"))
      word_searcher.save_pretrained(idx_dir=project_name+"/word_searcher")
      span_searcher.save_pretrained(idx_dir=project_name+"/span_searcher")                                    
      self.word_searcher = word_searcher
      self.span_searcher = span_searcher
                                    
  @staticmethod
  def from_pretrained(project_name):
      init_models()
      self = torch.load(open(f"{project_name}/riverbed_analyzer.pickle", "rb"))
      self.word_searcher = SearchIndexer.from_pretrained(idx_dir=project_name+"/word_searcher")
      self.span_searcher = SearchIndexer.from_pretrained(idx_dir=project_name+"/span_searcher")   
      return self
                  


