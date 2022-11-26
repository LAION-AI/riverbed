

# ################################################################################
#  SPAN AND DOCUMENT INDEXER
#  The *Indexer is responsible for tranforming the text, storing the text in some datastructure for retreival by the 
#  SearchIndexer. 
#
#  This class is for indexing multiple documents, into spans which are fragments of one or more sentences (not necessarily a paragraph).
#  each span is a dict/json object. A span can specific to semantic search or bm25, and can include be indexed (span_idx) by (file name, line no, offset). 
#  The spans are also clustered. Finally the spans are serialzied into a jsonl.gz file.
#  assumes the sentences inside a document are NOT shuffeled, but documents can be shuffled. 4
# This class is used  to generate spans and index the spans. It also provides APIs for searching the spans.
#we can use the ontology for query expansion as part of the bm25 search. 

class RiverbedIndexer(IndexerMixin):

  def __init__(self, project_name, processor, span2idx, batch, retained_batch, span_lfs,  span2cluster_label, \
                start_idx = 0, embed_search_field="text", bm25_field="text", text_span_size=1000, embedder="minilm", do_ontology=True, running_features_per_label={}, \
                ner_to_generalize=(), span_level_feature_extractors=default_span_level_feature_extractors, \
                running_features_size=100, label2term_frequency=None, document_frequency=None, domain_stopwords_set=stopwords_set,\
                use_synonym_replacement=False, ):
    super().__init__()
    self.project_name = project_name
    self.idx = start_idx
    self.embed_search_field = embed_search_field
    self.bm25_field = bm25_field

  
  # the similarity models sometimes put too much weight on proper names, etc. but we might want to cluster by general concepts
  # such as change of control, regulatory actions, etc. The proper names themselves can be collapsed to one canonical form (The Person). 
  # Similarly, we want similar concepts (e.g., compound words) to cluster to one canonical form.
  # we do this by collapsing to an NER label and/or creating a synonym map from compound words to known tokens. See _create_ontology
  # and we use that data to generalize the sentence here.  
  #assumes the text has already been tokenized and ents has been sorted with longerst entity to shortest entity
  # TODO: have an option NOT to generalize the prefix. 
  def _generalize_text_and_filter_ents(self, tokenizer, tokenized_text, ents, ner_to_generalize=(), use_synonym_replacement=False):
      if not ner_to_generalize and not synonyms and not ents: return tokenized_text, ents

      #do a second tokenize if we want to do synonym replacement.
      if use_synonym_replacement:
        tokenized_text = tokenizer.tokenize(text, use_synonym_replacement=True)  
      filtered_ents = []

      #replace with placeholders
      for idx, ent in enumerate(ents):
          entity, label = ent
          if "@#@" not in text: break
          if f"@#@{idx}@#@" not in text: continue
          text = text.replace(f"@#@{idx}@#@", entity) 
      text = text.replace("_", " ")

      #see if there are multiple of the same labels 
      label_cnts = dict([(a,b) for a, b in Counter([label for entity, label in ents]).items() if b > 1])
      max_label_cnts = copy.copy(label_cnts)
      entity_2_id = {}
      for idx, ent in enumerate(ents):
          entity, label = ent
          entity_id = entity_2_id.get(entity) 
          if "@#@" not in tokenized_text: break
          if f"@#@{idx}@#@" not in tokenized_text: continue
          if entity not in entity_2_id and label in label_cnts:
            entity_id = entity_2_id[entity] = 1 + (max_label_cnts[label] - label_cnts[label])
            label_cnts[label] = label_cnts[label] - 1
          filtered_ents.append((entity, label,  text.count(f"@#@{idx}@#@")))
          if label in ner_to_generalize:   
            if label == 'ORG':
              tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Organization' + ('' if entity_id is None else f" {entity_id}"))
            elif label == 'PERSON':
              tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Person'+ ('' if entity_id is None else f" {entity_id}"))
            elif label == 'FAC':
              tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Facility'+ ('' if entity_id is None else f" {entity_id}"))
            elif label in ('GPE', 'LOC'):
              tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Location'+ ('' if entity_id is None else f" {entity_id}"))
            elif label in ('DATE', ):
              tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Date'+ ('' if entity_id is None else f" {entity_id}"))
            elif label in ('LAW', ):
              tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Law'+ ('' if entity_id is None else f" {entity_id}"))  
            elif label in ('EVENT', ):
              tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Event'+ ('' if entity_id is None else f" {entity_id}"))            
            elif label in ('MONEY', ):
              tokenized_text = tokenized_text.replace(f"@#@{idx}@#@", 'The Amount'+ ('' if entity_id is None else f" {entity_id}"))
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

      return tokenized_text, filtered_ents

  
  #consolidate multiple lines until we make up about text_span_size chars.
  #split by approximately text spans of text_span_size characters, respecting word and sentence boundaries
  def consolidate_and_split_spans(self, lines_iterator, line_position=0, curr_file_size=1, text_span_size=1000, return_tokenized=False):
    def do_one_batch(batch):
      ret = []
      # conslidate
      text = " ".join(batch)
      text  = self.tokenizer.tokenize(line, use_synonym_replacement=False) 
      len_text = len(text)
      offset = 0
      #now split
      while offset < len_text:
        max_rng  = min(len_text, offset+text_span_size+1)
        if text[max_rng-1] != ' ':
          #TODO: extend for non english periods and other punctuations
          add_punc = False
          if '. ' in text[max_rng:]:
            add_punc = "."
            max_rng = max_rng + text[max_rng:].index('. ')+1
          elif '? ' in text[max_rng:]:
            add_punc = "?"
            max_rng = max_rng + text[max_rng:].index('? ')+1
          elif '! ' in text[max_rng:]:
            add_punc = True
            max_rng = max_rng + text[max_rng:].index('! ')+1
          elif ' ' in text[max_rng:]:
            max_rng = max_rng + text[max_rng:].index(' ')
          else:
            max_rng = len_text
        orig_text = text2 = text[offset:max_rng].strip()
        if add_punc: text2 = text2 + add_punc
        position = (line_position+offset)/curr_file_size
        if not return_tokenized:
          text2 = text2.replace("_", " ").replace("  ", " ").replace("  ", " ")
        ret.append({'text': orig_text, 'embedding_text': text2, 'keywords': text2, 'position':position,})
        offset = max_rng
      return ret
    ####
    batch = []
    curr_batch_size = 0
    for line in lines_iterator:
      if curr_batch_size >=text_span_size:
        for data in do_one_batch(batch):
          yield data
        batch = []
        curr_batch_size = 0
      batch.append(line)
      curr_batch_size += len(line)
    if batch:
      for data in do_one_batch(batch):
        yield data


  #transform a doc into a span batch, breaking up doc into spans
  #all spans/leaf nodes of a cluster are stored as a triple of (name, lineno, offset)
  #extract both bm25_field and embed_search_field and inject intout output data yielded from this method
  def process(self, lines_iterator, filename=None, lineno_arr=None, start_idx=None, chunk_size=500, num_queries=20, dedup=False, curr_file_size=0, tokenizer=None, text_span_size=1000, ner_to_generalize=(), use_synonym_replacement=False, seen=None, *args, **kwargs):
    if seen is None: seen = {}      
      return dat

    
      
    data_iter = super().process(lines_iterator, filename, lineno_arr, start_idx, chunk_size, num_queries, extract_callback, *args, **kwargs)
    for data in data_iterator:
        text =  data['embedding_text']
        bm25_content =  data['keywords']
        curr_ents = data['ents']
        for idx, ent in enumerate(curr_ents):
          text = text.replace(ent[0], f' @#@{idx}@#@ ')
        # we do placeholder replacement tokenizing to make ngram tokens underlined, so that we don't split a span in the middle of an ner token or ngram.
        tokenized_text, ents2 = _generalize_text_and_filter_ents(self.tokenizer, text, curr_ents, ner_to_generalize, use_synonym_replacement=use_synonym_replacement) 
        if prefix and offset > 0:
            _, text2 = text2.split(" || ... ", 1)
        span['tokenized_text'] = tokenized_text 
        span['ents'] = ents2
        yield span
    running_features_per_label = {}
        

  

  def tokenize(self, *args, **kwargs):
    return self.tokenizer.tokenize(*args, **kwargs)

  # the main method for processing documents and their spans. 
  #THIS CODE NEEDS TO BE REWRITTEN TO USE THE NEW FRAMEWORK: should call self.searcher(indexer=self)
  def index(self, content_store_objs):
    global clip_model, minilm_model, labse_model
    model = self.model
    tokenizer = self.tokenizer
    searcher = self.searcher.switch_search_context(f"{project_name}.jsonl")
    os.system(f"mkdir -p {project_name}.jsonl_idx")
    span2idx = self.span2idx = OrderedDict() if not hasattr(self, 'span2idx') else self.span2idx
    span_clusters = self.span_clusters = {} if not hasattr(self, 'span_clusters') else self.span_clusters
    label2term_frequency = self.label2term_frequency = {} if not hasattr(self, 'label2term_frequency') else self.label2term_frequency
    document_frequency = self.document_frequency = {} if not hasattr(self, 'document_frequency') else self.document_frequency
    span2cluster_label = self.span2cluster_label = {} if not hasattr(self, 'span2cluster_label') else self.span2cluster_label
    if (not hasattr(model, 'kenlm_model') or model.kenlm_model is not None) and auto_create_tokenizer_and_model:
      tokenizer, model = self.tokenizer, self.model = RiverbedModel.create_tokenizer_and_model(project_name, files, )
    kenlm_model = self.model.kenlm_model 
    use_model(embedder)
    name = files.pop()
    f = open(name) 
    domain_stopwords_set = set(list(stopwords_set) + list(stopwords.keys()))


    if seen is None: seen = {}
    
    for name, curr_file_size, content_store_obj in content_store_objs.items():
      # we will collapse adjacent lines that are short. 
      prior_line = ""
      batch = []
      retained_batch = []
      curr = ""
      cluster_embeddings = None
      curr_date = ""
      curr_position = 0
      next_position = 0
      curr_file_size = os.path.getsize(name)
      position = 0
      line = ""
      lineno = -1
      curr_lineno = 0
      seen={}
      for line in content_store_obj:
        line = line.strip() #TODO: read the data from the field
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
                  for key_words, tokenized_text in zip(self.processor.process_bm25_field(curr_file_size, self.tokenizer), 
                                            self.processor.process_embed_search_field(curr_file_size, {'text': curr,}, \ 
                                                                                      text_span_size=text_span_size, \
                                                                                      ner_to_generalize=ner_to_generalize, \
                                                                                      use_synonym_replacement=use_synonym_replacement)):
              
                    item['name'] =name
                    item['lineno'] = curr_lineno
                    item['position'] = curr_position
                    
                    batch.append(item)
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
          retained_batch, span2idx, span2cluster_label, label2term_frequency, document_frequency = self._create_span_embeds_and_span2cluster_label(project_name, curr_file_size, jsonl_file_idx, span2idx, batch, \
                                                      retained_batch, jsonl_file,  f"{batch_id_prefix}_", span_lfs,  span2cluster_label, text_span_size, \
                                                      kmeans_batch_size=kmeans_batch_size, epoch = epoch, embed_batch_size=embed_batch_size, min_prev_ids=min_prev_ids, \
                                                      max_ontology_depth=max_ontology_depth, max_top_parents=max_top_parents, do_ontology=True, embedder=embedder, \
                                                      running_features_per_label=running_features_per_label, ner_to_generalize=ner_to_generalize, span_level_feature_extractors=span_level_feature_extractors, \
                                                      running_features_size=running_features_size, label2term_frequency=label2term_frequency, document_frequency=document_frequency, domain_stopwords_set=domain_stopwords_set, \
                                                      verbose_snrokel=verbose_snrokel,  span_per_cluster=span_per_cluster, use_synonym_replacement=use_synonym_replacement, )  
          batch = []
      
      # do one last line and finish processing if there's anything left
      if curr: 
          curr = curr.replace(". .", ". ").replace("..", ".").replace(":.", ".")
          hash_id = hash(curr)
          if not dedup or (hash_id not in seen):
              curr_ents = list(itertools.chain(*[[(e.text, e.label_)] if '||' not in e.text else [(e.text.split("||")[0].strip(), e.label_), (e.text.split("||")[-1].strip(), e.label_)] for e in spacy_nlp(curr).ents]))
              curr_ents = list(set([e for e in curr_ents if e[0]]))
              curr_ents.sort(key=lambda a: len(a[0]), reverse=True)
              for item in self._create_spans_batch(curr_file_size, {'text': curr, 'ents': curr_ents, }, text_span_size=text_span_size, ner_to_generalize=ner_to_generalize, use_synonym_replacement=use_synonym_replacement)
                item['name'] =name
                item['lineno'] = curr_lineno
                item['position'] = curr_position
                batch.append(item)
          
      #do the last bactch    
      if batch: 
          batch_id_prefix += 1
          retained_batch, span2idx, span2cluster_label, label2term_frequency, document_frequency = self._create_span_embeds_and_span2cluster_label(project_name, curr_file_size, jsonl_file_idx, spanf2idx, batch, \
                                                      retained_batch, jsonl_file,  f"{batch_id_prefix}_", span_lfs,  span2cluster_label, text_span_size, \
                                                      kmeans_batch_size=kmeans_batch_size, epoch = epoch, embed_batch_size=embed_batch_size, min_prev_ids=min_prev_ids,  \
                                                      max_ontology_depth=max_ontology_depth, max_top_parents=max_top_parents, do_ontology=True, embedder=embedder,\
                                                      running_features_per_label=running_features_per_label, ner_to_generalize=ner_to_generalize, span_level_feature_extractors=span_level_feature_extractors, \
                                                      running_features_size=running_features_size, label2term_frequency=label2term_frequency, document_frequency=document_frequency, domain_stopwords_set=domain_stopwords_set, \
                                                      verbose_snrokel=verbose_snrokel)  
          batch = []
          
      
      
          
    span2idx, span_clusters, label2term_frequency, document_frequency, span2cluster_label = self.span2idx, self.span_clusters, self.label2term_frequency, self.document_frequency, self.span2cluster_label                    
    self.searcher = self.searcher.switch_search_context(self.project_name, data_iterator=data_iterator, search_field="tokenized_text", bm25_field="text", embedder=embedder, \
                             auto_embed_text=True, auto_create_bm25_idx=True, auto_create_embeddings_idx=True)
    
    #TODO: cleanup label2term_frequency, document_frequency
    return self

  def gzip_jsonl_file(self):
    if not os.path.exists(f"{self.project_name}/spans.jsonl"):
      os.system(f"gzip {self.project_name}/spans.jsonl")
    GzipFileByLineIdx(f"{self.project_name}/spans.jsonl.gz")
    
  def save_pretrained(self, project_name=None):
      if project_name is None: project_name = self.project_name
      os.system(f"mkdir -p {project_name}")
      pickle.dump(self, open(f"{project_name}/indexer.pickle", "wb"))
    
  @staticmethod
  def from_pretrained(project_name):
      self = pickle.load(open(f"{project_name}/indexer.pickle", "rb"))
      return self

