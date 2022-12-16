#translation based stuff 
import os
try:
  import transformers
except:
  os.system("pip install transformers  sentencepiece")
  
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, BertModel, BertTokenizerFast


import torch
from torch.nn.functional import cosine_similarity
from .filtering import *
import string
punc = string.punctuation + "¿？,،、º。゜ "

#We assume we are running on CPU only
#use labse to do a comparison
try:
  if m2m100_model is None: pass
except:
  #m2m100_model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100").half().eval().to('cuda')  
  #m2m100_tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100")
  m2m100_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").half().eval().to('cuda')  
  m2m100_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
  labse_model = BertModel.from_pretrained("sentence-transformers/LaBSE").half().eval().to('cuda')  
  labse_tokenizer = BertTokenizerFast.from_pretrained("sentence-transformers/LaBSE")
  
#TODO - add some multitrans, backtrans to get more diversity
#TODO - add option to return as paragraph by lang. 
def get_translation_set(text, threshold=0.75, langs = ["af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br", "bs", "ca", "ceb", \
                                         "cs", "cy", "da", "de", "el", "en", "es", "et", "fa", "ff", "fi", "fr", "fy", \
                                         "ga", "gd", "gl", "gu", "ha", "he", "hi", "hr", "ht", "hu", "hy", "id", "ig", \
                                         "ilo", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "lb", "lg", "ln", \
                                         "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr", "ms", "my", "ne", "nl", "no", \
                                         "ns", "oc", "or", "pa", "pl", "ps", "pt", "ro", "ru", "sd", "si", "sk", "sl", "so", \
                                         "sq", "sr", "ss", "su", "sv", "sw", "ta", "th", "tl", "tn", "tr", "uk", "ur", "uz", "vi", \
                                         "wo", "xh", "yi", "yo", "zh", "zu"], multi_trans=True):
  ret = []
  if type(text) is str: text = [text]
  else:
    text = list(text)
  with torch.no_grad():
    labse_text = labse_tokenizer(text, padding=True, return_tensors="pt").to('cuda')  
    en_embed = labse_model(**labse_text).pooler_output
      
    for target_lang in langs:

      input = m2m100_tokenizer(text, padding=True, return_tensors="pt").to('cuda')
      generated_tokens = m2m100_model.generate(**input, forced_bos_token_id=m2m100_tokenizer.get_lang_id(target_lang))
      trans_text = m2m100_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
      labse_text = labse_tokenizer(trans_text, padding=True, return_tensors="pt").to('cuda')
      
      all_trans_embed = labse_model(**labse_text).pooler_output
      similarity = cosine_similarity(en_embed,all_trans_embed, dim=1)
      #trs = []
      for sim, tr in zip(similarity,  trans_text):
        #print (sim, tr)
        if sim >= threshold and not high_ngram(tr):
          ret.append(tr)
          #trs.append(tr)
      #if add_backtrans:

    return set(text+ret) # [a.lower() for a in ret]) 
