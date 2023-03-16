import sys, os
import itertools
try:
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))         
except:
    pass
from char_manager import junk, special_char
from stopwords import stopwords as all_stopwords
from langid_manager import *
from banned_words import banned_words
from flagged_words import flagged_words
from cjk import lang_is_cjk
import langid
import logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    format='%(asctime)s : %(processName)s : %(levelname)s : %(message)s',
    level=logging.INFO)

lang_2_max_stopword_len = dict([(lang, max(s.count(" ")+1 if not lang_is_cjk(lang) else len(s) for s in arr)) for lang, arr in all_stopwords.items()])
lang_2_max_bannedword_len = dict([(lang, max(s.count(" ")+1 if not lang_is_cjk(lang) else len(s) for s in arr)) for lang, arr in banned_words.items()])
lang_2_max_flaggedword_len = dict([(lang, max(s.count(" ")+1 if not lang_is_cjk(lang) else len(s) for s in arr)) for lang, arr in flagged_words.items()])

def check_good_sentence(s, src_lang, stopwords, show_err=False, lang_groups=[], fast=True, ret_score=False, stopword_ratio_cutoff=0.06, bannedwords=None, flaggedwords=None, \
        banned_word_ratio_cutoff=0.005, flagged_word_ratio_cutoff=0.01, junk_ratio=0.16, max_word_len=2, do_langid_check=True, \
        flagged_cnts=None, banned_cnts=None, all_cnts=None, max_flagged_banned_word_len=4):
    is_cjk = lang_is_cjk(src_lang)
    
    if flagged_cnts is not None: flagged_cnt = flagged_cnts[src_lang] = flagged_cnts.get(src_lang, {})
    if banned_cnts is not None: banned_cnt = banned_cnts[src_lang] = banned_cnts.get(src_lang, {})
    if all_cnts is not None: all_cnt = all_cnts[src_lang] = all_cnts.get(src_lang, {})
    #max_flagged_banned_word_len = max(lang_2_max_flaggedword_len.get(src_lang, max_word_len), lang_2_max_bannedword_len.get(src_lang, max_word_len))
    #basic dejunk
    junk_score, flagged_score, banned_score, stopword_score = 0.0, 0.0, 0.0, 0.0
    if bannedwords is None:
      bannedwords = set(list(itertools.chain(*[list(banned_words.get(lang, [])) for lang in list(lang_groups)+['en']])))
    if flaggedwords is None:
      flaggedwords = set(list(itertools.chain(*[list(flagged_words.get(lang, [])) for lang in list(lang_groups)+['en']])))
    s = s.lower().strip()
    good_sentence = True
    if not s:
      good_sentence = False
      if ret_score: return good_sentence, junk_score, flagged_score, banned_score, stopword_score
      return good_sentence
    junk_score = len([s2 for s2 in s if s2 in junk])/len(s)
    if junk_score >= junk_ratio:
      good_sentence = False
      if fast:
        if ret_score: return good_sentence, junk_score, flagged_score, banned_score, stopword_score
        return good_sentence
    if lang_is_cjk(src_lang):
      s_arr = s
    else:
      s_arr = [s2.strip(special_char) for s2 in s.lower().split() if s2.strip(special_char)]
    len_s = len(s_arr)
    if len_s == 0:
      good_sentence = False
      if ret_score: return good_sentence, junk_score, flagged_score, banned_score, stopword_score
      return good_sentence

    #stopword check
    stop_cnt = total_cnt = 1
    if stopwords:
        word_len = lang_2_max_stopword_len.get(src_lang, max_word_len)
        len_s = len(s_arr)
        stop_cnt = 0
        total_cnt = 0
        for i in range(len_s):
          if s_arr[i] is None: continue
          for j in range(min(len_s, i+word_len), i, -1):
            word = "".join(s_arr[i:j]) if is_cjk else " ".join(s_arr[i:j])
            if word in stopwords:
              stop_cnt += 1
              s_arr[i] = "".join(s_arr[i:j]) if is_cjk else " ".join(s_arr[i:j]) 
              for k in range(i+1, j):
                s_arr[k] = None
              break
          total_cnt += 1
        stopword_score =  (stop_cnt/total_cnt) 
        if stopword_score < stopword_ratio_cutoff:
          good_sentence = False
          if fast:
            if ret_score: return good_sentence, junk_score, flagged_score, banned_score, stopword_score
            return good_sentence

    if flaggedwords:
        b_cnt = 0
        f_cnt = 0
        total_cnt = 0
        for i in range(len_s):
          if s_arr[i] is None: continue
          word_len = max_flagged_banned_word_len
          for j in range(min(len_s, i+word_len),i,-1):
            word = "".join(s_arr[i:j]) if is_cjk else " ".join(s_arr[i:j])
            is_flagged = word in flaggedwords
            is_banned = word in bannedwords
            if is_flagged or is_banned:
              if is_flagged: f_cnt += 1
              if is_banned: b_cnt += 1
              s_arr[i] =  word
              for k in range(i+1, j):
                s_arr[k] = None
          total_cnt += 1
        flagged_score = (f_cnt/total_cnt)
        banned_score = b_cnt/total_cnt
        #if flagged_score or banned_score: print ('flagged_score', flagged_score, 'banned_score', banned_score)
        if all_cnts is not None:
          for w in s_arr:
            if w: all_cnt[w] = all_cnt.get(w,0) + 1 
        if banned_score > banned_word_ratio_cutoff/2.0 and flagged_score >= banned_word_ratio_cutoff:
          if fast and flagged_cnts is not None:
            for w in s_arr:
              if w: flagged_cnt[w] = flagged_cnt.get(w,0) + 1  
          if banned_cnts is not None:
            for w in s_arr:
              if w: banned_cnt[w] = banned_cnt.get(w,0) + 1
          good_sentence = False
          if fast:
            if ret_score: return good_sentence, junk_score, flagged_score, banned_score, stopword_score
            return good_sentence
        if flagged_score > flagged_word_ratio_cutoff:
          if flagged_cnts is not None:
            for w in s_arr:
              if w: flagged_cnt[w] = flagged_cnt.get(w,0) + 1          
          good_sentence = False
          if fast:
            if ret_score: return good_sentence, junk_score, flagged_score, banned_score, stopword_score
            return good_sentence
         
    if do_langid_check:
      lang =  langid.classify(s)
      if lang:
        lang = lang[0]
      else:
        good_sentence = False
      if show_err and lang != src_lang and lang not in lang_groups:
        logger.info ((src_lang, lang))
      if not lang == src_lang:
        good_sentence = False

    if ret_score: return good_sentence, junk_score, flagged_score, banned_score, stopword_score
    return good_sentence
