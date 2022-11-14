import easyocr, math
import tempfile, os
import string
from PIL import Image
punc = string.punctuation + "¿？,،、º。゜"
lst1 = ['en', 'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr', 'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl', 'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv', 'sw', 'tl', 'tr', 'uz', 'vi', ]
lst2 = ["en",'ch_sim',]
lst3 = ['en', 'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom', ]
lst4 = ["ar","fa","ur","ug","en"]
lst5 = ['en', 'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd','ava', 'dar', 'inh', 'che', 'lbe', 'lez', 'tab', 'tjk',]
lst6 = ["en", "ja"]
lst7 = ['en', 'bn', 'as', ]
lst8 = ['en', 'th', ]
lst9 = ["en",  'ch_tra', ]
lst10 = ["en", "ko"]
lst11 = ['en', 'ta', ]
lst12 = ['en', 'te', ]
lst13 = ['en', 'kn']
#not working: #'sa', 'bgc', 'mni', 
try:
  if ocr_lang2reader is not None: pass
except:
  most_recently_used_model = None
  most_recently_used_langs =  None
  ocr_lang2reader = {}
  all_ocr_readers = []
  for lst in [lst1, lst2, lst3, lst4, lst5, lst6, lst7, lst8, lst9, lst10, lst11, lst12, lst13]:
    ocr_reader = easyocr.Reader(lst) 
    all_ocr_readers.append([ocr_reader, [lst]])
    for lang in lst:
      if lang not in ocr_lang2reader:
        ocr_lang2reader[lang] = ocr_reader

trannum = str.maketrans("0123456789", "1111111111")

#TODO:we might create a lang/word detector using the first few layers of the 13 models and run them all together to estimate which ocr model is best to make the ocr decoding faster.
def do_ocr_imgs(all_images, ocr_threshold=0.70, ocr_threshold2 = None, min_len=20, ignore_mostly_numbers=True, distance_for_newline=0.7):
  global most_recently_used_model, most_recently_used_langs, punc
  if ocr_threshold2 is None: ocr_threshold2 = ocr_threshold/2
  used_most_recent_model = False
  text = ""
  ret = []
  for image_id in all_images:
    data = []
    max_x = 0
    max_y = 0
    temp_name = None
    if image_id.endswith(".bmp") or image_id.endswith(".webp") :
      temp_name = tempfile._get_default_tempdir() + "/" + next(tempfile._get_candidate_names()) + ".png"
      Image.open(image_id).save(temp_name)
      image_id = temp_name
    best_text2 = None
    best_ret = None
    best_score = 0.0
    for reader, langs in [(most_recently_used_model, most_recently_used_langs)]+ [a for a in all_ocr_readers if a[0] != most_recently_used_model]:
      if reader is None: 
        continue
      text2=""
      result = reader.readtext(image_id)
      found_ocr = 0
      found = 0
      for coord, text3, prob in result:
        max_x = max(max_x, max([a[0] for a in coord]))
        max_y = max(max_y, max([a[1] for a in coord]))
        
        if prob > ocr_threshold2:
          found_ocr += prob
          found += 1
          text2 = text2 + " " + str(text3) 
          if text2[-1] not in punc and data:
             dst = math.sqrt(((data[-1][0][0][0] - coord[0][0])/max_x)**2 + ((data[-1][0][0][1] - coord[0][1])/max_y)**2)
             if dst > distance_for_newline: text2 = text2 + "."
          data.append((coord, text3, prob))
      text2 = text2.strip()
      
      if ignore_mostly_numbers:
        text4 = text2.translate(trannum)
        if text4.count("1") > len(text2)/3: 
          #print ("mostly numbers")
          continue
      if found > 0 and found_ocr/found >= ocr_threshold and  len(text2) >= min_len: 
        most_recently_used_model = reader
        most_recently_used_langs = langs
        text = text + " " + text2 
        if text2[-1] not in punc: text = text + "."
        best_text2 = text2
        best_ret = (data, found_ocr/found , langs)
        break
      
      if found > 0 and best_score < found_ocr/found and  len(text2) >= min_len:
        best_text2 = text2
        best_ret = (data, found_ocr/found , langs)
        best_score = found_ocr/found 
         
    if best_text2: 
      if best_text2[-1] not in punc: best_text2 = best_text2 + "."
      text = text + " " + best_text2
      ret.append(best_ret)
    if temp_name is not None: os.system(f"rm {temp_name}")
  #if we found nothing that matched the thresholds, we might want to return the highest scoring
  return text.strip(), ret
