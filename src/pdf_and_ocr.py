import easyocr, math
import tempfile, os
import string
import pdfplumber
from pdfminer.layout import LTTextContainer, LTTextBox, LTFigure, LTTextLine, LTImage
from pdfminer.high_level import extract_pages
from pdfminer.image import ImageWriter
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from PIL import Image

punc_and_space = string.punctuation + "¿？,،、º。゜ "
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
# maybe do two passes, one with all_ocr_readers, then do lang detect and then one with reader for the particular predominant lang(s)
def do_ocr_imgs(all_images, ocr_threshold=0.70, ocr_threshold2 = None, min_len=20, ignore_mostly_numbers=True, distance_for_newline=0.7):
  global most_recently_used_model, most_recently_used_langs, punc_and_space
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
          if text2[-1] not in punc_and_space and data:
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
        if text2[-1] not in punc_and_space: text = text + "."
        best_text2 = text2
        best_ret = (data, found_ocr/found , langs)
        break
      
      if found > 0 and best_score < found_ocr/found and  len(text2) >= min_len:
        best_text2 = text2
        best_ret = (data, found_ocr/found , langs)
        best_score = found_ocr/found 
         
    if best_text2: 
      if best_text2[-1] not in punc_and_space: best_text2 = best_text2 + "."
      text = text + " " + best_text2
      ret.append(best_ret)
    if temp_name is not None: os.system(f"rm {temp_name}")
  #if we found nothing that matched the thresholds, we might want to return the highest scoring
  return text.strip(), ret

#extract 
def extract_from_pdf(data, image_store_path, img_template_tag="###img###%s###", metadata_start_tag="###meta###0####", metadata_end_tag="###/meta###", title_tag="###title###0###"):
  temp_name = tempfile._get_default_tempdir() + "/" + next(tempfile._get_candidate_names()) + ".pdf"
  
  with open(
            temp_name,
            "wb",
          ) as fre:
            fre.write(data) 
            fre.cl
  text = ""
  meta = ""
  title = ""
  try:
      fp = open(temp_name, 'rb')
      parser = PDFParser(fp)
      if doc.info:
        meta = ""
        for c in doc.info:
          for a, b in c.items():
            try: 
              val = b.decode("utf8") 
            except:
              continue
            if a.strip().lower() == "title": title = val
            meta += " | "+a+": "+val
        meta = meta.strip(" |")
        if meta: meta = text = metadata_start_tag+" "+ meta + " "+metadata_end_tag
      fp.close()
      fp = parser = doc = None 
  except:
      if temp_name is not None: os.system(f"rm {temp_name}")
      return None                    
  el = []
  tables = []
  try:
    
    #TODO - refactor to use only pdfplumber and figure out how to get tables in context.
    with pdfplumber.open(temp_name) as pdf:
        for page in pdf.pages:
          for element in page:
            el.append(element)
          tables.append(page.extract_tables(table_settings={}))
      #for page_layout in extract_pages(temp_name):
      #  for element in page_layout:
      #    el.append(element)
  except:
      pass
  if not el:
      if temp_name is not None: os.system(f"rm {temp_name}")
      return None
  found_text = False
  image_files = []
  iw = ImageWriter(image_store_path)
  for element in el:
      if isinstance(element, LTTextContainer) or \
      isinstance(element, LTTextBox) or \
      isinstance(element, LTTextLine):
          found_text = True
          text += " "+element.get_text()
      elif isinstance(element, LTImage):
        try:
          image_id = iw.export_image(element)
          image_files.append(image_id)
          text += " "+img_template_tag%image_id
        except:
          continue
      elif isinstance(element, LTFigure) :
        el2 = []
        try:
          for element2 in element:
           el2.append(element2)
        except:
          pass
        for element2 in el2:
          if isinstance(element2, LTTextContainer) or \
          isinstance(element2, LTTextBox) or \
          isinstance(element2, LTTextLine):
              found_text = True
              text += " "+element2.get_text()
          elif isinstance(element2, LTImage):
            try:
              image_id = iw.export_image(element2)
              image_files.append(image_id)
              text += " "+img_template_tag%image_id
            except:
              continue

                     
  # if there is no text, then run OCR:
  if not found_text:
      text = meta
      text2, ocr_data = do_ocr_imgs(image_files)
      #TODO: do we keep the ocr_data - e.g., coordinates and other metadata for the text?
      if text2:
        for image_file in image_files:
          os.system(f"rm {image_file}")          
          text = text + " " + text2
      else:
        return ("", image_files)
      
  
  #NOTE: we do not take care of the case where there are embded text images inside a pdf. Only the case where everything is an image.
  if temp_name is not None: os.system(f"rm {temp_name}")
                     
  return (text, image_files)
