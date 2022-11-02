#@title Extract WARC Routines
#original code from https://github.com/LAION-AI/Big-Interleaved-Dataset which is under Apache 2
from riverbed.simhash import *
from riverbed.stopwords import *
from riverbed.preprocess_manager import *
from riverbed.kenlm_manager import *
from resiliparse.parse import detect_encoding
from resiliparse.parse.html import HTMLTree
from resiliparse.extract.html2text import extract_plain_text
import resiliparse
from urllib.parse import urljoin
import json
import os
from fastwarc.warc import ArchiveIterator
from pathlib import Path
from multiprocessing import Pool
import logging
import multiprocessing
import time
import simhash
import tqdm
from functools import partial
import logging
import gzip
logger = logging.getLogger(__name__)

logging.basicConfig(
    format='%(asctime)s : %(processName)s : %(levelname)s : %(message)s',
    level=logging.INFO)

# setup global config manually
global_config = dict()
global_config["cwd"] = Path().absolute()
global_config["warc_house"] = global_config["cwd"] / "warchouse/"
global_config["log_store"] = global_config["cwd"] / "logstore/"
global_config["warc_store"] = global_config["cwd"] / "warc_store/"

for y in global_config.values():
    y.mkdir(parents=True, exist_ok=True)

import itertools
import re

BILD_ELEMENT = re.compile('###\w+###\d+###')
NAV = re.compile('###nav###\d+###([^#]+)###/nav###')
META = re.compile('###meta###\d+###([^#]+)###/meta###')
CODE = re.compile('###code###\d+###([^#]+)###/code###')
TABLE = re.compile('###table###\d+###([^#]+)###/table###')
END_ELEMENT = re.compile('###/([^#]+)###')

def get_inner_el(ele, el_type):
  if type(ele) == resiliparse.parse.html.HTMLTree:
    return list(itertools.chain(*[get_inner_el(e, el_type) for e in ele.body.get_elements_by_tag_name(el_type)]))
  if ele.tag == el_type and not ele.get_elements_by_tag_name(el_type):
    return [ele]
  return list(itertools.chain(*[get_inner_el(e, el_type) for e in ele.get_elements_by_tag_name(el_type)]))

DIGIT_REGEX = re.compile(r"\d")

def parser_bytes(dup_span, dup_doc, warc_stats, url, html_byte, window_size, tokenization,  text_span_num_words=50, min_img_height=75,  min_img_width=75, sentence_dedup_shingle_size=5, cleanup_dup_span_limit=1000000, cleanup_dup_doc_limit=1000000, default_kenlm_wikipedia = "/content/drive/Shareddrives/LAION/kenlm_ccnet_wikipedia_models"):
    global lang_model, parameters_filtering
    html_byte = html_byte.replace(b"<polygon", b"<div").replace(b"<POLYGON", b"<DIV") # fixing a bug in resiliparse
    #- consider whether we want to use html2text and/or inscripts to parse some bild elements, like tables and navs.
    #- consider adding h1-h2-h3, etc. elements as section lables to spans of text/bild elements. you could create spans for text that associates to a section/subsections, along with position of those section/subsections
    #- consider tying the text for an internal anchor link to the anchor tag.  like a table of contents text points to an internal section in the doc.
    #- consider whether we want to extract keywords from URL as a type of BILD element.
    main, iframe_links, vids, imgs, auds, maps, rights, codes, tables, navs, metas, inputs, texts = dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict() , dict(), dict()
    bild2elements = {"imgs": imgs, "vids": vids, "auds": auds, "iframes": iframe_links, "rights": rights, "codes": codes, "tables": tables, "navs": navs, "metas": metas, "inputs": inputs, 'texts': texts }
    #each element in the bild table has the form (url, position, relative_position, bild id, simhash_code, val, additional info such as coordinates if any)
    #the url, position is the unique id for each row. 
    #relative position is position/length of document. This will tell you if an image is in the header or footer of a page for example.
    #bild id is a name like ###img###1###. 
    #val is usually url but could also be other content potentially (such as a table's content, code, etc.)
    #coordinates is special for images (and video??) as these are boxes with labels
    page_config = {"img_count": 0, "vid_count": 0, "aud_count": 0, "iframe_count": 0, "rights_count": 0, "code_count":0, "table_count": 0, "nav_count":0, "meta_count": 0, "image_coords_count": 0, "input_count": 0, 'text_count': 0 }
        
    # gather up all the ids or create ids for visible elements. we will use this to find coords/locations of each element.
    iframes_ids, vids_ids, imgs_ids, auds_ids, rights_ids, codes_ids, tables_ids, navs_ids, inputs_ids, texts_ids = [], [], [], [], [], [], [], [], [], []
    bild2ids = {"imgs": imgs_ids, "vids": vids_ids, "auds": auds_ids, "iframes": iframes_ids, "rights": rights_ids, "codes": codes_ids, "tables": tables_ids, "navs": navs_ids, "inputs": inputs_ids, 'texts': texts_ids}
    #bild2ids maps the html element id to the bild id. where there is no html element id, we use the bild id. 

    
    encoding = detect_encoding(html_byte)
    tree = HTMLTree.parse_from_bytes(html_byte, encoding)
    if not tree or not  tree.body:
      return None, None, None
    # we are walking through two copies of the tree in order to create ids in rendered html to get position info, while parsing the regular tree for the bild elements
    render_tree =  HTMLTree.parse_from_bytes(html_byte, encoding)
    if not render_tree.head.get_elements_by_tag_name('base'):
      base = render_tree.create_element('base')
      base_url = url.split("/")
      base.setattr("href", base_url[0]+"//"+base_url[2])
      #print (base)
      render_tree.head.append_child(base)
      render_tree.head.insert_before(base, render_tree.head.first_child)
         
    for ele in tree.head.get_elements_by_tag_name("script"):
      ele.parent.remove_child(ele)   
    for ele in tree.body.get_elements_by_tag_name("script"):
      ele.parent.remove_child(ele)
    for ele in tree.body.get_elements_by_tag_name("p"):
      if ele.text and not ele.text.endswith("."):
        ele.text = ele.text+"."
    for level in range(1, 7):
      for ele in tree.body.get_elements_by_tag_name("h"+str(level)):
        if ele.text and not ele.text.endswith("."):
          ele.text = ele.text+"."

    id2ids = bild2ids["rights"]
    for ele, render_ele in zip(tree.body.get_elements_by_tag_name("a"), render_tree.body.get_elements_by_tag_name("a")):
      text = " ".join(ele.text.split())
      #TODO for other languages. make it a set and run through the split words. 
      text_lower = text.lower()
      if (len(text) < 100 and ("privacy policy" in text_lower or "terms of use" in text_lower)) or (len(text) < 25 and ("eula" in text_lower or "policy" in text_lower or "policies" in text_lower or "terms" in text_lower or "privacy" in text_lower or "rights" in text_lower or "legal" in text_lower or  "patent" in text_lower or "license" in text_lower or "notice" in text_lower or "disclaimer" in text_lower or "agreement" in text_lower)):
        csrc = ele.getattr("href")
        if not csrc or not ele.parent: continue
        try:
          rights[f"###rights###{page_config['rights_count']}###"] = text_lower.replace(" ", "_")+":"+urljoin(
              url, csrc
          )
        except:
          continue
        ele.text = f" ###rights###{page_config['rights_count']}### " + text + " "   
        _id = (render_ele.getattr("id") if render_ele.getattr("id") else f"###rights###{page_config['rights_count']}###", f"###rights###{page_config['rights_count']}###")
        render_ele.setattr("id", _id[0])
        id2ids.append(_id)
        page_config["rights_count"] += 1
    
    id2ids = bild2ids["imgs"]
    for ele, render_ele in zip(tree.body.get_elements_by_tag_name("img"), render_tree.body.get_elements_by_tag_name("img")):
        csrc = ele.getattr("src")
        if not csrc: continue
        height = ele.getattr("height")
        width = ele.getattr("width")
        try:
          height= int(height.strip("px"))
        except:
          height = None
        try:
          width= int(width.strip("px"))
        except:
          width = None
        if height is not None and height <= min_img_height: continue
        if width is not None and width <= min_img_width: continue
        try:
          imgs[f"###img###{page_config['img_count']}###"] = urljoin(url, csrc)
        except:
          continue
        nele = tree.create_element("p")
        nele.text = f"###img###{page_config['img_count']}###" + ("" if not ele.getattr("alt") or "." in ele.getattr("alt")[:-1] else (" [ " + ele.getattr("alt") + " ] "))
        _id = (render_ele.getattr("id") if render_ele.getattr("id") else f"###img###{page_config['img_count']}###", f"###img###{page_config['img_count']}###")
        render_ele.setattr("id", _id[0])
        render_ele.setattr("src", "")
        id2ids.append(_id)
        ele.parent.append_child(nele)
        ele.parent.replace_child(nele, ele)
        usemap = ele.getattr("usemap")
        if usemap: 
          a_map = [a for a in tree.body.get_elements_by_tag_name("map") if a.getattr('name') and a.getattr('name').lower() == usemap.lstrip("#").lower()]
          if a_map:
            a_map = a_map[0]
            all_coords = [(area.getattr('shape'), area.getattr('coords'), area.getattr('alt')) for area in a_map.get_elements_by_tag_name("area") if area.getattr('alt')]
            if all_coords:
              maps[f"###img###{page_config['img_count']}###"] = all_coords
        page_config["img_count"] += 1

    id2ids = bild2ids["iframes"]
    for ele, render_ele in zip(tree.body.get_elements_by_tag_name("iframe"), render_tree.body.get_elements_by_tag_name("iframe")):
        csrc = ele.getattr("src")
        if not csrc or not ele.parent: continue
        try:
          iframe_links[f"###iframe###{page_config['iframe_count']}###"] = urljoin(
            url, csrc
          )
        except:
          continue
        nele = tree.create_element("p")
        nele.text = f"###iframe###{page_config['iframe_count']}###" + ("" if not ele.getattr("alt") or "." in ele.getattr("alt")[:-1] else (" [ " + ele.getattr("alt") + " ] "))
        _id = (render_ele.getattr("id") if render_ele.getattr("id") else f"###iframe###{page_config['iframe_count']}###", f"###iframe###{page_config['iframe_count']}###")
        render_ele.setattr("id", _id[0])
        render_ele.setattr("src", "")
        id2ids.append(_id)
        page_config["iframe_count"] += 1
        ele.parent.append_child(nele)
        ele.parent.replace_child(nele, ele)
    
    id2ids = bild2ids["vids"]
    for ele, render_ele in zip(tree.body.get_elements_by_tag_name("video"), render_tree.body.get_elements_by_tag_name("video")):
        if len(ele.get_elements_by_tag_name("source")) > 0:
            mele = ele.get_elements_by_tag_name("source")
            csrc = mele[0].getattr("src")
            if csrc and ele.parent:
              try:
                vids[f"###video###{page_config['vid_count']}###"] = urljoin(url, csrc)
              except:
                continue
              nele = tree.create_element("p")
              nele.text = f"###video###{page_config['vid_count']}###" + ("" if not ele.getattr("alt") or "." in ele.getattr("alt")[:-1] else (" [ " + ele.getattr("alt") + " ] "))
              _id = (render_ele.getattr("id") if render_ele.getattr("id") else f"###video###{page_config['vid_count']}###", f"###video###{page_config['vid_count']}###")
              render_ele.setattr("id", _id[0])
              id2ids.append(_id)
              render_ele.setattr("src", "")
              page_config["vid_count"] += 1
              ele.parent.insert_before(nele, ele)
              ele.parent.remove_child(ele) # why don't you do a replace here?
              continue # are we doing multiple videos per elements? if not, then we should skip the below.
        if ele.getattr("src") and ele.parent:
            csrc = ele.getattr("src")
            if csrc:
              try:
                vids[f"###video###{page_config['vid_count']}###"] = urljoin(url, csrc)
              except:
                continue
              nele = tree.create_element("p")
              nele.text =  f"###video###{page_config['vid_count']}###" + ("" if not ele.getattr("alt") or "." in ele.getattr("alt")[:-1] else (" [ " + ele.getattr("alt") + " ] "))
              _id = (render_ele.getattr("id") if render_ele.getattr("id") else f"###video###{page_config['vid_count']}###", f"###video###{page_config['vid_count']}###")
              render_ele.setattr("id", _id[0])
              id2ids.append(_id)
              render_ele.setattr("src", "")
              page_config["vid_count"] += 1
              ele.parent.append_child(nele)
              ele.parent.replace_child(nele, ele)
    
    id2ids = bild2ids["auds"]
    for ele, render_ele in zip(tree.body.get_elements_by_tag_name("audio"), render_tree.body.get_elements_by_tag_name("audio")):
        if len(ele.get_elements_by_tag_name("source")) > 0:
            mele = ele.get_elements_by_tag_name("source")
            csrc = mele[0].getattr("src")
            if csrc and ele.parent:
              try:
                auds[f"###audio###{page_config['aud_count']}###"] = urljoin(url, csrc)
              except:
                continue
              nele = tree.create_element("p")
              nele.text = f"###audio###{page_config['aud_count']}###" + ("" if not ele.getattr("alt") or "." in ele.getattr("alt")[:-1] else (" [ " + ele.getattr("alt") + " ] "))
              _id = (render_ele.getattr("id") if render_ele.getattr("id") else f"###audio###{page_config['aud_count']}###", f"###audio###{page_config['aud_count']}###")
              render_ele.setattr("id", _id[0])
              id2ids.append(_id)
              render_ele.setattr("src", "")
              page_config["aud_count"] += 1
              ele.parent.insert_before(nele, ele)
              ele.parent.remove_child(ele) # why don't you do a replace here?
              continue # are we doing multiple videos per elements? if not, then we should skip the below.
        if ele.getattr("src") and ele.parent:
            csrc = ele.getattr("src")
            try:
              auds[f"###audio###{page_config['aud_count']}###"] = urljoin(url, csrc)
            except:
              continue
            nele = tree.create_element("p")
            nele.text = f"###audio###{page_config['aud_count']}###" + ("" if not ele.getattr("alt") or "." in ele.getattr("alt")[:-1] else (" [ " + ele.getattr("alt") + " ] "))
            _id = (render_ele.getattr("id") if render_ele.getattr("id") else f"###audio###{page_config['aud_count']}###", f"###audio###{page_config['aud_count']}###")
            render_ele.setattr("id", _id[0])
            id2ids.append(_id)
            render_ele.setattr("src", "")
            ele.parent.append_child(nele)
            ele.parent.replace_child(nele, ele)
            page_config["aud_count"] += 1
    
    
    meta_text = ""
    for ele in itertools.chain(tree.head.get_elements_by_tag_name("meta"), tree.body.get_elements_by_tag_name("meta")):
      if ele.parent and ele.getattr('name') and ele.getattr('content'): #and  'adsense' not in ele.getattr('name') and 'verification' not in ele.getattr('name')  and ele.getattr('name')  not in ('apple-mobile-web-app-capable', 'applicable-device', 'viewport'):
        new_meta = ele.getattr('name').replace("\n", " ").replace("  ", " ").strip() + ": " + ele.getattr('content').replace("\n", " ").replace("  ", " ").strip()
        if meta_text:
          meta_text = meta_text + "; "+ new_meta
        else:
          meta_text =  new_meta
        ele.parent.remove_child(ele)
    if meta_text: 
      nele = tree.create_element("p")
      metas[f"###meta###{page_config['meta_count']}###"] = meta_text
      nele.text = f" ###meta###{page_config['meta_count']}###  "+meta_text+ " ###/meta### "
      page_config['meta_count'] += 1
      tree.body.append_child(nele)
      tree.body.insert_before(nele, tree.body.first_child)

    
    id2ids = bild2ids["codes"]
    for ele, render_ele in zip(tree.body.get_elements_by_tag_name("code"), render_tree.body.get_elements_by_tag_name("code")):
      text = ele.text
      if ele.parent and ("{" in text and "}" in text) or ("(" in text and ")" in text) or "table " in text or "class " in text or "function " in text or "def " in text or " void " in text or "*" in text:
        #print ( "###code###"+text+"###/code###")
        nele = tree.create_element("p")
        codes[f"###code###{page_config['code_count']}###"] = text
        nele.text = f" ###code###{page_config['code_count']}### "+text+ " ###/code### " 
        _id = (render_ele.getattr("id") if render_ele.getattr("id") else f"###code###{page_config['code_count']}###", f"###code###{page_config['code_count']}###")
        render_ele.setattr("id", _id[0])
        id2ids.append(_id)
        page_config['code_count'] += 1
        ele.parent.append_child(nele)
        ele.parent.replace_child(nele, ele)
    
    id2ids = bild2ids["navs"]
    for ele, render_ele in zip(tree.body.get_elements_by_tag_name("code"), render_tree.body.get_elements_by_tag_name("nav")):
      text = ele.text.replace("\t", " ").replace("  ", " ").strip()
      if text and ele.parent:
          nele = tree.create_element("p")
          text = " | ".join([a.strip() for a in text.split("\n") if a.strip()])
          if text:
            navs[f"###nav###{page_config['nav_count']}###"] = text
            nele.text = f" ###nav###{page_config['nav_count']}### "+text+ " ###/nav### "
            _id = (render_ele.getattr("id") if render_ele.getattr("id") else f"###nav###{page_config['nav_count']}###", f"###nav###{page_config['nav_count']}###")
            render_ele.setattr("id", _id[0])
            id2ids.append(_id)
            page_config['nav_count'] += 1
            ele.parent.append_child(nele)
            ele.parent.replace_child(nele, ele)

    
    id2ids = bild2ids["inputs"]
    for form, render_form in zip(tree.body.get_elements_by_tag_name("form"), render_tree.body.get_elements_by_tag_name("form")):
      form_name = (form.getattr("name") if form.getattr("name") else (form.getattr("id") if form.getattr("id")  else ""))
      els = list(zip(form.get_elements_by_tag_name("input"), render_form.get_elements_by_tag_name("input")))
      if len(els) < 3: continue #short forms are likely navigational forms and not content forms
      for ele, render_ele in els:
        val = ele.getattr("id") if ele.getattr("id") else ele.getattr("name")
        if val:
          value = ("" if ele.getattr("type") == "hidden" or not  ele.getattr("value") or len(ele.getattr("value")) > 10 else ":"+ele.getattr("value"))
          inputs[f"###inputs###{page_config['input_count']}###"] =  (form_name+"."+val if form_name else val) + value
          nele = tree.create_element("p")
          nele.text = f"###inputs###{page_config['input_count']}###" 
          _id = (render_ele.getattr("id") if render_ele.getattr("id") else f"###inputs###{page_config['input_count']}###", f"###inputs###{page_config['input_count']}###")
          render_ele.setattr("id", _id[0])
          id2ids.append(_id)
          page_config["input_count"] += 1
          ele.parent.append_child(nele)
          ele.parent.replace_child(nele, ele)
    
    #inner most tables, which likely corresponds to data and not layout
    inner_most_tables = get_inner_el(tree, 'table')
    render_inner_most_tables = get_inner_el(render_tree, 'table')
    if inner_most_tables:
      id2ids = bild2ids["tables"]
      for ele, render_ele in zip(inner_most_tables, render_inner_most_tables):
        table = []
        for tr in ele.get_elements_by_tag_name('tr'):
          if tr.text.strip():
            if tr.get_elements_by_tag_name('caption'):
              table.append(next(tr.get_elements_by_tag_name('caption').text.strip()))
            elif tr.get_elements_by_tag_name('th'):
              all_th = [th for th in tr.get_elements_by_tag_name('th') if th.text.strip()]
              if len(all_th) > 1:
                table.append (" | ".join ([th.text.replace("\n", " ").replace("  ", " ").strip() for th in all_th]))
                table.append (" | ".join (["---" for th in all_th]))
            elif tr.get_elements_by_tag_name('td'):
              all_td = [td for td in tr.get_elements_by_tag_name('td') if td.text.strip()]
              if len(all_td) > 1:
                table.append (" | ".join ([td.text.replace("\n", " ").replace("  ", " ").strip() for td in all_td])) 
                
        if (len(table) > 3 or (len(table) > 1 and "--" not in table[1])) and table[0].count("|") > 0: 
          if ele.parent:
            nele = tree.create_element("p")
            text = "; ".join(table)
            tables[f"###table###{page_config['table_count']}###"] = text
            nele.text = f" ###table###{page_config['table_count']}### "+text+" ###/table### "
            _id = (render_ele.getattr("id") if render_ele.getattr("id") else f"###table###{page_config['table_count']}###", f"###table###{page_config['table_count']}###")
            render_ele.setattr("id", _id[0])
            id2ids.append(_id)
            page_config['table_count'] += 1
            ele.parent.append_child(nele)
            ele.parent.replace_child(nele, ele)
    
    #NO bild elements
    if sum(page_config.values()) == 0:
      return None, None, None

    text = extract_plain_text(
      tree,
      preserve_formatting=False, #why False?
      main_content=False,
      list_bullets=True, # why False?
      alt_texts=True,
      links=False,
      form_fields=False,
      noscript=False,
    )
    if not text: return None, None, None
    text = text.strip()
    if text[0] == "<" and "> "  in text:
      text = text.split("> ")[1].strip()
    if not text: return None, None, None
    text = " ".join(text.replace(".. ", ". ").split())

    #According to the ccnet paper, we want to remove paragraph dups in order to do a better lang_id. 
    #So, here we will dedup and do some filtering on an unformatted version of the text.
    #First, we want to create a version of the text that has no bild elements
    #The function incremental_span_and_document_neardedup will remove span duplicates within the same document. 
    #And keep a span even if it is duplicated in *other* documents.
    #Then we can run the text through our lang_id filters, and our other filters which should be less sensitive to duplicates.
    #This is not as general as any prefix, but is relatively fast and "good enough" and can be run in an online fashion.
    unformatted_text = text
    unformatted_text = " ".join(unformatted_text.split())
    #we are computing filtering, langid and perplexity scores based on text without 
    #the bild interleaved content or duplicates since that skews the filtering 
    #(e.g., bild elements like code or a table may contain no stopwords for example). So let's remove them.
    #NOTE: double spaces denote sentence break for incremental_span_and_document_neardedup, so we 
    #create "psuedo" sentences betwen the BILD elements. Consider whether we want to do this or just replace with a single space.
    unformatted_text = re.sub(NAV, '  ', unformatted_text)
    unformatted_text = re.sub(CODE, '  ', unformatted_text)
    unformatted_text = re.sub(META, '  ', unformatted_text)
    unformatted_text = re.sub(TABLE, '  ', unformatted_text)
    unformatted_text = re.sub(BILD_ELEMENT, '  ', unformatted_text)
    doc_is_dup, unformatted_text, text = \
      incremental_span_and_document_neardedup(dup_span, dup_doc, \
                                              unformatted_text=unformatted_text, \
                                              formatted_text = text, \
                                              shingle_size = sentence_dedup_shingle_size, \
                                              cleanup_dup_span_limit=cleanup_dup_span_limit, \
                                              cleanup_dup_doc_limit=cleanup_dup_doc_limit, \
                                              normalize_text=True, \
                                              keep_first_dup_in_unformatted_text=False, \
                                              keep_first_dup_in_formatted_text=True, \
                                              replace_char='*')
    
    #incremental_span_and_document_neardedup returns doc_is_dup == 0 if no dups, 1 if partial dups, and 2 if complete dups.
    if doc_is_dup == 2: # this is a complete duplicate
      return None, None, None

    # the text now has been paragraph deduped within the same document. 
    main["text"] = text
    #create positional information in the bild2elements columns. uid is (page_url, position). we assume no overlaps for the same bild type. 
    len_text = len(text)
    bild_cnt = 0
    for bild_element_type, aHash in bild2elements.items():
      id2 = []
      for key, val in aHash.items():
        position = text.find(key)
        if position >= 0:
          bild_cnt += 1
          if maps.get(key):
            id2.append((url, position, position/len_text, key, hashing(val), val, maps[key]))
          else:
            id2.append((url, position, position/len_text, key, hashing(val), val, ()))
      main[bild_element_type] = id2
    if bild_cnt < 1:
      return None, None, None
    id2 = []
    #TODO: add text elements and divs into the render_html
    if False:
      #create positional information for text spans
      is_dup_chunk = set(itertools.chain(*is_dup_chunk.values()))
      for ch_idx, val in enumerate(chunks):
        is_dup = ch_idx in is_dup_chunk
        key = "###text###"+str(ch_idx)+"###"
        position = text.find(key)
        if position >= 0:
          id2.append((url, position, position/len_text, key, hashing(val), val, is_dup))

    main["texts"] = id2

    for bild_element_type, items in bild2ids.items():
      main[bild_element_type+"_ids"] = items
    
    page_config["image_coords_count"] = len(maps) #we handle this as a special case
    for key, val in page_config.items():
        warc_stats[key] = warc_stats.get(key,0) + val


    #according to the cc-net paper, dup paragraphs can affect low resource langs. 
    #let's remove the rest of the duplicates.
    #this is used ONLY for filtering.

    
    #langid detection and filtering
    lang, lang_score_pred = lang_id(unformatted_text)
    param = parameters_filtering.get(lang,  parameters_filtering["default"])
    lang_id_min_cutoff = param["lang_id_min_cutoff"]
    if lang_score_pred < lang_id_min_cutoff: 
      #print ('***', lang, '***', lang_score_pred, '****', clean_text)
      return None, None, None
    is_cjk = lang in {'zh', 'zh-classical', 'zh-min-nan', 'zh-yue', 'ko', 'ja', 'th', 'jv'}

    #min length filtering. we are filtering very short pages, unless it has some form of bild element.
    number_words_min_cutoff =  param["number_words_min_cutoff"]
    if is_cjk:
      if len(unformatted_text) < number_words_min_cutoff:
        return None, None, None
      elif len(unformatted_text.split()) < number_words_min_cutoff:
        return None, None, None

    #IDEA: we may want to do a running mean/stdev and filter based on something like 2 stdev for stopword filtering and perplexity filtering.
    #TODO: create stopwords for the rest of the languages
    #stopwords ratio filtering
    stopwords_min_cutoff = param["stopwords_min_cutoff"]
    stopword_score = get_stopword_score(lang, unformatted_text)
    if stopword_score <= stopwords_min_cutoff: 
      #print ('*** stopwords ***', lang, stopword_score, stopwords_min_cutoff, lang_score_pred, '***', clean_text)
      return None, None, None

    #TODO: put in the junk and repeition filtering.
        
    perplexity_max_cutoff = param["perplexity_max_cutoff"]
    perplexity_score = -1
    kenlm_model = load_kenlm_model(lang, default_kenlm_wikipedia = default_kenlm_wikipedia)
    if kenlm_model:
        kenlm_model = kenlm_model["wikipedia"]
    if kenlm_model:
        perplexity_score = kenlm_model.get_perplexity(unformatted_text)
        if perplexity_score > perplexity_max_cutoff: 
          #print ('perplexity filtered', lang, perplexity_score, clean_text)
          return None, None, None
    
    #ner = detect_ner_with_regex_and_context(text, lang)
    #main["ner"] = ner
    main["url"] = url
    main["html_stats"] = page_config
    main["lang"] = lang
    main["lang_score"] = lang_score_pred
    main["perplexity_score"] = perplexity_score
    main["stopword_score"] = stopword_score 
    #some written languages do not use spaces, in which case the tokenization must be character based
    #hack since we are over-riding the default tokenization. We can and should have different hashing parameters for each lang.
    #NOTE: we can consider each lang having their own simhash space. we can limit the compute that way.
    if is_cjk: tokenization = "character"
    text_to_hash = (re.sub(END_ELEMENT, ' ', re.sub(BILD_ELEMENT, ' ', main["text"])) + " " + \
                       " ".join(str(item) for item in imgs.values()) + " " + \
                       " ".join(str(item) for item in vids.values()) + " " + \
                       " ".join(str(item) for item in auds.values()) + " " + \
                       " ".join(str(item) for item in iframe_links.values()) + " ").replace("  ", " ")
    simhash_code = hashing(text_to_hash, \
                       window_size=window_size, \
                       tokenization=tokenization)
    main["simhash_code"] = simhash_code
    #main["text_to_hash"] = text_to_hash
    if sum(page_config.values()) > 0:
       render_tree = str(render_tree).strip()
        
    else:
        render_tree= ""

    return main, page_config, render_tree


def pipeline(dup_span, dup_doc, warc_record_store, warcpath, window_size, tokenization, verbose=False, default_kenlm_wikipedia="/content/drive/Shareddrives/LAION/kenlm_ccnet_wikipedia_models"):
    warc_path2 = os.path.splitext(warcpath)[0].replace(".warc", "")
    logging.basicConfig(
        filename=f"{global_config['log_store']}/{warc_path2}.log",
        level=logging.DEBUG,
        filemode="w",
        format="%(process)d:%(asctime)s:%(levelname)s:%(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    st = time.time()
    lang2simhashfilepath={}
    with open(warcpath, "rb") as f:
        warc_stats = {"img_count": 0, "vid_count": 0, "aud_count": 0, "iframe_count": 0, "rights_count": 0, "code_count":0, "table_count": 0, "nav_count":0, "meta_count": 0, "image_coords_count": 0, "input_count": 0, 'text_count': 0 }
        warc_stats["warc_html_hits"] = 0
        warc_stats["warc_exception_counts"] = 0
        warc_stats["warc_path"] = warc_path2
        warc_stats["current_window_size"] = window_size
        warc_stats["current_tokenization"] = tokenization
        store_path = global_config["warc_house"] / Path(
            warc_path2 
        )
        warc_stats_path = str(store_path).rstrip("/") + "_stats.json"
        warc_simhash_to_filepath = str(store_path).rstrip("/") + "_simhash_to_filepath"
        store_path.mkdir(parents=True, exist_ok=True)
        os.system(f"mkdir -p {warc_simhash_to_filepath}")
        if verbose:
          an_iter = tqdm.tqdm(ArchiveIterator(f, max_content_length=4 * 1024**2))
        else:
          an_iter = ArchiveIterator(f, max_content_length=4 * 1024**2)
        for index_r, record in enumerate(
              an_iter
          ):
            if record.headers is None:
                continue
            if record.http_headers is None:
                continue
            if (
                record.headers["WARC-Type"] == "response"
                and record.content_length >= 128
            ):
                content_type = str(record.http_content_type).lower()
                if content_type.startswith("text/html"):
                    warc_stats["warc_html_hits"] += 1
                    url = str(record.headers["WARC-Target-URI"])
                    html_bytes = record.reader.read()
                    html_pack, html_stats, render_html = parser_bytes(dup_span, dup_doc, warc_stats, url, html_bytes, window_size=window_size, tokenization=tokenization, default_kenlm_wikipedia=default_kenlm_wikipedia)
                    if not html_pack:
                      warc_stats["warc_exception_counts"] += 1
                      logging.debug(
                          f"An exception occured at index {warc_stats['warc_html_hits']}: Total exceptions: {warc_stats['warc_exception_counts']}"
                        )
                    else:
                      logging.debug(
                          f'Sucessflly parsed record index:{warc_stats["warc_html_hits"]}'
                  
                      )
                      if render_html:
                        lang = html_pack["lang"]
                        recno = warc_stats["warc_html_hits"]
                        lang2simhashfilepath[lang] = aHash = lang2simhashfilepath.get(lang, {})
                        html_pack["file_path"] =  f'{store_path}/record{recno}.html'
                        aHash[html_pack["simhash_code"]] =aHash.get(html_pack["simhash_code"], [])+[html_pack["file_path"]]
                        warc_record_store.write(json.dumps(html_pack)+"\n")
                        with open(
                            f'{store_path}/record{recno}.html',
                            "w",
                          ) as fre:
                            fre.write(render_html) 
                            fre.close()
                      if (warc_stats["warc_html_hits"]  + 1) % 1000 == 0:
                        json.dump(warc_stats, open(warc_stats_path, "w"))
        logging.debug(f"This took this much time:{time.time()-st}s")
        json.dump(warc_stats, open(warc_stats_path, "w"))
        for lang, aHash in lang2simhashfilepath.items():
          #change this to store into a out of core df or database?
          with open(f"{warc_simhash_to_filepath}/{lang}.tsv", "w") as lang2simhashfile:
            lang2simhashfile.write("\n".join([str(simhash_code)+"\t"+str(filepath) for simhash_code, filepath in aHash.items()])+"\n")
        return warc_stats


