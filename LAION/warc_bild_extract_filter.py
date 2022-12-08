#@title Extract WARC Routines
#original code from https://github.com/LAION-AI/Big-Interleaved-Dataset which is under Apache 2

import glob, os
from tqdm import trange
from time import sleep

from ..simhash import *
from ..stopwords import *
from ..filtering import *
from ..kenlm_manager import *
from ..pdf_and_ocr import *

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
import statistics


from PIL import Image
logger = logging.getLogger(__name__)

"""
    #each element in the bild2elements table has the form:
    #     (position, relative_position, bild id, simhash_code, val, is_yt_dl_supported, flagged for nsfw, additional info such as coordinates if any)
    #the url, position is the unique id for each row. 
    #relative position is position/length of document. This will tell you if an image is in the header or footer of a page for example.
    #bild id is a name like ###img###1###. 
    #val is usually url but could also be other content potentially (such as a table's content, code, etc.)
    #coordinates is special for images (and video??) as these are boxes with labels
    
"""    
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
NAV = re.compile('###nav###\d+###([^▁]+)▁/nav###')
META = re.compile('###meta###\d+###([^▁]+)▁/meta###')
CODE = re.compile('###code###\d+###([^▁]+)▁/code###')
FORM = re.compile('###form###\d+###([^▁]+)▁/form###')
ADDRESS = re.compile('###address###\d+###([^▁]+)▁/address###')
TABLE = re.compile('###table###\d+###([^▁]+)▁/table###')
END_ELEMENT = re.compile('###/([^#]+)###|▁/([^#]+)###')

def get_inner_el(ele, el_type):
  if type(ele) == resiliparse.parse.html.HTMLTree:
    return list(itertools.chain(*[get_inner_el(e, el_type) for e in ele.body.get_elements_by_tag_name(el_type)]))
  if ele.tag == el_type and not ele.get_elements_by_tag_name(el_type):
    return [ele]
  return list(itertools.chain(*[get_inner_el(e, el_type) for e in ele.get_elements_by_tag_name(el_type)]))

DIGIT_REGEX = re.compile(r"\d")


def is_yt_dl_supported (val):
  if "://" not in val: return 0
  base_url = val.split("://")[1].split("/")[0]
  base_url = base_url.split(".")
  if len(base_url) <= 1: return 0  
  if base_url[1] in yt_dlp_supported_sites: return 1
  if ".".join(base_url[-2:]) in yt_dlp_supported_sites: return 1
  return 0


def is_nsfw(val):
  if "://" not in val: return -1
  base_url = val.split("://")[1].split("/")[0]
  base_url = base_url.split(".")
  if len(base_url) <= 1: return -1
  url1 = base_url[1] 
  url2 = ".".join(base_url[-2:]) 
  if "porn" in url1 or ".xxx/" in url1 or "xxx." in url1 or "porn" in url2 or ".xxx/" in url2 or "xxx." in url2: return 1
  if url1 in all_porn_hosts: return 1
  if url1 in all_porn_hosts: return 1
  if url2 in all_gore_hosts: return 2
  if url2 in all_gore_hosts: return 2  
  return 0

def extract_html(html_bytes, warc_stats, idx, url):
    global lang_model, parameters_filtering, BILD_ELEMENT, NAV, META, CODE, FORM, ADDRESS, TABLE, END_ELEMENT, DIGIT_REGEX
    html_byte = html_byte.replace(b"<polygon", b"<div").replace(b"<POLYGON", b"<DIV") # fixing a bug in resiliparse
    #- consider whether we want to use html2text and/or inscripts to parse some bild elements, like tables and navs.
    #- and/or use pdf html table reader
    #- consider adding title, h1-h2-h3, etc. elements as section lables to spans of text/bild elements. you could create spans for text that associates to a section/subsections, along with position of those section/subsections
    #- consider tying the text for an internal anchor link to the anchor tag.  like a table of contents text points to an internal section in the doc.
    #- consider whether we want to extract keywords from URL as a type of BILD element.
    record, iframe_links, vids, imgs, auds, maps, rights_links, codes, tables, navs, metas, inputs, texts, forms, addresses, summary = dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict() , dict(), dict(), dict(), dict(), dict()
    bild2elements = {"imgs": imgs, "vids": vids, "auds": auds, "iframes": iframe_links, "rights_links": rights_links, "codes": codes, "tables": tables, "navs": navs, "metas": metas, "inputs": inputs, "forms": forms, "addresses": addresses, 'texts': texts, 'summary': summary }
    page_config = {"img_count": 0, "vid_count": 0, "aud_count": 0, "iframe_count": 0, "rights_links_count": 0, "code_count":0, "table_count": 0, "nav_count":0, "meta_count": 0, "image_coords_count": 0, "input_count": 0, 'text_count': 0, 'form_count': 0, 'address_count': 0, 'summary_count': 0 }
        
    # gather up all the ids or create ids for visible elements. we will use this to find coords/locations of each element.
    iframes_ids, vids_ids, imgs_ids, auds_ids, rights_links_ids, codes_ids, tables_ids, navs_ids, inputs_ids, texts_ids, forms_ids, addresses_ids, summary_ids = [], [], [], [], [], [], [], [], [], [], [], [], []
    bild2ids = {"imgs": imgs_ids, "vids": vids_ids, "auds": auds_ids, "iframes": iframes_ids, "rights_links": rights_links_ids, "codes": codes_ids, "tables": tables_ids, "navs": navs_ids, "inputs": inputs_ids, 'texts': texts_ids, 'forms': forms_ids, 'addresses': addresses_ids, 'summary': summary_ids}
    #bild2ids maps the html element id to the bild id. where there is no html element id, we use the bild id. 

    encoding = detect_encoding(html_byte)
    tree = HTMLTree.parse_from_bytes(html_byte, encoding)
    if not tree or not  tree.body:
      warc_stats["warc_exception_no_body_data_counts"] += 1
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
    
    title = tree.title.strip()
    found_title = False
    #Add a title and create psuedo sentences at certain html elements. 
    for level in range(1, 7):
      for ele in tree.body.get_elements_by_tag_name("h"+str(level)):
        if not found_title and title in ele.text:
          ele.text = ele.text.replace(title, f" ###title###0### {title}. ", 1)
          found_title = True
        if ele.text and not ele.text.endswith("."):
          ele.text = ele.text+"."
    for ele in list(tree.body.get_elements_by_tag_name("p")) + \
              list(tree.body.get_elements_by_tag_name("div")) + \
              list(tree.body.get_elements_by_tag_name("hr")) + \
              list(tree.body.get_elements_by_tag_name("pre")) + \
              list(tree.body.get_elements_by_tag_name("article")) + \
              list(tree.body.get_elements_by_tag_name("main")) + \
              list(tree.body.get_elements_by_tag_name("section")) + \
              list(tree.body.get_elements_by_tag_name("aside")):
      if not found_title and title in ele.text:
        ele.text = ele.text.replace(title, f" ###title###0### {title}. ", 1)
      if ele.text and not ele.text.endswith("."):
        ele.text = ele.text+"."


    id2ids = bild2ids["rights_links"]
    for ele, render_ele in zip(tree.body.get_elements_by_tag_name("a"), render_tree.body.get_elements_by_tag_name("a")):
      text = " ".join(ele.text.split())
      #TODO for other languages. make it a set and run through the split words. Need to do some translations.
      text_lower = text.lower().strip()
      if (len(text) < 100 and (text_lower in rights_links_kw or any(a in rights_links_kw for a in text_lower.split()))):
        csrc = ele.getattr("href")
        if not csrc or not ele.parent: continue
        try:
          rights_links[f"###rights_links###{page_config['rights_links_count']}###"] = text_lower.replace(" ", "_")+":"+urljoin(
              url, csrc
          )
        except:
          rights_links[f"###rights_links###{page_config['rights_links_count']}###"] = text_lower.replace(" ", "_")+":"+csrc
        nele = tree.create_element("p")
        nele.text = f" ###rights_links###{page_config['rights_links_count']}### " + text + " " 
        _id = (render_ele.getattr("id") if render_ele.getattr("id") else f"###rights_links###{page_config['rights_links_count']}###", f"###rights_links###{page_config['rights_links_count']}###")
        render_ele.setattr("id", _id[0])
        id2ids.append(_id)
        ele.parent.append_child(nele)
        ele.parent.replace_child(nele, ele)
        page_config["rights_links_count"] += 1
    
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
          imgs[f"###img###{page_config['img_count']}###"] = csrc
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
        
    #treat figcaption like an alt tag
    for ele in tree.body.get_elements_by_tag_name("figcaption"):
      nele = tree.create_element("p")
      nele.text = "[ "+ele.text+" ]"
      ele.parent.append_child(nele)
      ele.parent.replace_child(nele, ele)
    
    id2ids = bild2ids["iframes"]
    for ele, render_ele in zip(tree.body.get_elements_by_tag_name("iframe"), render_tree.body.get_elements_by_tag_name("iframe")):
        csrc = ele.getattr("src")
        if not csrc or not ele.parent: continue
        try:
          iframe_links[f"###iframe###{page_config['iframe_count']}###"] = urljoin(
            url, csrc
          )
        except:
          iframe_links[f"###iframe###{page_config['iframe_count']}###"] = csrc
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
                vids[f"###video###{page_config['vid_count']}###"] = csrc
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
                vids[f"###video###{page_config['vid_count']}###"] = csrc
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
                auds[f"###audio###{page_config['aud_count']}###"] = csrc
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
              auds[f"###audio###{page_config['aud_count']}###"] = csrc
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
      text = ele.text.strip()
      if text and ele.parent: # and (ele.tag == 'code' or (not ele.get_elements_by_tag_name("code") and ("{" in text and "}" in text) or ("(" in text and ")" in text) or "table " in text or "class " in text or "function " in text or "def " in text or " void " in text or "/*" in text or "*/" in text or "//" in text:
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
    for ele, render_ele in zip(list(tree.body.get_elements_by_tag_name("nav")) + list(tree.body.get_elements_by_tag_name("footer")) + list(tree.body.get_elements_by_tag_name("menu")), \
                               list(render_tree.body.get_elements_by_tag_name("nav")) + list(render_tree.body.get_elements_by_tag_name("footer")) + list(render_tree.body.get_elements_by_tag_name("menu"))):
      text = ele.text.replace("\t", " ").replace("  ", " ").strip()
      text = " | ".join([a.strip() for a in text.split("\n") if a.strip()])
      if text and ele.parent:
          nele = tree.create_element("p")          
          navs[f"###nav###{page_config['nav_count']}###"] = text
          nele.text = f" ###nav###{page_config['nav_count']}### "+text+ " ###/nav### "
          _id = (render_ele.getattr("id") if render_ele.getattr("id") else f"###nav###{page_config['nav_count']}###", f"###nav###{page_config['nav_count']}###")
          render_ele.setattr("id", _id[0])
          id2ids.append(_id)
          page_config['nav_count'] += 1
          ele.parent.append_child(nele)
          ele.parent.replace_child(nele, ele)


    id2ids = bild2ids["summary"]
    for detail, render_detail in zip(tree.body.get_elements_by_tag_name("details"), render_tree.body.get_elements_by_tag_name("details")):
      detail_name = (detail.getattr("name") if detail.getattr("name") else (detail.getattr("id") if detail.getattr("id")  else ""))
      els = list(zip(detail.get_elements_by_tag_name("summary"), render_detail.get_elements_by_tag_name("summary")))
      if els:
        ele, render_ele = els[0][0], els[0][1]
        summary_text = ele.text.strip()
        summary[f"###summary###{page_config['summary_count']}###"] =  summary_text +" | "+" ".join(ele.text.replace(summary_text, '').strip().split())
        nele = tree.create_element("p")
        nele.text = f"###summary###{page_config['summary_count']}### " +summary[f"###summary###{page_config['summary_count']}###"] 
        _id = (render_ele.getattr("id") if render_ele.getattr("id") else f"###summary###{page_config['summary_count']}###", f"###summary###{page_config['summary_count']}###")
        render_ele.setattr("id", _id[0])
        id2ids.append(_id)
        page_config["summary_count"] += 1
        detail.parent.append_child(nele)
        detail.parent.replace_child(nele, detail)
    
    
    id2ids = bild2ids["inputs"]
    for form, render_form in zip(tree.body.get_elements_by_tag_name("form"), render_tree.body.get_elements_by_tag_name("form")):
      form_name = (form.getattr("name") if form.getattr("name") else (form.getattr("id") if form.getattr("id")  else ""))
      els = list(zip(form.get_elements_by_tag_name("input"), render_form.get_elements_by_tag_name("input")))
      if len(els) < 3: continue #short forms are likely navigational forms and not content forms
      for ele, render_ele in els:
        val = ele.getattr("id") if ele.getattr("id") else ele.getattr("name")
        if val:
          value = ("" if ele.getattr("type") == "hidden" or not  ele.getattr("value") or len(ele.getattr("value")) > 10 else ":"+ele.getattr("value"))
          inputs[f"###input###{page_config['input_count']}###"] =  (form_name+"."+val if form_name else val) + value
          nele = tree.create_element("p")
          nele.text = f"###inputs###{page_config['input_count']}###" 
          _id = (render_ele.getattr("id") if render_ele.getattr("id") else f"###input###{page_config['input_count']}###", f"###input###{page_config['input_count']}###")
          render_ele.setattr("id", _id[0])
          id2ids.append(_id)
          page_config["input_count"] += 1
          ele.parent.append_child(nele)
          ele.parent.replace_child(nele, ele)
    
    id2ids = bild2ids["forms"]
    for ele, render_ele in zip(tree.body.get_elements_by_tag_name("form"), render_tree.body.get_elements_by_tag_name("form")):
      text = ele.text.replace("\t", " ").replace("  ", " ").strip()
      text = " | ".join([a.strip() for a in text.split("\n") if a.strip()])
      if text and ele.parent:
        forms[f"###form###{page_config['form_count']}###"] =  text
        nele = tree.create_element("p")
        nele.text = f"###form###{page_config['form_count']}### "+text+" ###/form### "
        _id = (render_ele.getattr("id") if render_ele.getattr("id") else f"###form###{page_config['form_count']}###" , f"###form###{page_config['form_count']}###" )
        render_ele.setattr("id", _id[0])
        id2ids.append(_id)
        page_config["form_count"] += 1
        ele.parent.append_child(nele)
        ele.parent.replace_child(nele, ele)

    id2ids = bild2ids["addresses"]
    for ele, render_ele in zip(tree.body.get_elements_by_tag_name("address"), render_tree.body.get_elements_by_tag_name("address")):
      text = ele.text.replace("\t", " ").replace("  ", " ").strip()
      text = " | ".join([a.strip() for a in text.split("\n") if a.strip()])
      if text and ele.parent: 
        addresses[f"###address###{page_config['address_count']}###"] =  text
        nele = tree.create_element("p")
        nele.text = f"###address###{page_config['address_count']}### "+text+" ###/address### "
        _id = (render_ele.getattr("id") if render_ele.getattr("id") else f"###address###{page_config['address_count']}###" , f"###address###{page_config['address_count']}###" )
        render_ele.setattr("id", _id[0])
        id2ids.append(_id)
        page_config["address_count"] += 1
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
      warc_stats["warc_exception_no_bild_elements_counts"] += 1
      return None

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
    if not text: 
      warc_stats["warc_exception_no_content_counts"] += 1
      return None
    text = text.strip()
    #TODO: do more cleaning, like text between {}, repetitive text and more text between <>
    if text[0] == "<" and "> "  in text:
      text = text.split("> ")[1].strip()
    if not text: 
      warc_stats["warc_exception_no_content_counts"] += 1
      return None
    record["title"] = title
    record["url"] = url
    record["stats"] = page_config   
    if sum(page_config.values()) > 0:
      render_html = str(render_tree).strip()
    else:
      render_html= ""          
    return (record, render_html, warc_stats)


def extract_pdf(pdf_bytes, warc_stats, recno, url, store_path, ocr_threshold=0.75):
    global lang_model, parameters_filtering, BILD_ELEMENT, NAV, META, CODE, FORM, ADDRESS, TABLE, END_ELEMENT, DIGIT_REGEX
    record, iframe_links, vids, imgs, auds, maps, rights_links, codes, tables, navs, metas, inputs, texts, forms, addresses, summary = dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict() , dict(), dict(), dict(), dict(), dict()
    bild2elements = {"imgs": imgs, "vids": vids, "auds": auds, "iframes": iframe_links, "rights_links": rights_links, "codes": codes, "tables": tables, "navs": navs, "metas": metas, "inputs": inputs, "forms": forms, "addresses": addresses, 'texts': texts, 'summary': summary }
    page_config = {"img_count": 0, "vid_count": 0, "aud_count": 0, "iframe_count": 0, "rights_links_count": 0, "code_count":0, "table_count": 0, "nav_count":0, "meta_count": 0, "image_coords_count": 0, "input_count": 0, 'text_count': 0, 'form_count': 0, 'address_count': 0, 'summary_count': 0 }
        
    # gather up all the ids or create ids for visible elements. we will use this to find coords/locations of each element.
    iframes_ids, vids_ids, imgs_ids, auds_ids, rights_links_ids, codes_ids, tables_ids, navs_ids, inputs_ids, texts_ids, forms_ids, addresses_ids, summary_ids = [], [], [], [], [], [], [], [], [], [], [], [], []
    bild2ids = {"imgs": imgs_ids, "vids": vids_ids, "auds": auds_ids, "iframes": iframes_ids, "rights_links": rights_links_ids, "codes": codes_ids, "tables": tables_ids, "navs": navs_ids, "inputs": inputs_ids, 'texts': texts_ids, 'forms': forms_ids, 'addresses': addresses_ids, 'summary': summary_ids}
    #bild2ids maps the html element id to the bild id. where there is no html element id, we use the bild id. 
    with open(
            f'{store_path}/record{recno}.pdf',
            "wb",
          ) as fre:
            fre.write(record.reader.read()) 
            fre.cl
    text = ""
    meta = ""
    title = ""
    try:
      fp = open(f'{store_path}/record{recno}.pdf', 'rb')
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
        if meta: meta = text = "###meta###0#### "+ meta + " ###/meta###"
      fp.close()
      fp = parser = doc = None 
    except:
      warc_stats["warc_exception_noresponse_counts"] += 1
      return None                    
    el = []
    try:
      for page_layout in extract_pages( f'{store_path}/record{recno}.pdf'):
        for element in page_layout:
          el.append(element)
    except:
      pass
    if not el:
      warc_stats["warc_exception_noresponse_counts"] += 1
      return None
    tables = []
    #TODO - refactor to use only pdfplumber and figure out how to get tables in context.
    with pdfplumber.open("background-checks.pdf") as pdf:
        for page in pdf.pages:
          tables.append(page.extract_tables(table_settings={}))


    found_text = False
    image_files = []
    iw = ImageWriter(store_path)
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
          text += f" ###img###{image_id}###"
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
              text += f" ###img###{image_id}###"
            except:
              continue

    # if there is no text, then run OCR:
    if not found_text and not tables:
      text = meta
      text2, data = do_ocr_imgs(image_files)
      text = text + " " + text2
      #TODO: do we keep coordinates and other metadata for the text?

    # maybe do two passes, one with all_ocr_readers, then do lang detect and then one with reader for the particular predominant lang(s)
    
    #TODO: add pdf forms and links
    #TODO: add the tables.

    #add title
    text = text.replace("\n", " ")
    
    
    if not text: return None
    #add img and meta counts and ids
    record['text'] = text
    record["title"] = title
    record["url"] = url
    record["stats"] = page_config   

    return (record, "", warc_stats)


def filter_and_tag_record(record, dup_span, dup_doc, idx, stopwords_scores_per_lang, perplexity_scores_per_lang, \
                 stopword_mean, stopword_stdev, perplexity_mean, perplexity_stdev,  simple_moving_avg_window, stopword_stdev_lower_bound, perplexity_stdev_upper_bound, \
                  window_size, tokenization,  special_char_max_cutoff=0.4, lang_id_min_cutoff=0.5, number_words_min_cutoff=30, \
                 text_span_num_words=50, min_img_height=75,  min_img_width=75, sentence_dedup_shingle_size=5, cleanup_dup_span_limit=1000000, \
                 cleanup_dup_doc_limit=1000000, default_kenlm_wikipedia = "/content/drive/Shareddrives/LAION/kenlm_ccnet_wikipedia_models"):    
    assert record is not None
    (record,  warc_stats) = record
    page_config = record['stats']
    text = record['text']
    text = " ".join(text.replace("。. ", "。 ").replace(". .", ".").replace(".. ", ". ").replace(" . . ", ". ").replace("?. ", "? ").replace("!. ", "! ").replace(",. ", ", ").replace(".”. ", ".” ").replace(".\" ", ".\" ").split())
    
    #According to the ccnet paper, we want to remove paragraph dups in order to do a better lang_id. 
    #So, here we will dedup and do some filtering on an unformatted version of the text.
    #First, we want to create a version of the text that has no bild elements or duplicates.
    #The function incremental_span_and_document_neardedup will remove span duplicates within the same document. 
    #But we will keep a span even if it is duplicated in *other* documents.
    #Then we can run the text through our lang_id filters, and our other filters which should be less sensitive to duplicates.
    #This is not as general as suffix array removal or md5 exact hash removal, but it is relatively fast, can be run in an online fashion, while keeping memory low.
    unformatted_text = text
    unformatted_text = " ".join(unformatted_text.split()).replace("###/", "▁/")
    #NOTE: double spaces denote sentence break for incremental_span_and_document_neardedup, so we 
    #create "psuedo" sentences betwen the BILD elements. Consider whether we want to do this or just replace with a single space.
    unformatted_text = re.sub(NAV, '  ', unformatted_text)
    unformatted_text = re.sub(CODE, '  ', unformatted_text)
    unformatted_text = re.sub(META, '  ', unformatted_text)
    unformatted_text = re.sub(FORM, '  ', unformatted_text)
    unformatted_text = re.sub(ADDRESS, '  ', unformatted_text)
    unformatted_text = re.sub(TABLE, '  ', unformatted_text)
    unformatted_text = re.sub(BILD_ELEMENT, '  ', unformatted_text)
    unformatted_text = re.sub(END_ELEMENT, '  ', unformatted_text)    
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
      warc_stats["warc_exception_duplicate_counts"] += 1
      return None, None, None
    if doc_is_dup == 1: # this is a complete duplicate
      warc_stats["warc_partial_duplicate_counts"] += 1

    # the text now has been paragraph deduped within the same document!
    record["text"] = text

    # now, create positional information in the bild2elements columns. uid is (page_url, position). we assume no overlaps for the same bild type. 
    len_text = len(text)
    bild_cnt = 0
    found_nsfw = -1
    for bild_element_type, aHash in bild2elements.items():
      id2 = []
      for key, val in aHash.items():
        position = text.find(key)
        if position >= 0:
          bild_cnt += 1
          is_supported = None if bild_element_type not in {'iframes',} else is_yt_dl_supported (val)
          flagged = is_nsfw(val)
          found_nsfw= max(found_nsfw, flagged)
          if is_supported: 
            warc_stats["warc_supported_"+bild_element_type] = warc_stats.get("warc_supported_"+bild_element_type, 0) + 1 
          if maps.get(key):
            id2.append((position, position/len_text, key, hashing(val), val, is_supported, flagged, maps[key]))
          else:
            id2.append((position, position/len_text, key, hashing(val), val, is_supported, flagged, ()))
      record[bild_element_type] = id2
    if bild_cnt < 1:
      warc_stats["warc_exception_no_bild_elements_counts"] += 1
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

    record["texts"] = id2

    for bild_element_type, items in bild2ids.items():
      record[bild_element_type+"_ids"] = items
    
    page_config["image_coords_count"] = len(maps) #we handle this as a special case
    for key, val in page_config.items():
        warc_stats[key] = warc_stats.get(key,0) + val
    
    #TODO: put in repeition filtering.

    #special char filtering
    unformatted_text2 = "".join(unformatted_text.split())
    if not unformatted_text2:
      warc_stats["warc_exception_no_content_counts"] += 1
      return None, None, None
    
    special_char_score = len([a for a in unformatted_text2 if a in special_characters_default])/len(unformatted_text2)
    if special_char_score > special_char_max_cutoff: 
      #print ('*** special char ****', special_char_score)
      warc_stats["warc_exception_special_char_counts"] += 1
      return None, None, None
    #langid detection and filtering
    lang, lang_score_pred = lang_id(unformatted_text)
    if lang_score_pred < lang_id_min_cutoff: 
      #print ('***', lang, '***', lang_score_pred, '****', clean_text)
      warc_stats["warc_exception_no_lang_counts"] += 1
      return None, None, None
    is_cjk = lang in {'zh', 'zh-classical', 'zh-min-nan', 'zh-yue', 'ko', 'ja', 'th', 'jv'}


    #min length filtering. we are filtering very short pages, unless it has some form of bild element.
    if is_cjk:
      if len(unformatted_text) < number_words_min_cutoff:
        warc_stats["warc_exception_num_words_cutoff_counts"] += 1
        return None, None, None
      elif len(unformatted_text.split()) < number_words_min_cutoff:
        warc_stats["warc_exception_num_words_cutoff_counts"] += 1
        warc_stats["warc_exception_num_words_cutoff_counts_"+ lang]  = warc_stats.get("warc_exception_num_words_cutoff_counts_"+ lang,0)+1
        return None, None, None

    #stopwords ratio filtering
    stopword_score = get_stopword_score(lang, unformatted_text)
    stopwords_min_cutoff = stopword_mean = stopword_median = stopword_stdev = stopword_quantiles = None
    stopwords_scores_per_lang[lang] = stopwords_scores = stopwords_scores_per_lang.get(lang,[])
    stopwords_scores.append(stopword_score)
    if len(stopwords_scores) < 2:
      stopwords_min_cutoff = stopword_score
    else:
      if len(stopwords_scores) > simple_moving_avg_window:
        stopwords_scores=stopwords_scores[-simple_moving_avg_window:]
      stopword_stdev = statistics.stdev(stopwords_scores)
      stopword_mean = statistics.mean (stopwords_scores)        
      stopword_median = statistics.median (stopwords_scores)        
      stopwords_min_cutoff = stopword_mean-(stopword_stdev*stopword_stdev_lower_bound)
    #if len(stopwords_scores) > 10: 
    #  stopword_quantiles = statistics.quantiles(stopwords_scores, n=10)
    if stopword_score < stopwords_min_cutoff: 
      warc_stats["warc_exception_stopwords_cutoff_counts"] += 1
      warc_stats["warc_exception_stopwords_cutoff_counts_"+ lang]  = warc_stats.get("warc_exception_stopwords_cutoff_counts_"+ lang,0)+1       
      return None, None, None
   
    #perplexity filtering
    perplexity_score = -1
    perplexity_max_cutoff = perplexity_mean = perplexity_median = perplexity_stdev = perplexity_quantiles = None
    kenlm_model = load_kenlm_model(lang, default_kenlm_wikipedia = default_kenlm_wikipedia)
    if kenlm_model:
        kenlm_model = kenlm_model["wikipedia"]
    if kenlm_model:
        perplexity_score = kenlm_model.get_perplexity(unformatted_text)
        perplexity_scores_per_lang[lang] = perplexity_scores = perplexity_scores_per_lang.get(lang,[])
        perplexity_scores.append(perplexity_score)
        if len(perplexity_scores) < 2:
          perplexity_max_cutoff = perplexity_score
        else:
          if len(perplexity_scores) > simple_moving_avg_window:
             perplexity_scores = perplexity_scores[-simple_moving_avg_window:] 
          perplexity_stdev = statistics.stdev(perplexity_scores)
          perplexity_mean = statistics.mean (perplexity_scores)
          perplexity_median = statistics.median (perplexity_scores)
          perplexity_max_cutoff = perplexity_mean+(perplexity_stdev*perplexity_stdev_upper_bound)
        #if len(perplexity_scores) > 10: 
        #    perplexity_quantiles = statistics.quantiles(perplexity_scores, n=10)
        if perplexity_score > perplexity_max_cutoff: 
          #print ('perplexity filtered', lang, perplexity_score, clean_text)
          warc_stats["warc_exception_perplexity_cutoff_counts"]  = warc_stats.get("warc_exception_perplexity_cutoff_counts",0)+1
          warc_stats["warc_exception_perplexity_cutoff_counts_"+ lang]  = warc_stats.get("warc_exception_perplexity_cutoff_counts_"+ lang,0)+1
          return None, None, None   

    #ner = detect_ner_with_regex_and_context(text, lang)
    #record["ner"] = ner
    record["title"] = title
    record["text_no_bild"] = unformatted_text
    record["url"] = url
    if found_nsfw > 0:
      record["nsfw"] = found_nsfw   
    else:
      record["nsfw"]  = is_nsfw(url)
    record["html_stats"] = page_config
    record["lang"] = lang
    record["lang_score"] = lang_score_pred
    record["stopword_score"] = stopword_score 
    record["stopword_stdev"] = stopword_stdev 
    record["stopword_mean"] = stopword_mean
    record["stopword_median"] = stopword_median
    record["stopwords_min_cutoff"] = stopwords_min_cutoff
    #record["stopword_quantiles"] = stopword_quantiles
    record["perplexity_score"] = perplexity_score
    record["perplexity_stdev"] = perplexity_stdev
    record["perplexity_mean"] = perplexity_mean
    record["perplexity_median"] = perplexity_median
    record["perplexity_max_cutoff"] = perplexity_max_cutoff
    record["special_char_score"] = special_char_score

    if record["nsfw"]:  warc_stats["warc_nsfw_counts"]  = warc_stats.get("warc_nsfw_counts",0) + 1
    #record["perplexity_quantiles"] = perplexity_quantiles
    #some written languages do not use spaces, in which case the tokenization must be character based
    #hack since we are over-riding the default tokenization. We can and should have different hashing parameters for each lang.
    #NOTE: we can consider each lang having their own simhash space. we can limit the compute that way.
    if is_cjk: tokenization = "character"

    #NOTE: For hashing text, to find similarity, we remove the nav and meta bild elements. 
    #We could remove other elements too like FORM, but this is a balancing act. 
    hash_text = text
    hash_text = " ".join(hash_text.split()).replace("###/", "▁/")
    hash_text = re.sub(NAV, '  ', hash_text)
    hash_text = re.sub(META, '  ', hash_text)
    hash_text = re.sub(BILD_ELEMENT, '  ', hash_text)
    hash_text = re.sub(END_ELEMENT, '  ', hash_text)    
    hash_text = hash_text + " " + \
                       " ".join(str(item) for item in imgs.values()) + " " + \
                       " ".join(str(item) for item in vids.values()) + " " + \
                       " ".join(str(item) for item in auds.values()) + " " + \
                       " ".join(str(item) for item in iframe_links.values()) 

    text_to_hash = " ".join(hash_text.split())
    simhash_code = hashing(text_to_hash, \
                       window_size=window_size, \
                       tokenization=tokenization)
    record["simhash_code"] = simhash_code
    #record["text_to_hash"] = text_to_hash


    return record


def pipeline(dup_span, dup_doc, \
             warc_record_store, warcpath, window_size, tokenization, verbose=False, \
             default_kenlm_wikipedia="/content/drive/Shareddrives/LAION/kenlm_ccnet_wikipedia_models", \
             stopwords_scores_per_lang=None, perplexity_scores_per_lang=None, \
             stopword_mean=None, stopwords_stdev=None, perplexity_mean=None, perplexity_stdev= None, \
             simple_moving_avg_window=500,  stopword_stdev_lower_bound=2, perplexity_stdev_upper_bound=2, \
             special_char_max_cutoff=0.4, lang_id_min_cutoff=0.5, number_words_min_cutoff=30, 
             do_render_html=True):
    if stopwords_scores_per_lang is None: stopwords_scores_per_lang ={}
    if perplexity_scores_per_lang is None: perplexity_scores_per_lang ={}
             
    warc_path2 = os.path.splitext(warcpath)[0].replace(".warc", "")
    logging.basicConfig(
        filename=f"{global_config['log_store']}/{warc_path2}.log",
        level=logging.DEBUG,
        filemode="w",
        format="%(process)d:%(asctime)s:%(levelname)s:%(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    st = time.time()
    with open(warcpath, "rb") as f:
        stopwords_scores_per_lang, perplexity_scores_per_lang, = {}, {}
        stopword_mean, stopwords_stdev, perplexity_mean, perplexity_stdev  = None, None, None, None
        warc_stats = {"img_count": 0, "vid_count": 0, "aud_count": 0, "iframe_count": 0, "rights_links_count": 0, "code_count":0, "table_count": 0, "nav_count":0, "meta_count": 0, "image_coords_count": 0, "input_count": 0, 'text_count': 0, 'form_count': 0, 'address_count': 0, 'summary_count': 0 }
        
        warc_stats["warc_html_hits"] = 0
        #warc_stats["warc_exception_noheader_counts"] = 0
        warc_stats["warc_exception_noresponse_counts"] = 0
        warc_stats["warc_exception_content_too_short_counts"] = 0
        warc_stats["warc_exception_perplexity_cutoff_counts"] = 0
        warc_stats["warc_exception_num_words_cutoff_counts"] = 0
        warc_stats["warc_exception_stopwords_cutoff_counts"]  = 0
        warc_stats["warc_exception_no_lang_counts"]  = 0
        warc_stats["warc_exception_no_bild_elements_counts"] = 0
        warc_stats["warc_exception_duplicate_counts"] = 0
        warc_stats["warc_partial_duplicate_counts"] = 0
        warc_stats["warc_exception_no_content_counts"] = 0
        warc_stats["warc_exception_no_body_data_counts"] = 0
        warc_stats["warc_exception_content_type_counts"] = 0
        warc_stats["warc_exception_special_char_counts"] = 0
        warc_stats["warc_path"] = warc_path2
        warc_stats["current_window_size"] = window_size
        warc_stats["current_tokenization"] = tokenization
        store_path = global_config["warc_house"] / Path(
            warc_path2 
        )
        warc_stats_path = str(store_path).rstrip("/") + "_stats.json"
        store_path.mkdir(parents=True, exist_ok=True)
        if verbose:
          an_iter = tqdm.tqdm(ArchiveIterator(f, max_content_length=4 * 1024**2))
        else:
          an_iter = ArchiveIterator(f, max_content_length=4 * 1024**2)
        for index_r, record in enumerate(
              an_iter
          ):
            if (index_r + 1) % 1000 == 0:
                json.dump(warc_stats, open(warc_stats_path, "w"))
                print (warc_stats)          
            #if record.headers is None:
            #    warc_stats["warc_exception_noheader_counts"] += 1
            #    print (record.http_headers)
            #    continue
            #elif record.http_headers is None:
            #    warc_stats["warc_exception_noheader_counts"] += 1
            #    print ('**', record.headers)
            #    continue
            #elif record.headers["WARC-Type"] != "response":
            #    warc_stats["warc_exception_noresponse_counts"] += 1
            #    continue
            #el
            
            if record.content_length < 128:
                warc_stats["warc_exception_content_too_short_counts"] += 1
                continue
            elif (
                record.headers["WARC-Type"] != "metadata"
                and record.content_length >= 128
            ): #: # "response"
                content_type = str(record.http_content_type).lower()
                content_type = str(record.http_content_type).lower()
                if content_type not in {"text/html", "application/pdf", }:
                  warc_stats["warc_exception_content_type_counts"] += 1
                elif content_type == "text/html":
                    url = str(record.headers["WARC-Target-URI"])
                    html_bytes = record.reader.read()
                    record = extract_html(dup_span, dup_doc,  warc_stats["warc_html_hits"])
                    html_pack, html_stats, render_html = filter_and_tag_record(record, stopwords_scores_per_lang, perplexity_scores_per_lang, stopword_mean, \
                                                                      stopwords_stdev, perplexity_mean, perplexity_stdev, simple_moving_avg_window, \
                                                                      stopword_stdev_lower_bound, perplexity_stdev_upper_bound, warc_stats, url, html_bytes, \
                                                                      special_char_max_cutoff=special_char_max_cutoff, lang_id_min_cutoff=lang_id_min_cutoff, number_words_min_cutoff=number_words_min_cutoff, \
                                                                      window_size=window_size, tokenization=tokenization, default_kenlm_wikipedia=default_kenlm_wikipedia)
                    if html_pack:
                      warc_stats["warc_html_hits"] += 1
                      warc_stats["warc_html_hits_"+html_pack["lang"]]  = warc_stats.get("warc_html_hits_"+html_pack["lang"],0) + 1
                      logging.debug(
                          f'Sucessflly parsed record index:{warc_stats["warc_html_hits"]}'
                  
                      )
                      if do_render_html and render_html:
                        lang = html_pack["lang"]
                        recno = warc_stats["warc_html_hits"]
                        html_pack["file_path"] =  f'{store_path}/record{recno}.html'
                        warc_record_store.write(json.dumps(html_pack)+"\n")
                        with open(
                            f'{store_path}/record{recno}.html',
                            "w",
                          ) as fre:
                            fre.write(render_html) 
                            fre.close()
                else: 
                    print (f"encountered {content_type}", str(record.headers["WARC-Target-URI"]))
                    pdf_bytes = record.reader.read()
                    record = extract_pdf(dup_span, dup_doc,  warc_stats["warc_html_hits"])
                    html_pack, html_stats, render_html = filter_and_tag_record(record, stopwords_scores_per_lang, perplexity_scores_per_lang, stopword_mean, \
                                                                      stopwords_stdev, perplexity_mean, perplexity_stdev, simple_moving_avg_window, \
                                                                      stopword_stdev_lower_bound, perplexity_stdev_upper_bound, warc_stats, url, pdf_bytes, \
                                                                      special_char_max_cutoff=special_char_max_cutoff, lang_id_min_cutoff=lang_id_min_cutoff, number_words_min_cutoff=number_words_min_cutoff, \
                                                                      window_size=window_size, tokenization=tokenization, default_kenlm_wikipedia=default_kenlm_wikipedia)
                      
                    
            else:
              warc_stats["warc_exception_noresponse_counts"] += 1

        logging.debug(f"This took this much time:{time.time()-st}s")
        json.dump(warc_stats, open(warc_stats_path, "w"))
        return warc_stats




def pipelines_for_warc_files(warc_files, tokenization, window_size,  save_dir, default_kenlm_wikipedia, simple_moving_avg_window, stopword_stdev_lower_bound, perplexity_stdev_upper_bound, special_char_max_cutoff, lang_id_min_cutoff, number_words_min_cutoff, ):
  if len(warc_files) > 1:
    warc_record_store_path = "./warchouse/" + warc_files[0].split("/")[-1].replace(".warc", "").replace(".gz", "") + "_" + warc_files[-1].split("/")[-1].replace(".warc", "").replace(".gz", "") +"_records.jsonl"
  else:
    warc_record_store_path = "./warchouse/" + warc_files[0].split("/")[-1].replace(".warc", "").replace(".gz", "") +".jsonl"

  with open(warc_record_store_path, "w", encoding="utf8") as warc_record_store:
    #TODO, save away in presistent storage and synch up every cycle the following data
    dup_span = {}
    dup_doc = {} 
    stopwords_scores_per_lang={}
    perplexity_scores_per_lang={}
    stopword_mean=None
    stopwords_stdev=None
    perplexity_mean=None
    perplexity_stdev=None
    for warc_file in get_warcs_from_save_dir(warc_files=warc_files):
      #print (warc_file)
      warc_stats = pipeline(dup_span, dup_doc, warc_record_store, warc_file,  tokenization= tokenization, \
                            window_size=window_size, verbose=False, default_kenlm_wikipedia=default_kenlm_wikipedia, \
                            stopwords_scores_per_lang=stopwords_scores_per_lang, perplexity_scores_per_lang=perplexity_scores_per_lang, \
                            stopword_mean=stopword_mean, stopwords_stdev=stopwords_stdev, perplexity_mean=perplexity_mean, perplexity_stdev=perplexity_stdev, \
                            simple_moving_avg_window=simple_moving_avg_window,  stopword_stdev_lower_bound=stopword_stdev_lower_bound, perplexity_stdev_upper_bound=perplexity_stdev_upper_bound,\
                            special_char_max_cutoff=special_char_max_cutoff, lang_id_min_cutoff=lang_id_min_cutoff, number_words_min_cutoff=number_words_min_cutoff, )
      os.system(f"rm {warc_file}") 
      #os.system(f"mv ./warchouse/*/ {save_dir}")


def extract_all_warcs(num_process = 6, simple_moving_avg_window=500, stopword_stdev_lower_bound=2, \
                      perplexity_stdev_upper_bound=2, special_char_max_cutoff=0.4, lang_id_min_cutoff=0.5, number_words_min_cutoff=30, 
                      save_dir="/content/drive/Shareddrives/LAION/CC-MAIN-2022-40/", dir="/content/drive/Shareddrives/LAION/CC-MAIN-2022-40/", \
                      tokenization= "character", window_size=24, default_kenlm_wikipedia = "/content/drive/Shareddrives/LAION/kenlm_ccnet_wikipedia_models"):
  #instead of downloading the kenlm models from FB research, we could save them away on a network drive
  #and copy them over to use with get_kenlm_models_from_savedir
  get_kenlm_models_from_savedir()
  os.system(f"mkdir -p {save_dir}/warchouse")
  files = glob.glob(f"{dir}*.gz")
  files.sort()
  batch_size = int(len(files)/num_process)
  plist=[]
  for rng in range(0, len(files), batch_size):
    max_rng = min(len(files), rng+batch_size)
    #print (len(files[rng:max_rng]))
    p = Process(target=pipelines_for_warc_files, args=(files[rng:max_rng], tokenization, window_size, save_dir+"/warchouse", default_kenlm_wikipedia, simple_moving_avg_window, stopword_stdev_lower_bound, perplexity_stdev_upper_bound, special_char_max_cutoff, lang_id_min_cutoff, number_words_min_cutoff))
    plist.append(p)
    p.start()
  
  t = trange(200*len(files), desc='number of records', leave=True)
  prev_len = 0
  for i in t:
    if not any(p for p in plist if p.is_alive()):
      break
    t.set_description(f"number of records({prev_len})", refresh=True)
    t.refresh()
    files2 = glob.glob(f"./warchouse/*/*")
    #print (len(files2))
    while len(files2) < prev_len + 10:
      sleep(0.01)
      files2 = glob.glob(f"./warchouse/*/*")
    prev_len = len(files2)
  for p in plist:
    p.join()

#TODO: aggregate all *.jsonl file into one big jsonl at end of processing
    
