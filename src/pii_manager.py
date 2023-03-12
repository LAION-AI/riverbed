#Need to refactor to use the new PIISA API.

#from https://github.com/piisa/muliwai/ which is under Apache 2.0
#for production, we should pip install or git clone from piisa
import re, regex
regex_rulebase = {
    #"CHEMICAL": {
    #  "default": [(re.compile('[A-Z][a-z]?\d*|\((?:[^()]*(?:\(.*\))?[^()]*)+\)\d+'), None, None)]
    #},
    "EQUATION": {
      "default": [(re.compile('\\\[([^\])+\\\]|\\\(([^\\\])+\\\)|\$([^\$])+\$|\\\begin\{math\}([^\\\])+\\\end\{math\}|\\\begin\{displaymath\}([^\\\])+\\\end\{displaymath\}|\\\begin\{equation\}([^\\\])+\\\end\{equation\}'), None, None)]
    },
    "AGE": {
      #TODO - finish out "years old" in other languages.
      "en": [
          (
              re.compile(
                  r"\S+ years old|\S+\-years\-old|\S+ year old|\S+\-year\-old", re.IGNORECASE
              ),
              None, None
          )
      ],
       "zh": [(regex.compile(r"([一二三四五六七八九十百\d]{1,3}歲|[一二三四五六七八九十百\d]{1,3}岁)"), None, None)],
    },
    "EMAIL": {
      "default": [(regex.Regex('(?:^|[\\s\\b\\\'\\"@,?!;:)(.\\p{Han}])([^\\s@,?!;:)(]+@[^,\\s!?;,]+[^\\s\\b\\\'\\"@,?!;:)(.])(?:$|[\\s\\b@,?!;:)(.\\p{Han}])', flags=regex.M | regex.V0), None, None)]
    },
    "DATE": {
        #TODO - separate all the languages out. Do pt, fr, es
        "id": [(re.compile('\d{4}|[0-3]?\d[-\./][0-3]?\d[-\./]\d{2,4}'), None, [('lahir', 'AGE'),])], 
        "default": [(re.compile('\d{4}|[0-3]?\d[-\./][0-3]?\d[-\./]\d{2,4}'), None, [('born', 'AGE'), ("ni a bi lori",'AGE'), ("wazalwa ngo",'AGE'), ("akazvarwa",'AGE'), ("o hlahile ka",'AGE'), ("anabadwa pa",'AGE'), ("wazalwa ngo",'AGE'), ("alizaliwa tarehe",'AGE'), ("amụrụ",'AGE'), ("ولد",'AGE'), ("生於",'AGE'), ("sinh ra",'AGE'), ("का जन्म ए",'AGE'), ("پیدا ہوا",'AGE'), ('lahir', 'AGE'),  ('জন্ম', 'AGE')])],
    },
    #https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py. Low to no PII 
    "TIME": {
      "default": [(re.compile('\d{1,2}:\d{2} ?(?:[ap]\.?m\.?)?|\d[ap]\.?m\.?', re.IGNORECASE), None, None),],
    },
    #if we want to match embeded PII within URLs
    "URL": {
      "default": [(re.compile('https?:\/\/[^\s\"\']{8,50}|www[^\s\"\']{8,50}', re.IGNORECASE), None, None)],
      "zh": [(regex.compile('(https?:\/\/.\P{Han}{1,}|www\.\P{Han}{1,50})', re.IGNORECASE), None, None)],
    },
    "PHONE": {
      "zh" : [(regex.compile(r"\d{4}-\d{8}"), None, ('pp', 'pp.', )),
              
              #from https://github.com/Aggregate-Intellect/bigscience_aisc_pii_detection/blob/main/language/zh/rules.py which is under Apache 2
              (regex.compile('(0?\d{2,4}-[1-9]\d{6,7})|({\+86|086}-| ?1[3-9]\d{9} , ([\+0]?86)?[\-\s]?1[3-9]\d{9})'), None, None),
        ],
      # we can probably remove one of the below
      "default": [
              # https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py phone with exts
              (regex.Regex(
                '(?:^|[\\s\\\'\\"(\\p{Han}])((?:\\+\\p{Nd}+[ \\/.\\p{Pd}]*)?(?:(?:\\(\\+?\\p{Nd}+\\))?(?:[ \\/.\\p{Pd}]*\\p{Nd})){7,}(?:[\\t\\f #]*\\p{Nd}+)?)(?:$|[\\s@,?!;:\\\'\\"(.\\p{Han}])',
                           flags=regex.M | regex.V0), None, ('pp', 'pp.', ))
      ]      
    },
    "IP_ADDRESS": {
        "default": [(regex.Regex('(?:^|[\\b\\s@?,!;:\\\'\\")(.\\p{Han}])((?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?:\\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}|(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\\.){3,3}(25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9]))(?:$|[\\s@,?!;:\'"(.\\p{Han}])', 
                                 flags=regex.M | regex.V0), None, None)]
              
        },
    "USER": {
      "default": [
              #generic user id
              (regex.Regex('(?:^|[\\s@,?!;:\\\'\\")(\\p{Han}])(@[^\\s@,?!;:\\\'\\")(]{3,})', flags=regex.M | regex.V0), None, None),
      ]    
    },
    #need a global license plate regex
    "LICENSE_PLATE": {
      "en": [
              #en license plate
              (regex.compile('[A-Z]{3}-\d{4}|[A-Z]{1,3}-[A-Z]{1,2}-\d{1,4}'), None, None)
      ],
      "zh": [ #from https://github.com/Aggregate-Intellect/bigscience_aisc_pii_detection/blob/main/language/zh/rules.py which is under Apache 2
              #LICENSE_PLATE
              (regex.compile('(\b[A-Z]{3}-\d{4}\b)'), None, None),
              (regex.compile('^(?:[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领 A-Z]{1}[A-HJ-NP-Z]{1}(?:(?:[0-9]{5}[DF])|(?:[DF](?:[A-HJ-NP-Z0-9])[0-9]{4})))|(?:[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领 A-Z]{1}[A-Z]{1}[A-HJ-NP-Z0-9]{4}[A-HJ-NP-Z0-9 挂学警港澳]{1})$'), None, None),
      ],
    },
    "KEY": {
      "default": [(regex.Regex('(?:^|[\\b\\s@?,!:;\\\'\\")(.\\p{Han}])((?:(?:[A-Za-z]+[\\p{Nd}\\p{Pd}\\/\\+\\=:_]+|[\\p{Nd}\\p{Pd}\\/\\+\\=:]+[A-Za-z]+)){4,}|(?:(?:\\p{Nd}{3,}|[A-Z]+\\p{Nd}+[A-Z]*|\\p{Nd}+[A-Z]+\\p{Nd}*)[ \\p{Pd}]?){3,})(?:$|[\\b\\s\\p{Han}@?,!;:\\\'\\")(.])', flags=regex.M | regex.V0), None, None)]
    },
    "ID": {
      "zh": [ #from https://github.com/Aggregate-Intellect/bigscience_aisc_pii_detection/blob/main/language/zh/rules.py which is under Apache 2
              #since we can't capture some of the zh rules under the general rules
              (regex.compile('(?:[16][1-5]|2[1-3]|3[1-7]|4[1-6]|5[0-4])\d{4}(?:19|20)\d{2}(?:(?:0[469]|11)(?:0[1-9]|[12][0-9]|30)|(?:0[13578]|1[02])(?:0[1-9]|[12][0-9]|3[01])|02(?:0[1-9]|[12][0-9]))\d{3}[\dXx]'), None, None),
              (regex.compile('(^[EeKkGgDdSsPpHh]\d{8}$)|(^(([Ee][a-fA-F])|([DdSsPp][Ee])|([Kk][Jj])|([Mm][Aa])|(1[45]))\d{7}$)'), None, None),
          ],
      "default": [
              #credit card from common regex
              (re.compile('((?:(?:\\d{4}[- ]?){3}\\d{4}|\\d{15,16}))(?![\\d])'), None, ('pp', 'pp.', )),
              #icd code - see https://stackoverflow.com/questions/5590862/icd9-regex-pattern
              (re.compile('[A-TV-Z][0-9][A-Z0-9](\.[A-Z0-9]{1,4})'), None, ('pp', 'pp.', )),
              # generic id with dashes - this sometimes catches a - or a / at the beginning of a number which might not be what we want.
              (re.compile('[A-Z#]{0,3}(?:[-\/ ]*\d){6,13}'), None, ('pp', 'pp.', )),
              # Meg's regex
              (regex.Regex('(?:^|[\\b\\s@?,!;:\\\'\\")(.\\p{Han}])([A-Za-z]*(?:[\\p{Pd}]*\\p{Nd}){6,})(?:$|[\\b\\s@?,!;:\\\'\\")(.\\p{Han}])', flags=regex.M | regex.V0), None, ('pp', 'pp.', )),
              # IBAN
              (re.compile('[A-Z]{2}\d+\d+[A-Z]{0,4}(?:[- ]*\d){10,32}[A-Z]{0,3}'), None, ('pp', 'pp.', )),
      ],
    },
 }

 #from https://github.com/piisa/muliwai/ which is under Apache 2.0
#This is an incomplete list. TODO - adapt from https://github.com/scrapinghub/dateparser/blob/master/dateparser/data/languages_info.py
country_2_lang = {
    'aa': ['ar'],
    'ad': ['ca'],
    'ae': ['ar'],
    'af': ['fa', 'ps'],
    'ag': ['en'],
    'ai': ['en'],
    'al': ['sq'],
    'am': ['hy'],
    'ao': ['pt'],
    'ar': ['es'],
    'as': ['en', 'sm'],
    'at': ['de'],
    'au': ['en'],
    'aw': ['nl', 'pap'],
    'ax': ['sv'],
    'az': ['az'],
    'ba': ['hr', 'bs', 'sr'],
    'bb': ['en'],
    'bd': ['bn'],
    'be': ['de', 'nl', 'fr'],
    'bf': ['fr'],
    'bg': ['bg'],
    'bh': ['ar'],
    'bi': ['en', 'fr', 'rn'],
    'bj': ['fr'],
    'bl': ['fr'],
    'bm': ['en'],
    'bn': ['ms'],
    'bo': ['es', 'qu', 'ay'],
    'bq': ['nl'],
    'br': ['pt'],
    'bs': ['en'],
    'bt': ['dz'],
    'bw': ['en', 'tn'],
    'by': ['ru', 'be'],
    'bz': ['en'],
    'ca': ['en', 'fr'],
    'cc': ['en'],
    'cd': ['fr'],
    'cf': ['sg', 'fr'],
    'cg': ['fr'],
    'ch': ['de', 'gsw', 'fr', 'it'],
    'ci': ['fr'],
    'ck': ['en'],
    'cl': ['es'],
    'cm': ['en', 'fr'],
    'cn': ['zh'],
    'co': ['es'],
    'cr': ['es'],
    'cu': ['es'],
    'cv': ['pt'],
    'cw': ['nl', 'pap'],
    'cx': ['en'],
    'cy': ['tr', 'el'],
    'cz': ['cs'],
    'de': ['de'],
    'dj': ['ar', 'fr'],
    'dk': ['da'],
    'dm': ['en'],
    'do': ['es'],
    'dz': ['ar', 'fr'],
    'ec': ['es', 'qu'],
    'ee': ['et'],
    'eg': ['ar'],
    'eh': ['ar'],
    'er': ['en', 'ar', 'ti'],
    'es': ['es'],
    'et': ['am'],
    'fi': ['fi', 'sv'],
    'fj': ['fj', 'en', 'hif'],
    'fk': ['en'],
    'fm': ['en'],
    'fo': ['fo'],
    'fr': ['fr'],
    'ga': ['fr'],
    'gb': ['en'],
    'gd': ['en'],
    'ge': ['ka'],
    'gf': ['fr'],
    'gg': ['en'],
    'gh': ['en'],
    'gi': ['en'],
    'gl': ['kl'],
    'gm': ['en'],
    'gn': ['fr'],
    'gp': ['fr'],
    'gq': ['pt', 'es', 'fr'],
    'gr': ['el'],
    'gt': ['es'],
    'gu': ['en', 'ch'],
    'gw': ['pt'],
    'gy': ['en'],
    'hk': ['zh', 'en'],
    'hn': ['es'],
    'hr': ['hr'],
    'ht': ['fr', 'ht'],
    'hu': ['hu'],
    'id': ['id'],
    'ie': ['en', 'ga'],
    'il': ['ar', 'he'],
    'im': ['en', 'gv'],
    'in': ['en', 'hi'],
    'io': ['en'],
    'iq': ['ar'],
    'ir': ['fa'],
    'is': ['is'],
    'it': ['it'],
    'je': ['en'],
    'jm': ['en'],
    'jo': ['ar'],
    'jp': ['ja'],
    'ke': ['en', 'sw'],
    'kg': ['ru', 'ky'],
    'kh': ['km'],
    'ki': ['en', 'gil'],
    'km': ['zdj', 'ar', 'fr', 'wni'],
    'kn': ['en'],
    'kp': ['ko'],
    'kr': ['ko'],
    'kw': ['ar'],
    'ky': ['en'],
    'kz': ['ru', 'kk'],
    'la': ['lo'],
    'lb': ['ar'],
    'lc': ['en'],
    'li': ['de', 'gsw'],
    'lk': ['ta', 'si'],
    'lr': ['en'],
    'ls': ['st', 'en'],
    'lt': ['lt'],
    'lu': ['de', 'lb', 'fr'],
    'lv': ['lv'],
    'ly': ['ar'],
    'ma': ['ar', 'tzm', 'fr'],
    'mc': ['fr'],
    'md': ['ro'],
    'me': ['sr'],
    'mf': ['fr'],
    'mg': ['mg', 'en', 'fr'],
    'mh': ['en', 'mh'],
    'mk': ['mk'],
    'ml': ['fr'],
    'mm': ['my'],
    'mn': ['mn'],
    'mo': ['pt', 'zh'],
    'mp': ['en'],
    'mq': ['fr'],
    'mr': ['ar'],
    'ms': ['en'],
    'mt': ['mt', 'en'],
    'mu': ['en', 'fr'],
    'mv': ['dv'],
    'mw': ['en', 'ny'],
    'mx': ['es'],
    'my': ['ms'],
    'mz': ['pt'],
    'na': ['en'],
    'nc': ['fr'],
    'ne': ['fr'],
    'nf': ['en'],
    'ng': ['en', 'yo'],
    'ni': ['es'],
    'nl': ['nl'],
    'no': ['nn', 'nb'],
    'np': ['ne'],
    'nr': ['na', 'en'],
    'nu': ['niu', 'en'],
    'nz': ['en', 'mi'],
    'om': ['ar'],
    'pa': ['es'],
    'pe': ['es', 'qu'],
    'pf': ['fr', 'ty'],
    'pg': ['tpi', 'ho', 'en'],
    'ph': ['fil', 'en'],
    'pk': ['en', 'ur'],
    'pl': ['pl'],
    'pm': ['fr'],
    'pn': ['en'],
    'pr': ['en', 'es'],
    'ps': ['ar'],
    'pt': ['pt'],
    'pw': ['en', 'pau'],
    'py': ['es', 'gn'],
    'qa': ['ar'],
    'qc': ['fr'],
    're': ['fr'],
    'ro': ['ro'],
    'rs': ['sr'],
    'ru': ['ru'],
    'rw': ['en', 'rw', 'fr'],
    'sa': ['ar'],
    'sb': ['en'],
    'sc': ['en', 'fr'],
    'sd': ['en', 'ar'],
    'se': ['sv'],
    'sg': ['zh', 'en', 'ta', 'ms'],
    'sh': ['en'],
    'si': ['sl'],
    'sj': ['nb'],
    'sk': ['sk'],
    'sl': ['en'],
    'sm': ['it'],
    'sn': ['fr', 'wo'],
    'so': ['ar', 'so'],
    'sr': ['nl'],
    'ss': ['en'],
    'st': ['pt'],
    'sv': ['es'],
    'sx': ['en', 'nl'],
    'sy': ['ar', 'fr'],
    'sz': ['en', 'ss'],
    'tc': ['en'],
    'td': ['ar', 'fr'],
    'tg': ['fr'],
    'th': ['th'],
    'tj': ['tg'],
    'tk': ['tkl', 'en'],
    'tl': ['pt', 'tet'],
    'tm': ['tk'],
    'tn': ['ar', 'fr'],
    'to': ['en', 'to'],
    'tr': ['tr'],
    'tt': ['en'],
    'tv': ['tvl', 'en'],
    'tw': ['zh'],
    'tz': ['en', 'sw'],
    'ua': ['ru', 'uk'],
    'ug': ['en', 'sw'],
    'um': ['en'],
    'us': ['en'],
    'uy': ['es'],
    'uz': ['uz'],
    'va': ['it'],
    'vc': ['en'],
    've': ['es'],
    'vg': ['en'],
    'vi': ['en'],
    'vn': ['vi'],
    'vu': ['en', 'bi', 'fr'],
    'wf': ['fr'],
    'ws': ['en', 'sm'],
    'ye': ['ar'],
    'yt': ['fr'],
    'za': ['en'],
    'zm': ['en'],
    'zw': ['en', 'sn', 'nd']
 }

#TODO - get the complete list for our language
lang_2_country = {
    'am': ['et'],
    'ar': [
        'ae','iq','dz','eg','sd','aa','il','ps','sa','bh','km','dj','er','eh','jo','kw',
        'lb','ly','ma','mr','om','qa','so','sy','td','tn','ye'
        ],
    'ay': ['bo'],
    'az': ['az'],
    'be': ['by'],
    'bg': ['bg'],
    'bi': ['vu'],
    'bn': ['bd'],
    'bs': ['ba'],
    'ca': ['ad'],
    'ch': ['gu'],
    'cs': ['cz'],
    'da': ['dk'],
    'de': ['at', 'ch', 'de', 'be', 'li', 'lu'],
    'dv': ['mv'],
    'dz': ['bt'],
    'el': ['gr', 'cy'],
    'en': [
        'pk','sd','au','ca','gb','gh','ie','in','nz','us','ai','as','ag','bi','bs','bz',
        'bm','bb','bw','cc','cm','ck','cx','ky','dm','er','fj','fk','fm','gg','gi','gm',
        'gd','gu','gy','hk','im','io','jm','je','ke','ki','kn','lr','lc','ls','mg','mh',
        'mt','mp','ms','mu','mw','na','nf','ng','nu','nr','pn','ph','pw','pg','pr','rw',
        'sg','sh','sb','sl','ss','sz','sx','sc','tc','tk','to','tt','tv','tz','ug','um',
        'vc','vg','vi','vu','ws','za','zm','zw'
    ],
    'es': [
        'ar','es','mx','bo','cl','co','cr','cu','do','ec','gq','gt','hn','ni','pa','pe',
        'pr','py','sv','uy','ve'
    ],
    'et': ['ee'],
    'fa': ['ir', 'af'],
    'fi': ['fi'],
    'fil': ['ph'],
    'fj': ['fj'],
    'fo': ['fo'],
    'fr': [
        'dz','ca','ch','fr','qc','bi','be','bj','bf','bl','cf','ci','cm','cd','cg','km',
        'dj', 'ga','gn','gp','gq','gf','ht','lu','mf','ma','mc','mg','ml','mq','mu','yt',
        'nc','ne', 'pf','re','rw','sn','pm','sc','sy','td','tg','tn','vu','wf'
    ],
    'ga': ['ie'],
    'gil': ['ki'],
    'gn': ['py'],
    'gsw': ['ch', 'li'],
    'gv': ['im'],
    'he': ['il'],
    'hi': ['in'],
    'hif': ['fj'],
    'ho': ['pg'],
    'hr': ['hr', 'ba'],
    'ht': ['ht'],
    'hu': ['hu'],
    'hy': ['am'],
    'id': ['id'],
    'is': ['is'],
    'it': ['ch', 'it', 'sm', 'va'],
    'ja': ['jp'],
    'ka': ['ge'],
    'kk': ['kz'],
    'kl': ['gl'],
    'km': ['kh'],
    'ko': ['kr', 'kp'],
    'ky': ['kg'],
    'lb': ['lu'],
    'lo': ['la'],
    'lt': ['lt'],
    'lv': ['lv'],
    'mg': ['mg'],
    'mh': ['mh'],
    'mi': ['nz'],
    'mk': ['mk'],
    'mn': ['mn'],
    'ms': ['bn', 'my', 'sg'],
    'mt': ['mt'],
    'my': ['mm'],
    'na': ['nr'],
    'nb': ['no', 'sj'],
    'nd': ['zw'],
    'ne': ['np'],
    'niu': ['nu'],
    'nl': ['nl', 'aw', 'be', 'bq', 'cw', 'sr', 'sx'],
    'nn': ['no'],
    'ny': ['mw'],
    'pap': ['aw', 'cw'],
    'pau': ['pw'],
    'pl': ['pl'],
    'ps': ['af'],
    'pt': ['br','pt','ao','cv','gw','gq','mo','mz','st','tl'],
    'qu': ['bo', 'ec', 'pe'],
    'rn': ['bi'],
    'ro': ['ro', 'md'],
    'ru': ['ru', 'ua', 'by', 'kz', 'kg'],
    'rw': ['rw'],
    'sg': ['cf'],
    'si': ['lk'],
    'sk': ['sk'],
    'sl': ['si'],
    'sm': ['as', 'ws'],
    'sn': ['zw'],
    'so': ['so'],
    'sq': ['al'],
    'sr': ['ba', 'me', 'rs'],
    'ss': ['sz'],
    'st': ['ls'],
    'sv': ['fi', 'se', 'ax'],
    'sw': ['ke', 'tz', 'ug'],
    'ta': ['lk', 'sg'],
    'tet': ['tl'],
    'tg': ['tj'],
    'th': ['th'],
    'ti': ['er'],
    'tk': ['tm'],
    'tkl': ['tk'],
    'tn': ['bw'],
    'to': ['to'],
    'tpi': ['pg'],
    'tr': ['tr', 'cy'],
    'tvl': ['tv'],
    'ty': ['pf'],
    'tzm': ['ma'],
    'uk': ['ua'],
    'ur': ['pk'],
    'uz': ['uz'],
    'vi': ['vn'],
    'wni': ['km'],
    'wo': ['sn'],
    'yo': ['ng'],
    'zdj': ['km'],
    'zh': ['cn', 'tw', 'hk', 'mo', 'sg']
}

#from https://github.com/piisa/muliwai/ which is under Apache 2.0
"""
Copyright, 2021-2022 Ontocord, LLC, All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import stdnum
import re, regex
import dateparser
try:
  from postal.parser import parse_address
except:
  parse_address = None
import sys, os
try:
  sys.path.append(os.path.abspath(os.path.dirname(__file__)))    
except:
  pass
#from stopwords import stopwords
#from country_2_lang import *
#from pii_regexes_rulebase import regex_rulebase
from stdnum import (bic, bitcoin, casrn, cusip, ean, figi, grid, gs1_128, iban, \
                    imei, imo, imsi, isan, isbn, isil, isin, ismn, iso11649, iso6346, \
                    iso9362, isrc, issn, lei,  mac, meid, vatin)
from stdnum.ad import nrt
from stdnum.al import nipt
from stdnum.ar import dni
from stdnum.ar import cbu
from stdnum.ar import cuit
from stdnum.at import businessid
from stdnum.at import tin
from stdnum.at import vnr
from stdnum.at import postleitzahl
from stdnum.at import uid
from stdnum.au import abn
from stdnum.au import acn
from stdnum.au import tfn
from stdnum.be import iban
#from stdnum.be import nn
from stdnum.be import vat
from stdnum.bg import vat
from stdnum.bg import egn
from stdnum.bg import pnf
from stdnum.br import cnpj
from stdnum.br import cpf
from stdnum.by import unp
from stdnum.ca import sin
from stdnum.ca import bn
from stdnum.ch import ssn
from stdnum.ch import vat
from stdnum.ch import uid
from stdnum.ch import esr
from stdnum.cl import rut
from stdnum.cn import ric
from stdnum.cn import uscc
from stdnum.co import nit
from stdnum.cr import cpj
from stdnum.cr import cr
from stdnum.cr import cpf
from stdnum.cu import ni
from stdnum.cy import vat
from stdnum.cz import dic
from stdnum.cz import rc
from stdnum.de import vat
from stdnum.de import handelsregisternummer
from stdnum.de import wkn
from stdnum.de import stnr
from stdnum.de import idnr
from stdnum.dk import cvr
from stdnum.dk import cpr
from stdnum.do import rnc
from stdnum.do import ncf
from stdnum.do import cedula
from stdnum.ec import ruc
from stdnum.ec import ci
from stdnum.ee import ik
from stdnum.ee import registrikood
from stdnum.ee import kmkr
from stdnum.es import iban
from stdnum.es import ccc
from stdnum.es import cif
from stdnum.es import dni
from stdnum.es import cups
from stdnum.es import referenciacatastral
from stdnum.es import nie
from stdnum.es import nif
from stdnum.eu import banknote
from stdnum.eu import eic
from stdnum.eu import vat
from stdnum.eu import at_02
from stdnum.eu import nace
from stdnum.fi import associationid
from stdnum.fi import veronumero
from stdnum.fi import hetu
from stdnum.fi import ytunnus
from stdnum.fi import alv
from stdnum.fr import siret
from stdnum.fr import tva
from stdnum.fr import nir
from stdnum.fr import nif
from stdnum.fr import siren
from stdnum.gb import upn
from stdnum.gb import vat
from stdnum.gb import nhs
from stdnum.gb import utr
from stdnum.gb import sedol
from stdnum.gr import vat
from stdnum.gr import amka
from stdnum.gt import nit
from stdnum.hr import oib
from stdnum.hu import anum
from stdnum.id import npwp
from stdnum.ie import vat
from stdnum.ie import pps
from stdnum.il import hp
from stdnum.il import idnr
from stdnum.in_ import epic
from stdnum.in_ import gstin
from stdnum.in_ import pan
from stdnum.in_ import aadhaar
from stdnum.is_ import kennitala
from stdnum.is_ import vsk
from stdnum.it import codicefiscale
from stdnum.it import aic
from stdnum.it import iva
from stdnum.jp import cn
from stdnum.kr import rrn
from stdnum.kr import brn
from stdnum.li import peid
from stdnum.lt import pvm
from stdnum.lt import asmens
from stdnum.lu import tva
from stdnum.lv import pvn
from stdnum.mc import tva
from stdnum.md import idno
from stdnum.me import iban
from stdnum.mt import vat
from stdnum.mu import nid
from stdnum.mx import curp
from stdnum.mx import rfc
from stdnum.my import nric
from stdnum.nl import bsn
from stdnum.nl import brin
from stdnum.nl import onderwijsnummer
from stdnum.nl import btw
from stdnum.nl import postcode
from stdnum.no import mva
from stdnum.no import iban
from stdnum.no import kontonr
from stdnum.no import fodselsnummer
from stdnum.no import orgnr
from stdnum.nz import bankaccount
from stdnum.nz import ird
from stdnum.pe import cui
from stdnum.pe import ruc
from stdnum.pl import pesel
from stdnum.pl import nip
from stdnum.pl import regon
from stdnum.pt import cc
from stdnum.pt import nif
from stdnum.py import ruc
from stdnum.ro import onrc
from stdnum.ro import cui
from stdnum.ro import cf
from stdnum.ro import cnp
from stdnum.rs import pib
from stdnum.ru import inn
from stdnum.se import vat
from stdnum.se import personnummer
from stdnum.se import postnummer
from stdnum.se import orgnr
from stdnum.sg import uen
from stdnum.si import ddv
from stdnum.sk import rc
from stdnum.sk import dph
from stdnum.sm import coe
from stdnum.sv import nit
from stdnum.th import pin
from stdnum.th import tin
from stdnum.th import moa
from stdnum.tr import vkn
from stdnum.tr import tckimlik
from stdnum.tw import ubn
from stdnum.ua import rntrc
from stdnum.ua import edrpou
from stdnum.us import ssn
from stdnum.us import atin
from stdnum.us import rtn
from stdnum.us import tin
from stdnum.us import ein
from stdnum.us import itin
from stdnum.us import ptin
from stdnum.uy import rut
from stdnum.ve import rif
from stdnum.vn import mst
from stdnum.za import tin
from stdnum.za import idnr

stdnum_mapper = {
    'ad.nrt':  stdnum.ad.nrt.validate,
    'al.nipt':  stdnum.al.nipt.validate,
    'ar.dni':  stdnum.ar.dni.validate,
    'ar.cbu':  stdnum.ar.cbu.validate,
    'ar.cuit':  stdnum.ar.cuit.validate,
    'at.businessid':  stdnum.at.businessid.validate,
    'at.tin':  stdnum.at.tin.validate,
    'at.vnr':  stdnum.at.vnr.validate,
    'at.postleitzahl':  stdnum.at.postleitzahl.validate,
    'at.uid':  stdnum.at.uid.validate,
    'au.abn':  stdnum.au.abn.validate,
    'au.acn':  stdnum.au.acn.validate,
    'au.tfn':  stdnum.au.tfn.validate,
    'be.iban':  stdnum.be.iban.validate,
    #'be.nn':  stdnum.be.nn.validate,
    'be.vat':  stdnum.be.vat.validate,
    'bg.vat':  stdnum.bg.vat.validate,
    'bg.egn':  stdnum.bg.egn.validate,
    'bg.pnf':  stdnum.bg.pnf.validate,
    'br.cnpj':  stdnum.br.cnpj.validate,
    'br.cpf':  stdnum.br.cpf.validate,
    'by.unp':  stdnum.by.unp.validate,
    'ca.sin':  stdnum.ca.sin.validate,
    'ca.bn':  stdnum.ca.bn.validate,
    'ch.ssn':  stdnum.ch.ssn.validate,
    'ch.vat':  stdnum.ch.vat.validate,
    'ch.uid':  stdnum.ch.uid.validate,
    'ch.esr':  stdnum.ch.esr.validate,
    'cl.rut':  stdnum.cl.rut.validate,
    'cn.ric':  stdnum.cn.ric.validate,
    'cn.uscc':  stdnum.cn.uscc.validate,
    'co.nit':  stdnum.co.nit.validate,
    'cr.cpj':  stdnum.cr.cpj.validate,
    'cr.cr':  stdnum.cr.cr.validate,
    'cr.cpf':  stdnum.cr.cpf.validate,
    'cu.ni':  stdnum.cu.ni.validate,
    'cy.vat':  stdnum.cy.vat.validate,
    'cz.dic':  stdnum.cz.dic.validate,
    'cz.rc':  stdnum.cz.rc.validate,
    'de.vat':  stdnum.de.vat.validate,
    'de.handelsregisternummer':  stdnum.de.handelsregisternummer.validate,
    'de.wkn':  stdnum.de.wkn.validate,
    'de.stnr':  stdnum.de.stnr.validate,
    'de.idnr':  stdnum.de.idnr.validate,
    'dk.cvr':  stdnum.dk.cvr.validate,
    'dk.cpr':  stdnum.dk.cpr.validate,
    'do.rnc':  stdnum.do.rnc.validate,
    'do.ncf':  stdnum.do.ncf.validate,
    'do.cedula':  stdnum.do.cedula.validate,
    'ec.ruc':  stdnum.ec.ruc.validate,
    'ec.ci':  stdnum.ec.ci.validate,
    'ee.ik':  stdnum.ee.ik.validate,
    'ee.registrikood':  stdnum.ee.registrikood.validate,
    'ee.kmkr':  stdnum.ee.kmkr.validate,
    'es.iban':  stdnum.es.iban.validate,
    'es.ccc':  stdnum.es.ccc.validate,
    'es.cif':  stdnum.es.cif.validate,
    'es.dni':  stdnum.es.dni.validate,
    'es.cups':  stdnum.es.cups.validate,
    'es.referenciacatastral':  stdnum.es.referenciacatastral.validate,
    'es.nie':  stdnum.es.nie.validate,
    'es.nif':  stdnum.es.nif.validate,
    'eu.banknote':  stdnum.eu.banknote.validate,
    'eu.eic':  stdnum.eu.eic.validate,
    'eu.vat':  stdnum.eu.vat.validate,
    'eu.at_02':  stdnum.eu.at_02.validate,
    'eu.nace':  stdnum.eu.nace.validate,
    'fi.associationid':  stdnum.fi.associationid.validate,
    'fi.veronumero':  stdnum.fi.veronumero.validate,
    'fi.hetu':  stdnum.fi.hetu.validate,
    'fi.ytunnus':  stdnum.fi.ytunnus.validate,
    'fi.alv':  stdnum.fi.alv.validate,
    'fr.siret':  stdnum.fr.siret.validate,
    'fr.tva':  stdnum.fr.tva.validate,
    'fr.nir':  stdnum.fr.nir.validate,
    'fr.nif':  stdnum.fr.nif.validate,
    'fr.siren':  stdnum.fr.siren.validate,
    'gb.upn':  stdnum.gb.upn.validate,
    'gb.vat':  stdnum.gb.vat.validate,
    'gb.nhs':  stdnum.gb.nhs.validate,
    'gb.utr':  stdnum.gb.utr.validate,
    'gb.sedol':  stdnum.gb.sedol.validate,
    'gr.vat':  stdnum.gr.vat.validate,
    'gr.amka':  stdnum.gr.amka.validate,
    'gt.nit':  stdnum.gt.nit.validate,
    'hr.oib':  stdnum.hr.oib.validate,
    'hu.anum':  stdnum.hu.anum.validate,
    'id.npwp':  stdnum.id.npwp.validate,
    'ie.vat':  stdnum.ie.vat.validate,
    'ie.pps':  stdnum.ie.pps.validate,
    'il.hp':  stdnum.il.hp.validate,
    'il.idnr':  stdnum.il.idnr.validate,
    'in_.epic':  stdnum.in_.epic.validate,
    'in_.gstin':  stdnum.in_.gstin.validate,
    'in_.pan':  stdnum.in_.pan.validate,
    'in_.aadhaar':  stdnum.in_.aadhaar.validate,
    'is_.kennitala':  stdnum.is_.kennitala.validate,
    'is_.vsk':  stdnum.is_.vsk.validate,
    'it.codicefiscale':  stdnum.it.codicefiscale.validate,
    'it.aic':  stdnum.it.aic.validate,
    'it.iva':  stdnum.it.iva.validate,
    'jp.cn':  stdnum.jp.cn.validate,
    'kr.rrn':  stdnum.kr.rrn.validate,
    'kr.brn':  stdnum.kr.brn.validate,
    'li.peid':  stdnum.li.peid.validate,
    'lt.pvm':  stdnum.lt.pvm.validate,
    'lt.asmens':  stdnum.lt.asmens.validate,
    'lu.tva':  stdnum.lu.tva.validate,
    'lv.pvn':  stdnum.lv.pvn.validate,
    'mc.tva':  stdnum.mc.tva.validate,
    'md.idno':  stdnum.md.idno.validate,
    'me.iban':  stdnum.me.iban.validate,
    'mt.vat':  stdnum.mt.vat.validate,
    'mu.nid':  stdnum.mu.nid.validate,
    'mx.curp':  stdnum.mx.curp.validate,
    'mx.rfc':  stdnum.mx.rfc.validate,
    'my.nric':  stdnum.my.nric.validate,
    'nl.bsn':  stdnum.nl.bsn.validate,
    'nl.brin':  stdnum.nl.brin.validate,
    'nl.onderwijsnummer':  stdnum.nl.onderwijsnummer.validate,
    'nl.btw':  stdnum.nl.btw.validate,
    'nl.postcode':  stdnum.nl.postcode.validate,
    'no.mva':  stdnum.no.mva.validate,
    'no.iban':  stdnum.no.iban.validate,
    'no.kontonr':  stdnum.no.kontonr.validate,
    'no.fodselsnummer':  stdnum.no.fodselsnummer.validate,
    'no.orgnr':  stdnum.no.orgnr.validate,
    'nz.bankaccount':  stdnum.nz.bankaccount.validate,
    'nz.ird':  stdnum.nz.ird.validate,
    'pe.cui':  stdnum.pe.cui.validate,
    'pe.ruc':  stdnum.pe.ruc.validate,
    'pl.pesel':  stdnum.pl.pesel.validate,
    'pl.nip':  stdnum.pl.nip.validate,
    'pl.regon':  stdnum.pl.regon.validate,
    'pt.cc':  stdnum.pt.cc.validate,
    'pt.nif':  stdnum.pt.nif.validate,
    'py.ruc':  stdnum.py.ruc.validate,
    'ro.onrc':  stdnum.ro.onrc.validate,
    'ro.cui':  stdnum.ro.cui.validate,
    'ro.cf':  stdnum.ro.cf.validate,
    'ro.cnp':  stdnum.ro.cnp.validate,
    'rs.pib':  stdnum.rs.pib.validate,
    'ru.inn':  stdnum.ru.inn.validate,
    'se.vat':  stdnum.se.vat.validate,
    'se.personnummer':  stdnum.se.personnummer.validate,
    'se.postnummer':  stdnum.se.postnummer.validate,
    'se.orgnr':  stdnum.se.orgnr.validate,
    'sg.uen':  stdnum.sg.uen.validate,
    'si.ddv':  stdnum.si.ddv.validate,
    'sk.rc':  stdnum.sk.rc.validate,
    'sk.dph':  stdnum.sk.dph.validate,
    'sm.coe':  stdnum.sm.coe.validate,
    'sv.nit':  stdnum.sv.nit.validate,
    'th.pin':  stdnum.th.pin.validate,
    'th.tin':  stdnum.th.tin.validate,
    'th.moa':  stdnum.th.moa.validate,
    'tr.vkn':  stdnum.tr.vkn.validate,
    'tr.tckimlik':  stdnum.tr.tckimlik.validate,
    'tw.ubn':  stdnum.tw.ubn.validate,
    'ua.rntrc':  stdnum.ua.rntrc.validate,
    'ua.edrpou':  stdnum.ua.edrpou.validate,
    'us.ssn':  stdnum.us.ssn.validate,
    'us.atin':  stdnum.us.atin.validate,
    'us.rtn':  stdnum.us.rtn.validate,
    'us.tin':  stdnum.us.tin.validate,
    'us.ein':  stdnum.us.ein.validate,
    'us.itin':  stdnum.us.itin.validate,
    'us.ptin':  stdnum.us.ptin.validate,
    'uy.rut':  stdnum.uy.rut.validate,
    've.rif':  stdnum.ve.rif.validate,
    'vn.mst':  stdnum.vn.mst.validate,
    'za.tin':  stdnum.za.tin.validate,
    'za.idnr':  stdnum.za.idnr.validate,
    'bic':  stdnum.bic.validate,
    'bitcoin':  stdnum.bitcoin.validate,
    'casrn':  stdnum.casrn.validate,
    'cusip':  stdnum.cusip.validate,
    'ean':  stdnum.ean.validate,
    'figi':  stdnum.figi.validate,
    'grid':  stdnum.grid.validate,
    'gs1_128':  stdnum.gs1_128.validate,
    'iban':  stdnum.iban.validate,
    'imei':  stdnum.imei.validate,
    'imo':  stdnum.imo.validate,
    'imsi':  stdnum.imsi.validate,
    'isan':  stdnum.isan.validate,
    'isbn':  stdnum.isbn.validate,
    'isil':  stdnum.isil.validate,
    'isin':  stdnum.isin.validate,
    'ismn':  stdnum.ismn.validate,
    'iso11649':  stdnum.iso11649.validate,
    'iso6346':  stdnum.iso6346.validate,
    'isrc':  stdnum.isrc.validate,
    'issn':  stdnum.issn.validate,
    'lei':  stdnum.lei.validate,
    'mac':  stdnum.mac.validate,
    'meid':  stdnum.meid.validate,
    'vatin':  stdnum.vatin.validate,
}

#this is based on lang_2_country. must be changed if we change lang_2_country.
lang_2_stdnum = {'am': [],
    'ar': ['il.hp', 'il.idnr'],
    'ay': [],
    'az': [],
    'be': ['by.unp'],
    'bg': ['bg.vat', 'bg.egn', 'bg.pnf'],
    'bi': [],
    'bn': [],
    'bs': [],
    'ca': ['ad.nrt'],
    'ch': [],
    'cs': ['cz.dic', 'cz.rc'],
    'da': ['dk.cvr', 'dk.cpr'],
    'de': ['at.businessid',
      'at.tin',
      'at.vnr',
      'at.postleitzahl',
      'at.uid',
      'ch.ssn',
      'ch.vat',
      'ch.uid',
      'ch.esr',
      'de.vat',
      'de.handelsregisternummer',
      'de.wkn',
      'de.stnr',
      'de.idnr',
      'be.iban',
      'be.vat',
      'li.peid',
      'lu.tva'],
    'dv': [],
    'dz': [],
    'el': ['gr.vat', 'gr.amka', 'cy.vat'],
    'en': ['au.abn',
      'au.acn',
      'au.tfn',
      'ca.sin',
      'ca.bn',
      'gb.upn',
      'gb.vat',
      'gb.nhs',
      'gb.utr',
      'gb.sedol',
      'ie.vat',
      'ie.pps',
      'in_.epic',
      'in_.gstin',
      'in_.pan',
      'in_.aadhaar',
      'nz.bankaccount',
      'nz.ird',
      'us.ssn',
      'us.atin',
      'us.rtn',
      'us.tin',
      'us.ein',
      'us.itin',
      'us.ptin',
      'mt.vat',
      'mu.nid',
      'sg.uen',
      'za.tin',
      'za.idnr'],
    'es': ['ar.dni',
      'ar.cbu',
      'ar.cuit',
      'es.iban',
      'es.ccc',
      'es.cif',
      'es.dni',
      'es.cups',
      'es.referenciacatastral',
      'es.nie',
      'es.nif',
      'mx.curp',
      'mx.rfc',
      'cl.rut',
      'co.nit',
      'cr.cpj',
      'cr.cr',
      'cr.cpf',
      'cu.ni',
      'do.rnc',
      'do.ncf',
      'do.cedula',
      'ec.ruc',
      'ec.ci',
      'gt.nit',
      'pe.cui',
      'pe.ruc',
      'py.ruc',
      'sv.nit',
      'uy.rut',
      've.rif'],
    'et': ['ee.ik', 'ee.registrikood', 'ee.kmkr'],
    'fa': [],
    'fi': ['fi.associationid',
      'fi.veronumero',
      'fi.hetu',
      'fi.ytunnus',
      'fi.alv'],
    'fil': [],
    'fj': [],
    'fo': [],
    'fr': ['ca.sin',
      'ca.bn',
      'ch.ssn',
      'ch.vat',
      'ch.uid',
      'ch.esr',
      'fr.siret',
      'fr.tva',
      'fr.nir',
      'fr.nif',
      'fr.siren',
      'be.iban',
      'be.vat',
      'lu.tva',
      'mc.tva',
      'mu.nid'],
    'ga': ['ie.vat', 'ie.pps'],
    'gil': [],
    'gn': ['py.ruc'],
    'gsw': ['ch.ssn', 'ch.vat', 'ch.uid', 'ch.esr', 'li.peid'],
    'gv': [],
    'he': ['il.hp', 'il.idnr'],
    'hi': ['in_.epic', 'in_.gstin', 'in_.pan', 'in_.aadhaar'],
    'hif': [],
    'ho': [],
    'hr': ['hr.oib'],
    'ht': [],
    'hu': ['hu.anum'],
    'hy': [],
    'id': ['id.npwp'],
    'is': ['is_.kennitala', 'is_.vsk'],
    'it': ['ch.ssn',
      'ch.vat',
      'ch.uid',
      'ch.esr',
      'it.codicefiscale',
      'it.aic',
      'it.iva',
      'sm.coe'],
    'ja': ['jp.cn'],
    'ka': [],
    'kk': [],
    'kl': [],
    'km': [],
    'ko': ['kr.rrn', 'kr.brn'],
    'ky': [],
    'lb': ['lu.tva'],
    'lo': [],
    'lt': ['lt.pvm', 'lt.asmens'],
    'lv': ['lv.pvn'],
    'mg': [],
    'mh': [],
    'mi': ['nz.bankaccount', 'nz.ird'],
    'mk': [],
    'mn': [],
    'ms': ['my.nric', 'sg.uen'],
    'mt': ['mt.vat'],
    'my': [],
    'na': [],
    'nb': ['no.mva', 'no.iban', 'no.kontonr', 'no.fodselsnummer', 'no.orgnr'],
    'nd': [],
    'ne': [],
    'niu': [],
    'nl': ['nl.bsn',
      'nl.brin',
      'nl.onderwijsnummer',
      'nl.btw',
      'nl.postcode',
      'be.iban',
      'be.vat'],
    'nn': ['no.mva', 'no.iban', 'no.kontonr', 'no.fodselsnummer', 'no.orgnr'],
    'ny': [],
    'pap': [],
    'pau': [],
    'pl': ['pl.pesel', 'pl.nip', 'pl.regon'],
    'ps': [],
    'pt': ['br.cnpj', 'br.cpf', 'pt.cc', 'pt.nif'],
    'qu': ['ec.ruc', 'ec.ci', 'pe.cui', 'pe.ruc'],
    'rn': [],
    'ro': ['ro.onrc', 'ro.cui', 'ro.cf', 'ro.cnp', 'md.idno'],
    'ru': ['ru.inn', 'ua.rntrc', 'ua.edrpou', 'by.unp'],
    'rw': [],
    'sg': [],
    'si': [],
    'sk': ['sk.rc', 'sk.dph'],
    'sl': ['si.ddv'],
    'sm': [],
    'sn': [],
    'so': [],
    'sq': ['al.nipt'],
    'sr': ['me.iban', 'rs.pib'],
    'ss': [],
    'st': [],
    'sv': ['fi.associationid',
      'fi.veronumero',
      'fi.hetu',
      'fi.ytunnus',
      'fi.alv',
      'se.vat',
      'se.personnummer',
      'se.postnummer',
      'se.orgnr'],
    'sw': [],
    'ta': ['sg.uen'],
    'tet': [],
    'tg': [],
    'th': ['th.pin', 'th.tin', 'th.moa'],
    'ti': [],
    'tk': [],
    'tkl': [],
    'tn': [],
    'to': [],
    'tpi': [],
    'tr': ['tr.vkn', 'tr.tckimlik', 'cy.vat'],
    'tvl': [],
    'ty': [],
    'tzm': [],
    'uk': ['ua.rntrc', 'ua.edrpou'],
    'ur': [],
    'uz': [],
    'vi': ['vn.mst'],
    'wni': [],
    'wo': [],
    'yo': [],
    'zdj': [],
    'zh': ['cn.ric', 'cn.uscc', 'tw.ubn', 'sg.uen'],
    'default':['bic',
      'bitcoin',
      'casrn',
      'cusip',
      'ean',
      'figi',
      'grid',
      'gs1_128',
      'iban',
      'imei',
      'imo',
      'imsi',
      'isan',
      'isbn',
      'isil',
      'isin',
      'ismn',
      'iso11649',
      'iso6346',
      'isrc',
      'issn',
      'lei',
      'mac',
      'meid',
      'vatin']
}

def ent_2_stdnum_type(text, src_lang=None):
  """ given a entity mention and the src_lang, determine potentially stdnum type """
  stdnum_type = []
  if src_lang is None:
    items = list(stdnum_mapper.items())
  else:
    l1 =  lang_2_stdnum.get(src_lang, []) + lang_2_stdnum.get('default', [])
    items = [(a1, stdnum_mapper[a1]) for a1 in l1]

  for ent_type, validate in items:
    try:
      found = validate(text)
    except:
      found = False
    if found:
      stdnum_type.append (ent_type)
  return stdnum_type

lstrip_chars = " ,،、<>{}[]|()\"'""《》«»:;"
rstrip_chars = " ,،、<>{}[]|()\"'""《》«»!:;?。.…．"
date_parser_lang_mapper = {'st': 'en', 'ny': 'en', 'xh': 'en', 'tt': 'en'}


def test_is_date(ent, tag, sentence, len_sentence, is_cjk, i, src_lang, sw, year_start=1600, year_end=2050):
    """
    Helper function used to test if an ent is a date or not
    We use dateparse to find context words around the ID/date to determine if its a date or not.
    For example, 100 AD is a date, but 100 might not be.
    Input:
      :ent: an entity mention
      :tag: either ID or DATE
      :sentence: the context
      :is_cjk: if this is a Zh, Ja, Ko text
      :i: the position of ent in the sentence
     Returns:
        (ent, tag): potentially expanded ent, and the proper tag. 
        Could return a potentially expanded ent, and the proper tag. 
        Returns ent as None, if originally tagged as 'DATE' and it's not a DATE and we don't know what it is.
     
    """
    # perform some fast heuristics so we don't have to do dateparser
    len_ent = len(ent)
    if len_ent > 17 or (len_ent > 8 and to_int(ent)):
      if tag == 'DATE': 
        #this is a very long number and not a date
        return None, tag
      else:
        #no need to check the date
        return ent, tag 
        
    if not is_cjk:
      if i > 0 and sentence[i-1] not in lstrip_chars: 
        if tag == 'DATE': 
          return None, tag
        else:
          return ent, tag
      if i+len_ent < len_sentence - 1 and sentence[i+len_ent+1] not in rstrip_chars: 
        if tag == 'DATE': 
          return None, tag
        else:
          return ent, tag

    int_arr = [(e, to_int(e)) for e in ent.replace("/", "-").replace(" ","-").replace(".","-").split("-")]
    if is_fast_date(ent, int_arr): 
      #this is most likely a date
      return ent, 'DATE'

    for e, val in int_arr:
      if val is not None and len(e) > 8:
        if tag == 'DATE': 
          #this is a very long number and not a date
          return None, tag

    #test if this is a 4 digit year. we need to confirm it's a real date
    is_date = False
    is_4_digit_year = False
    if tag == 'DATE' and len_ent == 4:
      e = to_int(ent)
      is_4_digit_year = (e <= year_end and e >= year_start)
    
    #now do dateparser
    if not is_4_digit_year:
      try:
        is_date =  dateparser.parse(ent, languages=[date_parser_lang_mapper.get(src_lang,src_lang)]) # use src_lang to make it faster, languages=[src_lang])
      except:
        is_date =  dateparser.parse(ent, languages=["en"])  
    if (not is_date and tag == 'DATE') or (is_date and tag == 'ID'):
        j = i + len_ent
        #for speed we can just use these 6 windows to check for a date.
        #but for completeness we could check a sliding window. 
        #Maybe in some countries a year could
        #be in the middle of a date: Month Year Day
        ent_spans = [(-3,0), (-2, 0), (-1, 0), \
              (0, 3), (0, 2), (0, 1)]
        before = sentence[:i]
        after = sentence[j:]
        if before and not is_cjk and before[-1] not in lstrip_chars:
          is_date = False
        elif after and not is_cjk and after[0] not in rstrip_chars:
          is_date = False
        else:
          if  not is_cjk:
            before = before.split()
            after = after.split()
          len_after = len(after)
          len_before = len(before)
          for before_words, after_words in ent_spans:
            if after_words > len_after: continue
            if -before_words > len_before: continue 
            if before_words == 0: 
                before1 = []
            else:
                before1 = before[max(-len_before,before_words):]
            after1 = after[:min(len_after,after_words)]
            if is_cjk:
              ent2 = "".join(before1)+ent+"".join(after1)
            else:
              ent2 = " ".join(before1)+" "+ent+" "+" ".join(after1)
            if ent2.strip() == ent: continue
            try:
              is_date = dateparser.parse(ent2, languages=[date_parser_lang_mapper.get(src_lang,src_lang)])# use src_lang to make it faster, languages=[src_lang])
            except:
              is_date = dateparser.parse(ent, languages=["en"])
            if is_date:
              #sometimes dateparser says things like "in 2020" is a date, which it is
              #but we want to strip out the stopwords.
              if before1 and before1[-1].lower() in sw:
                before1 = before1[:-1]
              if after1 and after1[0].lower() in sw:
                after1 = after1[1:]
              if is_cjk:
                ent2 = "".join(before1)+ent+"".join(after1)
              else:
                ent2 = " ".join(before1)+" "+ent+" "+" ".join(after1)
              ent = ent2.strip()
              tag = "DATE"
              return ent, tag

    if tag == 'DATE' and not is_date:
      return None, tag

    return ent, tag

def to_int(s):
  try:
    return int(s)
  except:
    return None

def is_fast_date(ent, int_arr=None, year_start=1600, year_end=2050):
  """search for patterns like, yyyy-mm-dd, dd-mm-yyyy, yyyy-yyyy """
  if int_arr:
    len_int_arr = len(int_arr)
    if len_int_arr == 1 or len_int_arr > 3: return False
  if int_arr is None:
    ent_arr = ent.replace("/", "-").replace(" ","-").replace(".","-")
    if not ("-" in ent_arr and ent_arr.count("-") <=2): return False
    int_arr = [(e, to_int(e)) for e in ent_arr.split("-")]
  is_date = False
  has_year = has_month = has_day = 0
  for e, val in int_arr:
    if val is None: 
      break
    if (val <= year_end and val >= year_start):
      has_year +=1
    elif val <= 12 and val >= 1:
      has_month += 1
    elif val <= 31 and val >= 1:
      has_day += 1
    else:
      return False
  if (has_year == 1 and has_month == 1) or \
        (has_year == 2 and has_month == 0 and has_day == 0) or \
        (has_year == 1 and has_month == 1 and has_day == 1):
      return True
  return False

#cusip number probaly PII?
def detect_ner_with_regex_and_context(sentence, src_lang,  tag_type= None, prioritize_lang_match_over_ignore=True, \
      ignore_stdnum_type={'isil', 'isbn', 'isan', 'imo', 'gs1_128', 'grid', 'figi', 'ean', 'casrn', 'cusip' }, \
      all_regex=None, context_window=20, min_id_length=6, max_id_length=50, \
      precedence={'PHONE':1, 'IP_ADDRESS':2, 'DATE':3, 'TIME':4, 'LICENSE_PLATE':5, 'USER':6, 'AGE':7, 'ID':8, 'KEY': 9,  'ADDRESS':10, 'URL':11, 'EQUATION': 12}):
      """
      Output:
       - This function returns a list of 4 tuples, representing an NER detection for [(entity, start, end, tag), ...]
      Input:
       :sentence: any text, including a sentence or a document to tag
       :src_lang: the language of the sentence
       :context_window: the contxt window in characters to check for context characters for any rules that requries context
       :max_id_length: the maximum length of an ID
       :min_id_length: the minimum length of an ID
       :tag_type: the type of NER tags we are detecting. If None, then detect everything.
       :ignore_stdnum_type: the set of stdnum we will consider NOT PII and not match as an ID
       :prioritize_lang_match_over_ignore: if true, and an ID matches an ingore list, we still keep it as an ID if there was an ID match for this particular src_lang
       :all_regex: a rulebase of the form {tag: {lang: [(regex, context, block), ...], 'default': [(regex, context, block), ...]}}. 
         context are words that must be found surronding the entity. block are words that must not be found.
         If all_regex is none, then we use the global regex_rulebase
       
      ALGORITHM:
        For each regex, we check the sentence to find a match and a required context, if the context exists in a window.
        If the regex is an ID or a DATE, test to see if it's a stdnum we know. Stdnum are numbers formatted to specific regions, or generally.
        If it is a stdnum and NOT a PII type (such as ISBN numbers) skip this ID.
          UNLESS If the stdnum is ALSO a PII type for the local region of the language, then consider it a matched stdnum.
        If it's a matched stdnum that is not skipped, save it as an ID.
        If the ID is not a stdum, check if the ID is a DATE. If it's a DATE using context words in a context window. 
          If it's a DATE then save it as a DATE, else save as ID.
        Gather all regex matches and sort the list by position, prefering longer matches, and DATEs and ADDRESSES over IDs.
        For all subsumed IDs and DATEs, remove those subsumed items. 
        Return a list of potentially overlapping NER matched.
      NOTE: 
      - There may be overlaps in mention spans. 
      - Unlike presidio, we require that a context be met. We don't increase a score if a context is matched.  
      - A regex does not need to match string boundaries or space boundaries. The matching code checks this. 
          We require all entities that is not cjk to have space or special char boundaries or boundaries at end or begining of sentence.
      - As such, We don't match embedded IDs: e.g., MyIDis555-555-5555 won't match the ID. This is to preven
        matching extremely nosiy imput that might have patterns of numbers in long strings.
      
      """

      sw = all_stopwords.get(src_lang, {})
      
      # if we are doing 'ID', we would still want to see if we catch an ADDRESS. 
      # ADDRESS may have higher precedence, in which case it might overide an ID match. 
      no_address = False
      if tag_type is not None and 'ID' in tag_type and 'ADDRESS' not in tag_type:
         no_address = True
         tag_type = set(list(tag_type)+['ADDRESS'])
         
      # if we are doing 'DATE' we would still want to do ID because they intersect.
      no_id = False
      if tag_type is not None and 'DATE' in tag_type and 'ID' not in tag_type:
         no_id = True
         tag_type = set(list(tag_type)+['ID'])

      # if we are doing 'AGE' we would still want to do DATE because they intersect.
      no_date = False
      if tag_type is not None and 'AGE' in tag_type and 'DATE' not in tag_type:
         no_date = True
         tag_type = set(list(tag_type)+['DATE'])
        

      is_cjk = src_lang in {'zh', 'zh-classical', 'zh-min-nan', 'zh-yue', 'ko', 'ja', 'th', 'jv'} 
      if is_cjk:
          sentence_set = set(sentence.lower())
      else:
          sentence_set = []
          #let's do a sanity check. there should be no words beyond 100 chars.
          #this will really mess up our regexes.
          for word in sentence.split(" "):
            len_word = len(word)
            if len_word > 100:
              sentence = sentence.replace(word, " "*len_word)
            else:
              sentence_set.append(word.lower())
          sentence_set = set([s.strip(rstrip_chars) for s in sentence_set])
      all_ner = []
      len_sentence = len(sentence)
        
      if all_regex is None:
        all_regex = regex_rulebase
      if tag_type is None:
        all_tags_to_check = list(all_regex.keys())
      else:
        all_tags_to_check = list(tag_type) 

      for tag in all_tags_to_check:
          regex_group = all_regex.get(tag)
          if not regex_group: continue
          for regex_context, extra_weight in [(a, 1) for a in regex_group.get(src_lang, [])] + [(a, 0) for a in regex_group.get("default", [])]:
              if True:
                  regex, context, block = regex_context
                  #if this regex rule requires a context, find if it is satisified in general. this is a quick check.
                  potential_context = False
                  if context:
                      for c1 in context:
                        c1 = c1.lower()
                        for c2 in c1.split():
                          c2 = c2.strip(rstrip_chars)
                          if c2 in sentence_set:
                              potential_context = True
                              break
                        if potential_context: break
                      if not potential_context:
                          continue
                  #now apply regex
                  for ent in list(set(list(regex.findall(sentence)))):
                      #print (ent)
                      if not isinstance(ent, str):
                        continue
                      ent = ent.strip()
                      #ent = ent.rstrip(rstrip_chars)
                      #ent = ent.lstrip(lstrip_chars)
                      if not ent:
                        continue
 
                      ent_is_4_digit=False
                      len_ent = len(ent)
                      if len_ent == 4:
                        try:
                          int(ent)
                          ent_is_4_digit=True
                        except:
                          ent_is_4_digit=False
                      sentence2 = sentence
                      delta = 0
                      #check to see if the ID or DATE is type of stdnum
                      is_stdnum = False
                      if tag in ('ID', 'DATE'):
                          #simple length test
                          ent_no_space = ent.replace(" ", "").replace(".", "").replace("-", "")
                          if len(ent_no_space) > max_id_length and tag == 'ID': continue
                          if len(ent_no_space) < min_id_length and tag == 'ID': continue
                            
                          #check if this is really a non PII stdnum, unless it's specifically an ID for a country using this src_lang. 
                          #TODO - complete the country to src_lang dict above. 
                          stnum_type = ent_2_stdnum_type(ent, src_lang)
                          
                          #if the stdnum is one of the non PII types, we will ignore it
                          if prioritize_lang_match_over_ignore:
                                is_stdnum = any(a for a in stnum_type if "." in a and src_lang in country_2_lang.get(a.split(".")[0], []))
                          if not ent_is_4_digit and not is_stdnum and any(a for a in stnum_type if a in ignore_stdnum_type):
                            #a four digit entity might be a year, so don't skip this ent
                            continue
                          #this is actually an ID of known stdnum and not a DATE
                          if any(a for a in stnum_type if a not in ignore_stdnum_type):
                            tag = 'ID'
                            is_stdnum = True

                      #let's check the FIRST instance of this DATE or ID is really a date; 
                      #ideally we should do this for every instance of this ID
                      if tag == 'DATE' or (tag == 'ID' and not is_stdnum):
                        ent, tag = test_is_date(ent, tag, sentence, len_sentence, is_cjk, sentence.index(ent),  src_lang, sw)
                        if not ent: continue
                        
                      #do some confirmation for addresses if libpostal is installed. TODO, test if this works for zh. libpostal appears to test for pinyin.
                      if tag == 'ADDRESS' and not parse_address: continue
                      if tag == 'ADDRESS' and parse_address:
                        address = parse_address(ent)
                        
                        if address and not any(ad for ad in address if ad[1] != 'house'):
                          continue # this isn't an address

                        if address and address[0][1] == 'house':
                          address = address[1:]

                        ent_lower = ent.lower()
                        if address[0][0].lower() in ent_lower:
                              ent = ent[ent_lower.index(address[0][0]):].strip(rstrip_chars)
                              #print ('**', ent)
                              if not ent or to_int (ent) is not None:
                                continue # this isn't an address
                              #TODO strip stopwords on either end of an ent for addresses - whether or not libpostal is installed
                        else:
                          pass
                          #print ('problem with address', address)
                        #print ('parse address', ent, '***', address)
                      #now let's check context, block lists and turn all occurances of ent in this sentence into a span mention and also check for context and block words
                      len_ent = len(ent)
                      while True:
                        if not ent or ent not in sentence2:
                          break
                        else:
                          i = sentence2.index(ent)
                          j = i + len_ent
                          if potential_context or block:
                              len_sentence2 = len(sentence2)
                              left = " "+ sentence2[max(0, i - context_window) : i].replace(",", " ").lower()+ " "
                              right = " "+ sentence2[j : min(len_sentence2, j + context_window)].replace(",", " ").lower() + " "
                              found_context = False
                              ent_lower = " "+ent.replace(",", " ").lower()+ " "
                              if context:
                                for c in context:
                                  c = c.lower()
                                  if is_cjk:
                                      if c in left or c in right or c in ent_lower:
                                          found_context = True
                                          break
                                  else:
                                      if (" "+c+" " in left or " "+c+" " in right or " "+c+" " in ent_lower):
                                          found_context = True
                                          #print ('foound context', c)
                                          break
                              else:
                                found_context = True
                              if block:
                                for c in block:
                                  new_tag = None
                                  if type(c) is tuple:
                                    c, new_tag = c
                                  c = c.lower()
                                  if is_cjk:
                                      if c in left or c in right or c in ent_lower:
                                          if new_tag is not None:
                                            tag = new_tag #switching the tag to a subsumed tag. DATE=>AGE
                                            break
                                          else:
                                            found_context = False
                                            break
                                  else:
                                      if (" "+c+" " in left or " "+c+" " in right or " "+c+" " in ent_lower):
                                          if new_tag is not None:
                                            tag = new_tag #switching the tag to a subsumed tag. DATE=>AGE
                                            break
                                          else:
                                            found_context = False
                                            break                                    
                              if not found_context:
                                delta += j
                                sentence2 = sentence2[i+len(ent):]
                                continue
                          #check to see if the entity is really a standalone word or part of another longer word.
                          # for example, we wont match a partial set of very long numbers as a 7 digit ID for example
                          if is_cjk or ((i+delta == 0 or sentence2[i-1]  in lstrip_chars) and (j+delta >= len_sentence-1 or sentence2[j] in rstrip_chars)): 
                            all_ner.append((ent, delta+i, delta+j, tag, extra_weight))
                          sentence2 = sentence2[i+len(ent):]
                          delta += j
                            
      all_ner = list(set(all_ner))
      # let's remove overlapping 
      # sort by length and position, favoring non-IDs first using the precedence list, 
      # and additionaly giving one extra weight to language specific regex (as opposed to default rules).
      # NOTE: this doesn't do a perfect overlap match; just an overlap to the prior item.
      all_ner.sort(key=lambda a: a[1]+(1.0/(1.0+(100*((precedence.get(a[3], min(20,len(a[3])))+a[4]))+a[2]-a[1]))))
      #print (all_ner)
      if not tag_type or 'ID' in tag_type:
        # now do overlaps prefering longer ents, and higher prededence items over embedded IDs or dates, etc.
        all_ner2 = []
        prev_mention = None
        for mention in all_ner:
          if prev_mention:
            if (prev_mention[1] == mention[1] and prev_mention[3] == mention[3] and prev_mention[4] > 0 and prev_mention[4] != mention[4]) or\
              (prev_mention[2] >= mention[1] and prev_mention[2] >= mention[2]): 
                continue
            else:
              prev_mention = mention
          else:
            prev_mention = mention
          all_ner2.append(mention[:4])
        all_ner = all_ner2
      #TODO - refactor to check the tag_type list instead to do filtering.
      if no_address:
         all_ner = [a for a in all_ner if a[3] != 'ADDRESS']
      if no_id:
         all_ner = [a for a in all_ner if a[3] != 'ID']   
      if no_date:
         all_ner = [a for a in all_ner if a[3] != 'DATE']           
      return all_ner

