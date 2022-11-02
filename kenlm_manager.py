#@title KenLM code
"""
Copyright, 2021-2022 Ontocord, LLC, and other authors of Muliwai, All rights reserved.
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
# from https://github.com/piisa/muliwai/blob/main/kenlm_manager.py
# which is based Eduardo Gonzalez Ponferrada/edugp's repo: https://huggingface.co/edugp/kenlm/blob/main/model.py which is under the Apache 2 License
# which is also based on https://github.com/facebookresearch/cc_net/ which is under the MIT License
# thank you edugp!!
import os
import re
import unicodedata
from typing import Dict
import warnings
import kenlm
import sentencepiece
from huggingface_hub import cached_download, hf_hub_url
from filelock import FileLock

## additional code to support kenlm entity querying
kenlm_models = {
    'wikipedia': {},
    'oscar': {},
    'mc4': {},
}

#NOTE: If you want to use the default cc_net kenlm wikipedia models, you will need to download them. You can manually download per the below or copy them from a saved dir.
#Alternately, they will be downloaded automatically using the load_kenlm_model function.

#see https://github.com/facebookresearch/cc_net/blob/main/Makefile. These are default models if there aren't any from edugp
ccnet_langs=set("af,ar,az,be,bg,bn,ca,cs,da,de,el,en,es,et,fa,fi,fr,gu,he,hi,hr,hu,hy,id,is,it,ja,ka,kk,km,kn,ko,lt,lv,mk,ml,mn,mr,my,ne,nl,no,pl,pt,ro,ru,uk,zh".split(","))
def download_ccnet_sp_kenlm_models(lang, default_kenlm_wikipedia="./kenlm_ccnet_wikipedia_models"):
  if not os.path.exists(f"{default_kenlm_wikipedia}/{lang}.arpa.bin"):
    with FileLock(f"{default_kenlm_wikipedia}/{lang}.arpa.bin.lock"):
        os.system(f"wget -c  -P {default_kenlm_wikipedia} http://dl.fbaipublicfiles.com/cc_net/lm/{lang}.arpa.bin")
  if not os.path.exists(f"{default_kenlm_wikipedia}/{lang}.sp.model"):
    with FileLock(f"{default_kenlm_wikipedia}/{lang}.sp.model.lock"):
        os.system(f"wget -c  -P {default_kenlm_wikipedia} http://dl.fbaipublicfiles.com/cc_net/lm/{lang}.sp.model")

def get_kenlm_models_from_savedir( default_kenlm_wikipedia="./kenlm_ccnet_wikipedia_models", save_dir="/content/drive/Shareddrives/LAION/kenlm_ccnet_wikipedia_models"):
  if not os.path.exists(default_kenlm_wikipedia):
    with FileLock(f"{default_kenlm_wikipedia}.lock"):
        os.system(f"cp -rf {save_dir} {default_kenlm_wikipedia}")

# WOULD be good if we can create models based on cc100.

# TODO figure out actual numbers. Also, add languge specific kenlm models. Check if there are variations b/c of
#  gender, so we would have at least two patterns.
public_figure_kenlm_cutoff_map = {
    'en': {'wikipedia': [{'cutoff': 500, 'pattern': "{} (born"}],  # in wikipedia, you often have: Lincoln (born .... )
           'oscar': [{'cutoff': 500, 'pattern': "{} was born"}],
           },
    'yo': {'wikipedia': [{'cutoff': 400, 'pattern': "{} ni a bi lori"}],
           'oscar': [{'cutoff': 400, 'pattern': "{} ni a bi lori"}],
           },
    'zu': {'wikipedia': [{'cutoff': 400, 'pattern': "{} wazalwa ngo"}],
           'oscar': [{'cutoff': 400, 'pattern': "{} wazalwa ngo"}],
           'mc4': [{'cutoff': 400, 'pattern': "{} wazalwa ngo"}],  # for now, we are using the mc4 model for zu and ig
           },
    'sn': {'wikipedia': [{'cutoff': 500, 'pattern': "{} akazvarwa"}],
           'oscar': [{'cutoff': 500, 'pattern': "{} akazvarwa"}],
           },
    'st': {'wikipedia': [{'cutoff': 500, 'pattern': "{} o hlahile ka"}],
           'oscar': [{'cutoff': 500, 'pattern': "{} o hlahile ka"}],
           },
    'ny': {'wikipedia': [{'cutoff': 500, 'pattern': "{} anabadwa pa"}],
           'oscar': [{'cutoff': 500, 'pattern': "{} anabadwa pa"}],
           },
    'xh': {'wikipedia': [{'cutoff': 500, 'pattern': "{} wazalwa ngo"}],
           'oscar': [{'cutoff': 500, 'pattern': "{} wazalwa ngo"}],
           },
    'sw': {'wikipedia': [{'cutoff': 500, 'pattern': "{} alizaliwa tarehe"}],
           'oscar': [{'cutoff': 500, 'pattern': "{} alizaliwa tarehe"}],
           },
    'ig': {'wikipedia': [{'cutoff': 300, 'pattern': "{} amụrụ"}],
           'oscar': [{'cutoff': 300, 'pattern': "{} amụrụ"}],
           'mc4': [{'cutoff': 300, 'pattern': "{} amụrụ"}],
           },
    'ar': {'wikipedia': [{'cutoff': 600, 'pattern': "ولد {} من"}],
           'oscar': [{'cutoff': 600, 'pattern': "ولد {} من"}]
           },
    'zh': {'wikipedia': [{'cutoff': 500, 'pattern': "{}生於"}],
           'oscar': [{'cutoff': 500, 'pattern': "{}生於"}]
           },
    'vi': {'wikipedia': [{'cutoff': 500, 'pattern': "{} sinh ra"},
                         {'cutoff': 500, 'pattern': "{} sáng lập"}],
           'oscar': [{'cutoff': 450, 'pattern': "{} sinh ra"},
                     {'cutoff': 450, 'pattern': "{} sáng lập"}],
           },
    'hi': {'wikipedia': [{'cutoff': 500, 'pattern': "{} का जन्म ए"}],
           'oscar': [{'cutoff': 500, 'pattern': "{} का जन्म ए"}],
           },
    'ur': {'wikipedia': [{'cutoff': 500, 'pattern': "{} پیدا ہوا"}],
           'oscar': [{'cutoff': 500, 'pattern': "{} پیدا ہوا"}],
           },
    'id': {'wikipedia': [{'cutoff': 500, 'pattern': "{} lahir"}],
           'oscar': [{'cutoff': 500, 'pattern': "{} lahir"}],
           },
    'bn': {'wikipedia': [{'cutoff': 500, 'pattern': "{} জন্ম"}],
           'oscar': [{'cutoff': 500, 'pattern': "{} জন্ম"}],
           }
}

# TODO: Instead of defaulting to the ccnet models, we will want to pick and choose from the ccnet/edugp wikipedia model
def load_kenlm_model(
        src_lang: str = "en",
        pretrained_models: list = ['wikipedia'],
        store_model: bool = True,
        cache_dir: str = "./kenlm_edugp_models",
        default_kenlm_wikipedia: str = "./kenlm_ccnet_wikipedia_models"
) -> dict:
    """
    Load all supported kenlm model for source language. Consider if we want to use an LRU.
    TODO: Incorporate OSCAR kenlm models. They are quite big, and we still need patterns and cutoffs.
    """
    assert len(pretrained_models) <= len(
        kenlm_models), 'Total of number kenlm models loads larger than supported kenlm models'
    src_lang = src_lang if src_lang in public_figure_kenlm_cutoff_map else "en"
    all_models = {}
    model_files = ["arpa.bin", "sp.model", ] # "sp.vocab"
    # cache to dir
    if cache_dir is None:
        cache_dir = os.path.expanduser('~') + "/.cache"
    
    # check if pretrain model exist
    for model_type in pretrained_models:
        if src_lang in kenlm_models[model_type]:
            all_models[model_type] = kenlm_models[model_type][src_lang]
        elif model_type == "wikipedia" and os.path.exists(f"{default_kenlm_wikipedia}/{src_lang}.arpa.bin"):
            model = KenlmModel(default_kenlm_wikipedia, src_lang, do_normalize_spacing_for_tok=True)
            all_models[model_type] = model
            if store_model:
              kenlm_models[model_type][src_lang] = model
        elif model_type == "wikipedia" and src_lang in ccnet_langs:
            download_ccnet_sp_kenlm_models(src_lang, default_kenlm_wikipedia)
            model = KenlmModel(default_kenlm_wikipedia, src_lang, do_normalize_spacing_for_tok=True)
            all_models[model_type] = model
            if store_model:
              kenlm_models[model_type][src_lang] = model
        elif model_type not in kenlm_models.keys():
            warnings.warn(f"{model_type} pretrained model is not supported!")
        else:
            os.system(f"mkdir -p {cache_dir}/{model_type}")
            found = True
            for model_file in model_files:
                if not os.path.exists(f"{cache_dir}/{model_type}/{src_lang}.{model_file}"):
                    try:
                        file_url = hf_hub_url(repo_id="edugp/kenlm",
                                              filename=f"{model_type}/{src_lang}.{model_file}")
                        file = cached_download(file_url)
                        os.system(f"ln -s {file} {cache_dir}/{model_type}/{src_lang}.{model_file}")
                    except:
                        warnings.warn(f'could not find model {src_lang}.{model_file}. will stop searching...')
                        found = False
                        break
            if found:
                model = KenlmModel(f"{cache_dir}/{model_type}", src_lang)
                all_models[model_type] = model
                if store_model:
                    kenlm_models[model_type][src_lang] = model
    return all_models


# TODO: refactor code in the faker_extensions with this code
def check_for_common_name(
        src_lang: str = "en",
        pretrained_models: list = ['wikipedia'],
        name: str = None,
        verbose: bool = False,
        kenlm_models=None,
        return_score=False,
):
    """
    Check if a name is a public figure or a very common name
    """
    # load all kenlm models and cutoff patterns
    if kenlm_models is None:
        kenlm_models = load_kenlm_model(src_lang, pretrained_models)
    public_patterns = public_figure_kenlm_cutoff_map.get(src_lang, public_figure_kenlm_cutoff_map.get('en'))
    for model_type, model in kenlm_models.items():
        for pattern in public_patterns.get(model_type, public_patterns.get('wikipedia')):
            test_name = pattern['pattern'].format(name)
            score = model.get_perplexity(test_name)
            if score < pattern['cutoff']:
                #if verbose:
                #    print(name, score)
                if return_score:
                    return True, score, pattern['cutoff']
                return True
    if return_score:
        return False, 0.0, 0.0
    return False


### Edugp code

class SentencePiece:
    def __init__(
            self,
            model: str,
    ):
        super().__init__()
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.load(str(model))

    def do(self, text: dict) -> dict:
        tokenized = self.sp.encode_as_pieces(text)
        return " ".join(tokenized)

#see https://github.com/facebookresearch/cc_net/blob/main/cc_net/text_normalizer.py
class KenlmModel:
    digit_re: re.Pattern = re.compile(r"\d")
    unicode_punct: Dict[str, str] = {
        "，": ",",
        "。": ".",
        "、": ",",
        "„": '"',
        "”": '"',
        "“": '"',
        "«": '"',
        "»": '"',
        "１": '"',
        "」": '"',
        "「": '"',
        "《": '"',
        "》": '"',
        "´": "'",
        "∶": ":",
        "：": ":",
        "？": "?",
        "！": "!",
        "（": "(",
        "）": ")",
        "；": ";",
        "–": "-",
        "—": " - ",
        "．": ". ",
        "～": "~",
        "’": "'",
        "…": "...",
        "━": "-",
        "〈": "<",
        "〉": ">",
        "【": "[",
        "】": "]",
        "％": "%",
        "►": "-",
    }
    unicode_punct_re = re.compile(f"[{''.join(unicode_punct.keys())}]")
    non_printing_chars_re = re.compile(
        f"[{''.join(map(chr, list(range(0, 32)) + list(range(127, 160))))}]"
    )
    kenlm_model_dir = None
    sentence_piece_model_dir = None

    # TODO: we are not doing the sacremoses tokenizer to get put spaces between escaped chars 
    # but consider whether we should do this for the ccnet models 
    # https://github.com/facebookresearch/cc_net/blob/bda555bd1cf1ee2e0b925363e62a61cd46c8b60d/cc_net/tokenizer.py#L23
    # does it make a difference?
    def __init__(
            self,
            model_dataset: str,
            language: str,
            lower_case: bool = False,
            remove_accents: bool = False,
            normalize_numbers: bool = True,
            punctuation: int = 1,
            do_normalize_spacing_for_tok: bool = False,
    ):
        self.model_dataset = model_dataset
        self.model = kenlm.Model(os.path.join(self.model_dataset, f"{language}.arpa.bin"))
        self.tokenizer = SentencePiece(os.path.join(self.model_dataset, f"{language}.sp.model"))
        self.do_normalize_spacing_for_tok = do_normalize_spacing_for_tok
        self.accent = remove_accents
        self.case = lower_case
        self.numbers = normalize_numbers
        self.punct = punctuation
        self.language = language

    @classmethod
    def from_pretrained(
            cls,
            model_dataset: str,
            language: str,
    ):
        return cls(
            model_dataset,
            language,
            False,
            language  in {"en", "my"},
            True,
            1,
        )

    def pp(self, log_score, length):
        return 10.0 ** (-log_score / length)

    def get_perplexity(self, doc: str, normalize_cc_net: bool = True):
        if normalize_cc_net:
            doc = self.normalize(
                doc,
                accent=self.accent,
                case=self.case,
                numbers=self.numbers,
                punct=self.punct,
            )
        # Tokenize (after normalizing): See https://github.com/facebookresearch/cc_net/blob/bda555bd1cf1ee2e0b925363e62a61cd46c8b60d/cc_net/mine.py#L352 for full pipeline
        if self.do_normalize_spacing_for_tok:
          doc = self.normalize_spacing_for_tok(doc)
        doc = self.tokenizer.do(doc)
        doc_log_score, doc_length = 0, 0
        for line in doc.split("\n"):
            log_score = self.model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
        return round(self.pp(doc_log_score, doc_length), 1)
    
    def normalize(
            self,
            line: str,
            accent: bool = True,
            case: bool = True,
            numbers: bool = True,
            punct: int = 1,
    ) -> str:
        line = line.strip()
        if not line:
            return line
        if case:
            line = line.lower()
        if accent:
            line = self.strip_accents(line)
        if numbers:
            line = self.digit_re.sub("0", line)
        if punct == 1:
            line = self.replace_unicode_punct(line)
        elif punct == 2:
            line = self.remove_unicode_punct(line)
        line = self.remove_non_printing_char(line)
        return line

    @staticmethod
    def normalize_spacing_for_tok(text: str, language: str = "en") -> str:
      res = (
          text.replace("\r", "")
          # remove extra spaces
          .replace("(", " (")
          .replace(")", ") ")
          .replace(" +", " ")
      )
      res = re.sub(r"\) ([\.\!\:\?\;\,])", r"\)\1", res)
      res = res.replace("( ", "(").replace(" )", ")")
      res = re.sub(r"(\d) \%", r"\1\%", res)
      res = res.replace(" :", ":").replace(" ;", ";")
      res = res.replace("`", "'").replace("''", ' " ')

      res = (
          res.replace("„", '"')
          .replace("“", '"')
          .replace("”", '"')
          .replace("–", "-")
          .replace("—", " - ")
          .replace(" +", " ")
          .replace("´", "'")
          .replace("([a-z])‘([a-z])", r"\1'\2/")
          .replace("([a-z])’([a-z])", r"\1'\2/")
          .replace("‘", '"')
          .replace("‚", '"')
          .replace("’", '"')
          .replace("''", '"')
          .replace("´´", '"')
          .replace("…", "...")
          # French quotes
          .replace(" « ", ' "')
          .replace("« ", '"')
          .replace("«", '"')
          .replace(" » ", '" ')
          .replace(" »", '"')
          .replace("»", '"')
          # handle pseudo-spaces
          .replace(" %", "%")
          .replace("nº ", "nº ")
          .replace(" :", ":")
          .replace(" ºC", " ºC")
          .replace(" cm", " cm")
          .replace(" ?", "?")
          .replace(" !", "!")
          .replace(" ;", ";")
          .replace(", ", ", ")
          .replace(" +", " ")
          .replace("．", ". ")
      )
      # English "quotation," followed by comma, style
      if language == "en":
          res = re.sub(r"\"([,\.]+)", r"\1\"", res)
      # Czech is confused
      elif language == "cs" or language == "cz":
          pass
      # German/Spanish/French "quotation", followed by comma, style
      else:
          res = res.replace(',"', '",')
          res = re.sub(
              r"(\.+)\"(\s*[^<])", r"\"\1\2", res
          )  # don't fix period at end of sentence

      if (
          language == "de"
          or language == "es"
          or language == "cz"
          or language == "cs"
          or language == "fr"
      ):
          res = re.sub(r"(\d) (\d)", r"\1,\2", res)
      else:
          res = re.sub(r"(\d) (\d)", r"\1.\2", res)
      return res

    @staticmethod
    def strip_accents(line: str) -> str:
        """Strips accents from a piece of text."""
        nfd = unicodedata.normalize("NFD", line)
        output = [c for c in nfd if unicodedata.category(c) != "Mn"]
        if len(output) == line:
            return line
        return "".join(output)

    def replace_unicode_punct(self, text: str) -> str:
        return "".join(self.unicode_punct.get(c, c) for c in text)

    def remove_unicode_punct(self, text: str) -> str:
        """More aggressive version of replace_unicode_punct but also faster."""
        return self.unicode_punct_re.sub("", text)

    def remove_non_printing_char(self, text: str) -> str:
        return self.non_printing_chars_re.sub("", text)

    def check_common_name(self, name: str, return_score: bool = False):
        """
        Check if a name is a common name.

        :param name: Name to check.
        :param return_score: If True, return the score of the name and cutoff threshold of the pattern.
        :return: True if name is a common name, False otherwise.
        """
        public_patterns = public_figure_kenlm_cutoff_map.get(self.language, public_figure_kenlm_cutoff_map.get('en'))
        model_type = self.model_dataset.split("/")[-1]
        for pattern in public_patterns.get(model_type, public_patterns.get('wikipedia')):
            test_name = pattern['pattern'].format(name)
            score = self.get_perplexity(test_name)
            if score < pattern['cutoff']:
                if return_score:
                    return True, score, pattern['cutoff']
                return True
        if return_score:
            return False, 0.0, 0.0
        return False
