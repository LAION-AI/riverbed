# riverbed
Tools for text datamining and NLP at scale

## installation


```
git clone https://github.com/ontocord/riverbed/
chmod ugo+x /content/riverbed/bin/lmplz
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install fasttext indexed_gzip whoosh transformers sentencepiece spacy nltk fast-pytorch-kmeans mmh3 tqdm
pip install tsnecuda==3.0.1+cu112 -f https://tsnecuda.isx.ai/tsnecuda_stable.html
python -m spacy download en_core_web_md
python -m nltk.downloader stopwords
```
