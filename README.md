# riverbed
Tools for content datamining and NLP at scale.

## motiviation
Given a set of text content in human language, or code, we would like to:
- Filter for quality, NSFW and potential illegal text
- create interesting features for the content, including augmenting the content.
- cluster, explore, search and visualize the content.
- label and create classifiers for the content. 
- search, store and share the content to user and to other AI models 


## installation

```
git clone https://github.com/ontocord/riverbed/
chmod ugo+x /content/riverbed/bin/lmplz
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install dataset datasets fasttext indexed_gzip whoosh transformers sentencepiece spacy nltk fast-pytorch-kmeans mmh3 tqdm
git clone --recursive https://github.com/seomoz/simhash-py
rm simhash-py/simash/*.cpp
python simhash-py/setup.py install build_ext --inplace
pip install tsnecuda==3.0.1+cu112 -f https://tsnecuda.isx.ai/tsnecuda_stable.html
python -m spacy download en_core_web_md
python -m nltk.downloader stopwords
```

## history

Originally written by Ontocord, LLC. Donated to LAION for the open source community.
