# riverbed
Tools for content datamining and NLP at scale.

## motiviation
Given a set of image data and/or text data in human language, or code, we would like to:
- create interesting features for the data, including augmenting the data.
- cluster, explore, search and visualize the data.
- label and create classifiers for the data. 
- search, filter, store and share the data. 

The data may be domain specific and may change over time. If a change is significant, we would like to be notified of the changes and/or automatically 
re-run some or all of the above.

## installation

```
git clone https://github.com/ontocord/riverbed/
chmod ugo+x /content/riverbed/bin/lmplz
pip install https://github.com/kpu/kenlm/archive/master.zip
pip install dataset datasets fasttext indexed_gzip whoosh transformers sentencepiece spacy nltk fast-pytorch-kmeans mmh3 tqdm
pip install tsnecuda==3.0.1+cu112 -f https://tsnecuda.isx.ai/tsnecuda_stable.html
python -m spacy download en_core_web_md
python -m nltk.downloader stopwords
```
