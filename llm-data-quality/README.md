# LLM Data Quality Pipeline (WIP)
## Motivation
To create a data quality framework and pipeline that could combine the best of everyone's code, while remains traceable and easily reproducible.

## Some principles
- extensible to different scorers/ filters/ clustering in whatever framework/ models, so everyone can contribute)
- reproducible and traceable via configuration management (which scorer, filter, clustering config)
- as part of experiment artifact for quicker iteration
- for meta analysis (like does diversity result in better model? does more reward model result in better model?)

## Proposed pipeline:
Config -> `ScorerPipeline` -> `FilterPipeline` (Removal) -> `ClusteringPipeline` -> Human QC/ label -> Some proxy data quality model training -> Good data -> Next Iteration

- Config so far only includes Filter, but that would extend to scorer/ clustering in the future.
- ScorePipeline involves a process of matching instruct+answer to a float, as such it includes reward model, perplexity, safety model, etc.
- FilterPipeline offers two approaches so far, based on absolute threshold, and zscore of scores.
- ClusteringPipeline involves any process involves pairwise comparison, including deduplication, semantic clustering, etc. This step involves sampling and removal of data point.

## Configuration Management
The pipeline can be fully initiated by a yaml as below.  
The pipeline will run each component in this order: `scorer`->`scorefilter` -> `clustering`  
- `scorer` and `clustering` contain a list of objects, which determines the order of execution.
`_impl__` corresponds to the class name in `implementation.py` of respective components, and other arguments refer to the arguments to instantiate that class.
- `scorefilter` is more statistic in nature as in practice a filtering is either represented as an absoluate or relative threshold.
  - `ge`: score greater than or equal to the threshold will pass the filter
  - `le`: score less than or equal to the threshold will pass the filter
  - list of conditions are perceived as `AND` logic. For example, if you set two conditions (x>=a & x<=b), effectively it will be a range: a <= x <= b

```shell
scorer:
- _impl_: PerplexityScorer
  score_id: perplexity
  model_id: gpt2
  batch_size: 8
  max_length: 1024
  device: -1
- _impl_: ToxicityScorer
  score_id: toxicity
  model_id: unitary/toxic-bert
  batch_size: 8
  max_length: 512
  device: -1
- _impl_: GibberishScorer
  score_id: gibberish
  model_id: madhurjindal/autonlp-Gibberish-Detector-492513457
  batch_size: 8
  max_length: 512
  device: -1
- _impl_: ContradictionScorer
  score_id: contradiction
  model_id: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
  batch_size: 8
  max_length: 512
  device: -1
- _impl_: LengthScorer
  score_id: length

scorefilter:
  absolute:
  - score_id: reward
    threshold: 0.5
    direction: ge
  - score_id: toxicity
    threshold: 0.1
    direction: le
  - score_id: gibberish
    threshold: 0.1
    direction: le
  - score_id: contradiction
    threshold: 0.5
    direction: le
  - score_id: length
    threshold: 2000
    direction: le
  - score_id: length
    threshold: 10
    direction: ge
  relative:
  - score_id: perplexity
    threshold: 1.65
    direction: le
  - score_id: perplexity
    threshold: -1.65
    direction: ge

clustering:
- _impl_: Dedup
- _impl_: SemanticKmeansClustering
  batch_size: 8
  device: -1
  model_id: facebook/contriever
  n_cluster: 100
  niter: 10
  sample_rate: 0.1
```

## How to run
### 1. as a script
```shell
python main.py test_data/test_confi_gpu.yaml test_data/test_data.json test_data
```
`test_data/test_data.json` is expected to be a list of [{"instruct": "How are you?", "answer": "I am good."}]

The filtered data and the removed data are saved for analysis/ quality check/ develop of quality classifier.

### 2. in Python
#### running whole pipeline
```python
import yaml
from datasets import load_dataset
from llmdq import Config, llmdq_pipeline


with open('test_data/test_config_gpu.yaml') as f:
    config = Config(**yaml.safe_load(f))

dataset = load_dataset('marianna13/random_dataset')
dataset = dataset['train'].rename_columns({"question": "instruct"})
clustered_data, removed_dataset = llmdq_pipeline(dataset, config)
```
#### running individual component
```python
from datasets import load_dataset
from llmdq.clustering import ClusteringPipeline, SemanticKmeansClustering



dataset = load_dataset('marianna13/random_dataset')
dataset = dataset['train'].rename_columns({"question": "instruct"})


clusteringpipe = ClusteringPipeline()
clusteringpipe.add([
    SemanticKmeansClustering("facebook/contriever",
                              batch_size=32,
                              device=0,
                              n_cluster=100,
                              sample_rate=0.1)
])
clustered_data = clusteringpipe.run(dataset)
```

An example can be found in [colab](https://colab.research.google.com/drive/1zGvPjHXDQiGq1c9SIYS_tZAzcFIWOICj?usp=sharing&authuser=2#scrollTo=tUEPRDMoH6Bi).


## How to add your module:
Inherit abstract class and add your implementation in `implementation.py`. Follow the signature to avoid break.


## TODO
- a lot of implementations
- performance optimisation (multiprocessing/ offload to GPU)
- more validation in initiation
- visualisation module

## Performance Optimisation (TODO)
Stage  | Time Complexity  (N=dataset size) |  Performance Optimisers
--- | --- | --- 
Scoring | O(N * n_score) | <ul><li>Single GPU settings: Async programming won’t help as it’s GPU-bounded</li><li>Multiple GPU settings: Async programming to dispatch different scorers to multiple GPUs (ie. we treat individual GPU as a webserver https://huggingface.co/docs/transformers/pipeline_webserver)
Filter | O(N), atomic operation is minimal | Multiprocessing for very large data
Clustering |O(N) + O(N * n_cluster)| <ul><li>Single GPU settings: GPU is used for clustering, use of half precision</li><li>Multiple GPU settings: For very large data, agglomerative clustering is required to prevent GPU OOM. Batches of data are fed into multiple GPUs for clustering. Then, clusters from multiple batches are merged based on centroids.  ANN search is then performed afterwards to assign clusters.

