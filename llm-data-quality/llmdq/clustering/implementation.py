from tqdm import tqdm
from typing import Iterable
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModel
import faiss
from datasets import Dataset
from llmdq.clustering.base import ClusteringBase


lg = logging.getLogger(__name__)


class Dedup(ClusteringBase):
    def run(self, instructanswer_dataset: Dataset) -> Dataset:
        """TODO"""
        return instructanswer_dataset


class SemanticKmeansClustering(ClusteringBase):
    """
    Implemented from: https://colab.research.google.com/drive/13eGPGqcHcfJQhqTgX-PnZ5C0Fkb8nVLJ?usp=sharing
    """
    def __init__(self, model_id, batch_size=8, device=-1, n_cluster=100, niter=10, sample_rate=0.1):
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id)
        self._batch_size = batch_size
        self._device = device
        self._device_pt = f"cuda:{self._device}" if self._device >= 0 else "cpu"
        self._model.to(self._device_pt)
        self._n_cluster = n_cluster
        self._niter = niter
        self._sampling_rate = sample_rate

    def _batching(self, iterable: list) -> Iterable:
        length = len(iterable)
        for ndx in range(0, length, self._batch_size):
            yield iterable[ndx:min(ndx + self._batch_size, length)]

    def _get_embedding(self, instructanswer_dataset: Dataset) -> np.ndarray:
        embed_list = []
        for d in tqdm(self._batching(instructanswer_dataset),
                      desc=self.__class__.__name__,
                      total=len(instructanswer_dataset)//self._batch_size+1):
            text = [i + "\n" + a for i, a in zip(d['instruct'], d['answer'])]
            inputs = self._tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self._device_pt)
            embed = self._model(**inputs)
            embed_list.append(embed.pooler_output.cpu().detach().numpy())  # use pooled output

        embed = np.vstack(embed_list)

        # normalised with l2-norm
        embed_l2 = np.atleast_1d(np.linalg.norm(embed, ord=2, axis=-1))
        embed_l2[embed_l2 == 0] = 1
        return embed / np.expand_dims(embed_l2, axis=-1)

    def _clustering(self, embeddings: np.ndarray) -> np.ndarray:
        kmeans = faiss.Kmeans(embeddings.shape[1], self._n_cluster, niter=self._niter,
                              gpu=True if self._device >= 0 else False)
        kmeans.train(embeddings)
        _, I = kmeans.index.search(embeddings, 1)
        return I.flatten()

    def _sampling(self, instructanswer_dataset: Dataset, member_list: np.ndarray) -> Dataset:
        """If a cluster size is lower than as if the cluster was following uniform distribution,
        the whole cluster is taken """
        targeted_sample_size = len(instructanswer_dataset) * self._sampling_rate // self._n_cluster
        sampled_index = set()
        for i in range(self._n_cluster):
            cluster_member = np.where(member_list == i)[0]
            if cluster_member.size <= targeted_sample_size:
                sampled_index.update(cluster_member.tolist())
            else:
                sampled_index.update(np.random.choice(cluster_member,
                                                      size=int(len(cluster_member) * self._sampling_rate),
                                                      replace=False).tolist())
        return instructanswer_dataset.select(list(sampled_index))

    def run(self, instructanswer_dataset: Dataset) -> Dataset:
        if len(instructanswer_dataset) <= self._n_cluster:
            lg.info(f"Data size smaller than or equal to {self._n_cluster}, cannot perform clustering")
            return instructanswer_dataset
        embeddings = self._get_embedding(instructanswer_dataset)
        member_list = self._clustering(embeddings)
        sampled_instructanswer_dataset = self._sampling(instructanswer_dataset, member_list)
        return sampled_instructanswer_dataset


class ClusteringPipeline(ClusteringBase):
    def __init__(self):
        self._clustering_list = []

    def add(self, clustering_list) -> None:
        self._clustering_list.extend(clustering_list)

    def run(self, instructanswer_dataset: Dataset) -> Dataset:
        for clust in self._clustering_list:
            instructanswer_dataset = clust.run(instructanswer_dataset)
        return instructanswer_dataset
