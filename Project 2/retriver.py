from __future__ import annotations

import heapq
import logging

import torch

from utils import cos_sim, dot_score

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseSearch(ABC):
    @abstractmethod
    def search(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        top_k: int,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        pass


# DenseRetrievalExactSearch is parent class for any dense model that can be used for retrieval
# Abstract class is BaseSearch
import os
import torch

class DenseRetrievalExactSearch(BaseSearch):
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, cache_dir="corpus_cache", **kwargs):
        self.model = model
        self.batch_size = batch_size
        self.score_functions = {"cos_sim": cos_sim, "dot": dot_score}
        self.score_function_desc = {
            "cos_sim": "Cosine Similarity",
            "dot": "Dot Product",
        }
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.results = {}
        self.cache_dir = cache_dir

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)  # 创建缓存文件夹

    def encode_corpus(self, corpus: list[dict], batch_num: int) -> torch.Tensor:
        """检查是否已有缓存的 corpus embedding，如果有就加载，否则计算并存储"""
        cache_path = os.path.join(self.cache_dir, f"corpus_batch_{batch_num}.pt")
        
        if os.path.exists(cache_path):
            logger.info(f"Loading cached corpus embeddings from {cache_path} ...")
            return torch.load(cache_path)

        corpus_with_instruction = [
            f"Represent the document for retrieval: {doc['title']} {doc['text']}"
            for doc in corpus
        ]
        sub_corpus_embeddings = self.model.encode(
            corpus_with_instruction,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=self.convert_to_tensor,
        )

        torch.save(sub_corpus_embeddings, cache_path)
        logger.info(f"Saved corpus embeddings to {cache_path}")
        return sub_corpus_embeddings

    def search(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        top_k: int,
        score_function: str,
        return_sorted: bool = True,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        # Create embeddings for all queries using model.encode_queries()
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError(
                f"score function: {score_function} must be either (cos_sim) for cosine similarity or (dot) for dot product"
            )

        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]
        queries_with_instruction = [f"Instruct: {q}" for q in queries] 
        query_embeddings = self.model.encode(
            queries_with_instruction,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=self.convert_to_tensor,
        )

        logger.info("Sorting Corpus by document length (Longest first)...")

        corpus_ids = sorted(
            corpus,
            key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
            reverse=True,
        )
        corpus = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches...")
        itr = range(0, len(corpus), self.corpus_chunk_size)

        result_heaps = {qid: [] for qid in query_ids}  # Keep only the top-k docs for each query
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info(f"Encoding Batch {batch_num + 1}/{len(itr)}...")
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))
            corpus_list = corpus[corpus_start_idx:corpus_end_idx]

            # 加载或计算 Corpus Embeddings
            sub_corpus_embeddings = self.encode_corpus(corpus_list, batch_num)

            # Compute similarites using either cosine-similarity or dot product
            cos_scores = self.score_functions[score_function](query_embeddings, sub_corpus_embeddings)
            cos_scores[torch.isnan(cos_scores)] = -1
            # Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores,
                min(top_k + 1, cos_scores.shape[1]),
                dim=1,
                largest=True,
                sorted=return_sorted,
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_ids[corpus_start_idx + sub_corpus_id]
                    if corpus_id != query_id:
                        if len(result_heaps[query_id]) < top_k:
                            # Push item on the heap
                            heapq.heappush(result_heaps[query_id], (score, corpus_id))
                        else:
                            # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                            heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score

        return self.results
