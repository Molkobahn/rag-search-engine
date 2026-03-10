import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    
def normalize(scores: list):
    if len(scores) == 0:
        print()
    max = float("-inf")
    min = float("inf")
    for score in scores:
        if score > max:
            max = score
        if score < min:
            min = score
    if max == min:
        result = [1.0] * len(scores)
        return result
    else:
        result = []
        for score in scores:
            result.append((score - min) / (max - min))
        return result