from sentence_transformers import SentenceTransformer
import numpy as np
import os
from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
)

EMBEDDING_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")

class SemanticSearch():

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if text == "" or text == " ":
            raise ValueError("Input string cannot be empty!")
        embedding = self.model.encode([text])
        return embedding[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        doc_str = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_str.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doc_str, show_progress_bar=True)
        with open(EMBEDDING_PATH, "wb+") as f:
            np.save(f, self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        if os.path.exists(EMBEDDING_PATH):
            with open(EMBEDDING_PATH, 'rb') as f:
                self.embeddings = np.load(f)
            if len(self.embeddings) == len(documents):
                return self.embeddings
            else:
                return self.build_embeddings(documents)
        else:
            return self.build_embeddings(documents)
        
    def search(self, query, limit):
        if type(self.embeddings) is None:
            raise ValueError("No embeddings loaded. Call 'load_or_create_embeddings' first.")
        embedding = self.generate_embedding(query)
        results = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(embedding, doc_embedding)
            results.append((similarity_score, self.documents[i]))
        sorted_results = sorted(results, key=lambda x:x[0], reverse=True)

        results = []
        for res in sorted_results[:limit]:
            results.append({
                "score": res[0],
                "title": res[1]['title'],
                "description": res[1]['description'],
            })
        return results


def verify_model():
    ss = SemanticSearch()
    print(f"Model leaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs: {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search_command(query, limit=DEFAULT_SEARCH_LIMIT):
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    results = ss.search(query, limit)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']} (score:{res['score']})")
        print(f"   {res['description'][:100]}...\n")
        
