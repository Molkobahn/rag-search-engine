from sentence_transformers import SentenceTransformer
import numpy as np
import os
import re
import json
from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    SCORE_PRECISION,
    load_movies,
    format_search_result,
)

EMBEDDING_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDDING_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")

class SemanticSearch():

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        all_chunks = []
        metadata = []
        for idx, doc in enumerate(documents):
            if doc['description'] == "":
                break
            chunks = semantic_chunk_command(doc['description'], max_chunk_size=4, overlap=1)
            for i, chunk in enumerate(chunks): 
                all_chunks.append(chunk)
                metadata.append({
                    'movie_idx': idx,
                    'chunk_idx': i,
                    'total_chunks': len(chunks)
                })
        self.chunk_embeddings = self.model.encode(all_chunks)
        self.chunk_metadata = metadata
        with open(CHUNK_EMBEDDING_PATH, "wb+") as f:
            np.save(f, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, 'w+') as f2:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)}, f2, indent=2)
        return self.chunk_embeddings
    
    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc
        if os.path.exists(CHUNK_EMBEDDING_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            with open(CHUNK_EMBEDDING_PATH, 'rb') as f:
                self.chunk_embeddings = np.load(f)
            with open(CHUNK_METADATA_PATH, 'r') as f2:
                metadata = json.load(f2)
                self.chunk_metadata = metadata['chunks']
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        embedded_query = self.generate_embedding(query)
        chunk_scores = []
        for i, c_e in enumerate(self.chunk_embeddings):
            cos_sim = cosine_similarity(embedded_query, c_e)
            chunk_scores.append({
                'chunk_idx': i,
                'movie_idx': self.chunk_metadata[i]['movie_idx'],
                'score': cos_sim,
            })
        movie_scores = {}
        for chunk_score in chunk_scores:
            idx = chunk_score['movie_idx']
            score = chunk_score['score'] 
            if idx not in movie_scores or score > movie_scores[idx]:
                movie_scores[idx] = score
        sorted_scores = dict(sorted(movie_scores.items(), key=lambda x: x[1], reverse=True))
        results = []
        filtered_scores = {k: sorted_scores[k] for k in list(sorted_scores)[:limit]}
        for movie_idx, score in filtered_scores.items():
            doc = self.documents[movie_idx]
            results.append(format_search_result(doc['id'], doc['title'], doc['description'][:100], round(score, SCORE_PRECISION)))
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
        

def chunk_command(text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
    split_text = text.split()
    chunks = []
    for i in range(0, len(split_text), chunk_size):
        if i >= chunk_size:
            chunks.append(" ".join(split_text[i-overlap:i+chunk_size]))
        else:
            chunks.append(" ".join(split_text[i:i+chunk_size]))
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunking {len(text)} characters")
        print(f"{i}. {chunk}")
        print()


def semantic_chunk_command(text, max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
    stripped_text = text.strip()
    if stripped_text == "":
        return []
    split_text = re.split(r"(?<=[.!?])\s+", stripped_text)
    if len(split_text) == 1:
        return split_text 
    chunks = []
    n = len(split_text)
    i = 0
    while i < n:
        chunk_words = split_text[i : i + max_chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break
        chunk = " ".join(chunk_words)
        stripped_chunk = chunk.strip()
        chunks.append(stripped_chunk)
        i += max_chunk_size - overlap
    return chunks


def embed_chunks_command():
    movies = load_movies()
    chunk_search_instance = ChunkedSemanticSearch()
    embeddings = chunk_search_instance.load_or_create_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked_command(query, limit=DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    search_instance = ChunkedSemanticSearch()
    search_instance.load_or_create_embeddings(movies)
    result = search_instance.search_chunks(query, limit)
    for i, res in enumerate(result, 1):
        print(f"\n{i}. {res['title']} (score:{res['score']:.4f})")
        print(f"    {res['document']}...")