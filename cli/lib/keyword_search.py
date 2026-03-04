from .search_utils import load_stopwords, load_movies, PROJECT_ROOT, CACHE_INDEX_PATH, CACHE_DOCMAP_PATH, DEFAULT_SEARCH_LIMIT, CACHE_TERM_FREQUENCIES_PATH, BM25_K1
import string
from nltk.stem import PorterStemmer
import pickle
import os
from collections import defaultdict, Counter
import math


class InvertedIndex:
    
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int] = Counter()


    def __add_document(self, doc_id, text):
        tokens = tokenization(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set((doc_id,))
            else:
                self.index[token].add(doc_id)
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = Counter(token=1)
            else:
                self.term_frequencies[doc_id][token] += 1

    
    def get_documents(self, term):
        ids = list(self.index[term.lower()])
        ids.sort()
        return ids
    

    def get_tf(self, doc_id, term):
        token = tokenization(term)
        if len(token) > 1:
            raise Exception("More than one token")
        doc = self.term_frequencies[doc_id]
        return doc[token[0]]


    def get_bm25_idf(self, term: str):
        term = tokenization(term)
        if len(term) > 1:
            raise Exception ("More than one token")
        n = len(self.docmap)
        df = len(self.index[term[0]])
        bm25_idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
        return bm25_idf


    def get_bm25_tf(self, doc_id, term, k1=BM25_K1):
        tf = self.get_tf(doc_id, term)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1)
        return bm25_tf

    def build(self):
        movies = load_movies()
        for movie in movies:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie['id']] = f"{movie['title']} {movie['description']}"
    

    def save(self):
        if not os.path.exists(PROJECT_ROOT + '/cache'):
            os.mkdir(PROJECT_ROOT + '/cache')
        with open(CACHE_INDEX_PATH, 'wb+') as index_file:
            pickle.dump(self.index, index_file)
        with open(CACHE_DOCMAP_PATH, 'wb+')as docmap_file:
            pickle.dump(self.docmap, docmap_file)
        with open(CACHE_TERM_FREQUENCIES_PATH, 'wb+') as term_frequencies_file:
            pickle.dump(self.term_frequencies, term_frequencies_file)


    def load(self):
        try:
            with open(CACHE_INDEX_PATH, "rb") as index_file:
                self.index = pickle.load(index_file)
        except Exception as err:
            print(f"File not found: {err}")
        try:
            with open(CACHE_DOCMAP_PATH, "rb") as docmap_file:
                self.docmap = pickle.load(docmap_file)
        except Exception as err:
            print(f"File not found: {err}")
        try:
            with open(CACHE_TERM_FREQUENCIES_PATH, "rb") as term_frequencies_file:
                self.term_frequencies = pickle.load(term_frequencies_file)
        except Exception as err:
            print(f"File not found: {err}")


def bm25_tf_command(doc_id, term, k1=BM25_K1):
    idx = InvertedIndex()
    idx.load()
    bm25_tf = idx.get_bm25_tf(doc_id, term, k1)
    return bm25_tf


def bm25_idf_command(term):
    idx = InvertedIndex()
    idx.load()
    bm25_idf = idx.get_bm25_idf(term)
    return bm25_idf


def tfidf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    tf = idx.get_tf(doc_id, term)
    term = tokenization(term)
    idf = math.log((len(idx.docmap) + 1) / (len(idx.index[term[0]]) + 1))
    return tf * idf


def idf_command(term):
    idx = InvertedIndex()
    idx.load()
    total_doc_count = len(idx.docmap)
    term = tokenization(term)
    term_match_doc_count = len(idx.index[term[0]])
    idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    return idf


def tf_command(doc_id, term):
    idx = InvertedIndex()
    try:
        idx.load()
    except Exception as err:
        print(f"Couldn't load file: {err}")
    res = idx.get_tf(doc_id, term)
    if res == 0:
        print(0)
    else:
        print(res)


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query, limit = DEFAULT_SEARCH_LIMIT):
    idx = InvertedIndex()
    try:
        idx.load()
    except Exception as err:
        print(f"Couldn't load file: {err}")
    result = []
    query_tokens = tokenization(query)
    docs = [] 
    docs.append(has_matching_token(query_tokens, idx))
    for ids in docs:
        for id in ids:
            result.append(idx.docmap[id])
            if len(result) >= limit:
                break
    return result


def has_matching_token(query_tokens, idx):
        for query_token in query_tokens:
           if query_token in idx.index:     
                return idx.get_documents(query_token)


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def tokenization(text):
    text = preprocess_text(text)
    stopwords = load_stopwords()
    tokens = text.split()
    stemmer = PorterStemmer()
    stemmed_tokens = []
    for token in tokens:
        if token in stopwords:
            tokens.remove(token)
        if token == "":
            tokens.remove(token)
    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))
    return stemmed_tokens






