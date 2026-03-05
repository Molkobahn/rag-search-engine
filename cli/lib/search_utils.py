import json
import os


DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
DATA_PATH = os.path.join(DATA_DIR, "movies.json")
STOPWORDS_PATH = os.path.join(DATA_DIR, "stopwords.txt")
CACHE_INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
CACHE_DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
CACHE_TERM_FREQUENCIES_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")
CACHE_DOC_LENGTHS_PATH = os.path.join(CACHE_DIR, "doc_lengths.pkl")

BM25_K1 = 1.5
BM25_B = 0.75

def load_movies():
    with open(DATA_PATH) as json_file:
        movies_dict = json.load(json_file)
    return movies_dict["movies"]


def load_stopwords():
    with open(STOPWORDS_PATH) as txt_file:
        stopwords = txt_file.read()
        stopwords = stopwords.splitlines()
    return stopwords