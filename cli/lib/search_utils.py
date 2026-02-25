import json
import os


DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_INDEX_PATH = os.path.join(PROJECT_ROOT, "cache", "index.pkl")
CACHE_DOCMAP_PATH = os.path.join(PROJECT_ROOT, "cache", "docmap.pkl")

def load_movies():
    with open(DATA_PATH) as json_file:
        movies_dict = json.load(json_file)
    return movies_dict["movies"]


def load_stopwords():
    with open(STOPWORDS_PATH) as txt_file:
        stopwords = txt_file.read()
        stopwords = stopwords.splitlines()
    return stopwords