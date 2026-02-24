from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords
import string
from nltk.stem import PorterStemmer

def search_command(query, limit = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    result = []
    for movie in movies:
        query_tokens = tokenization(query)
        title_tokens = tokenization(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            result.append(movie)
            if len(result) >= limit:
                break
    return result


def has_matching_token(query_tokens, title_tokens):
        for query_token in query_tokens:
            for title_token in title_tokens:
                if query_token in title_token:
                    return True
        return False


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