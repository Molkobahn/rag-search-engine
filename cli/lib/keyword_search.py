from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords
import string

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
    split_text = text.split()
    for item in split_text:
        if item in stopwords:
            split_text.remove(item)
        if item == "":
            split_text.remove(item)
    return split_text