from .keyword_search import tokenization
from .search_utils import load_movies, PROJECT_ROOT, CACHE_INDEX_PATH, CACHE_DOCMAP_PATH
import pickle
import os



class InvertedIndex:
    
    def __init__(self):
        self.index = {}
        self.docmap = {}


    def __add_document(self, doc_id, text):
        tokens = tokenization(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set((doc_id,))
            else:
                self.index[token].add(doc_id)

    
    def get_documents(self, term):
        ids = list(self.index[term.lower()])
        ids.sort()
        return ids
    

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
