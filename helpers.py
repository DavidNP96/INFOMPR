import pandas as pd
import json
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
#from data import load_data
import itertools

def find_product_name():
        
# def load_data(self):
#     start = time.time()


    dir_path = os.path.dirname(os.path.realpath(__file__))

    data = getDF(dir_path + r"\meta_CDs_and_Vinyl.json")
    
    with open("id_to_title.json", "w") as file:
        data_map = {}
        for asin, title in zip(data["asin"], data["title"]):
            data_map[asin] = title
        json.dump(data_map, file)

def parse(path):
    g = open(path, 'rb')
    for l in g:
        yield json.loads(l)

def getDF( path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def create_vectorizer(data):
    # create corpus
    corpus = []
    for text in data["reviewText"]:
        if type(text) == str:  
            corpus.append(text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    with open("./pickles/tfidf_vectorizer.p", "wb") as outfile:
        pickle.dump(X, outfile)
        
def split_into_words(sentences):
	"""Splits multiple sentences into words and flattens the result"""
	return list(itertools.chain(*[_.split(" ") for _ in sentences]))

def get_word_ngrams(n, sentences):
	"""Calculates word n-grams for multiple sentences."""
	assert len(sentences) > 0
	assert n > 0

	words = split_into_words(sentences)
	return get_ngrams(n, words)

def get_ngrams(n, text):
	"""Calcualtes n-grams.
	Args:
        which n-grams to calculate
    	text: An array of tokens
	Returns:
        set of n-grams
	"""
	ngram_set = set()
	text_length = len(text)
	max_index_ngram_start = text_length - n
	for i in range(max_index_ngram_start + 1):
		ngram_set.add(tuple(text[i:i + n]))
	return ngram_set

if __name__ == "__main__":
    data = load_data.Data()
    create_vectorizer(data.data)
