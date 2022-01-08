import pandas as pd
import json
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from data import load_data

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

if __name__ == "__main__":
    data = load_data.Data()
    create_vectorizer(data.data)
