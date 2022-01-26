# returns a train_x train_y test_x test_y 
from numpy.lib.function_base import digitize
import pandas as pd
import os
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from tokenize import tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import json
from collections import defaultdict
import gzip
import time
import pickle
import random
from multiprocessing import Process, Queue
import re
from tqdm import tqdm


CPT_DATA_PATH = "cpt_2_data.p"
class Data:
    def __init__(self, cpt_2=False):
        self.cpt_2 = cpt_2
        # self.id_to_title = self.load_id_to_title()
        
        self.load_data_conc()

    def load_data_conc(self):

        start = time.time()
        if os.path.exists(f"../pickles/{CPT_DATA_PATH}"):
            with open(f"../pickles/{CPT_DATA_PATH}", "rb") as file:
                self.data = pickle.load(file)
                end = time.time()
                print("time spend loading the data = ",end - start)
                if self.cpt_2==True:
                    return
                
        self.prepare_data()
        end = time.time()
        print("time spend = ",end - start)
    def prepare_data(self):
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        paths = [dir_path + r"\CDs_and_Vinyl_5.json", dir_path + r"\meta_CDs_and_Vinyl.json" ]
        # for path in paths:
        #     threads.append(asyncio.create_task(self.load_data(path)))
        # group = await asyncio.gather(*threads)
        threads = list()
        out_queue = Queue()
        res_lst = []
        # process = Process(target=self.load_data, args=(path,out_queue))
        workers = [ Process(target=self.load_data, args=(path,out_queue)) for path in paths ]
        for work in workers:  
            work.start()
            
        for j in range(len(workers)):
            obj = out_queue.get()
            if obj[0] == "meta":
                self.meta_data = obj[1]
            else:
                self.data = obj[1]
        
        for work in workers: 
            work.join()

        print("merging data")
        begin = time.time()
        self.data = self.data.merge(self.meta_data, on="asin", how = 'inner')
        self.data = self.data
        eind = time.time()
        print("merging data took", eind-begin)
        
        # for line in self.data:
        #     # line = json.loads(line)
        #     id = line["asin"]
            
        #     try:
        #         # line["title"] = self.id_to_title[id]
        #         meta_data = self.meta_data.loc[self.meta_data['asin'] == id]
        #         print("meta_data", meta_data)

        #         # review_records.append(line)
        #     except Exception as e:
        #         # print("couldn't find id", e)
        #         i+=1
        #         pass
                
        # data = pd.DataFrame.from_records(review_records)
        # print("missed ids", i)
        # return data
    
    def load_data(self, file, out_queue):
        if "meta" in file:
            obj = self.getDF(file)
            out_queue.put(["meta",obj])
            print("loaded metadata")
        else:
            obj =  self.getDF(file)
            out_queue.put(["data",obj])
            print("loaded data")
        
    def parse(self, path):
        with open(path, 'rb') as file:
            for l in file:
                yield json.loads(l)

    def getDF(self, path):
        with open(path, 'rb') as file:
            df = []
            for line in file:
                df.append(json.loads(line))
            
            return pd.DataFrame.from_records(df)

    def split_data(self, test_size=1/9, cpt_2=False):

        """split the data into train and test.
        cpt_2 flag will make the funciton return text as input and output. While with the cpt_flag of it will return tf-idf vectors
        """
        print("in split data")
        if os.path.exists(f"../pickles/{CPT_DATA_PATH}"):
            return
        self.instances = self.data[["title", "reviewText"]]
        self.instances["startToken"] = " <review>"
        self.instances.insert(0, "ReviewPrompt", "<ReviewPrompt> ")
                       
        self.instances["cpt_input"] = self.instances["ReviewPrompt"] + self.instances["title"] + self.instances["startToken"] + self.instances["reviewText"]
        self.instances["startLabels"] = self.instances['ReviewPrompt']+ self.instances['title'] + self.instances['startToken']
        self.data = self.instances[["cpt_input", "reviewText", "startLabels"]]
        self.data = self.data[self.data['reviewText'].notna()]
        print("pickling data")
        if self.cpt_2:
            with open(f"../pickles/{CPT_DATA_PATH}", "wb") as file:
                pickle.dump( self.data, file)
            return self.data
        else:
            return self.vectorize()
    
    def vectorize(self):
        if os.path.exists(f"../pickles/dataX.p"):
            with open(f"../pickles/dataX.p", "rb") as f1:
                self.dataX = pickle.load(f1)
            with open(f"../pickles/dataY.p", "rb") as f2:
                self.dataY = pickle.load(f2)
        else:

            start = time.time()
            print("vectorizing documents...")
            self.dataX = []
            self.dataY = []
            self.data = self.data[self.data['cpt_input'].notna()]
            for instance in tqdm(self.data.itertuples()):
                seq_length = len(instance.startLabels)
                
                reviewText = instance.reviewText
                # self.review_text.append(reviewText)

                n_chars = len(reviewText)

                
                for i in range(0, n_chars - seq_length, 1):
            
                    seq_in = reviewText[i:i + seq_length]
                    seq_out = reviewText[i + seq_length]
                    self.dataX.append(seq_in)
                    self.dataY.append(seq_out)
            
            
            with open("../pickles/dataX.p", "wb") as file:
                pickle.dump(self.dataX, file)
            with open("../pickles/dataY.p", "wb") as file2:
                pickle.dump(self.dataY, file2)

            end = time.time()
            print("instances processed it took", end-start)
            return self.create_bow()

    
    # create Bag-of-Words vectors for the train and test sentences
    def create_bow(self):
        print("creating bow")
        # X_index = np.arange(0, len(self.dataX), dtype=np.int8)
        # Y_index = np.arange(0, len(self.dataY), dtype=np.int8)
        # print("xindex", X_index[:10])
        # train_x, test_x, self.train_y, self.test_y = train_test_split(X_index, self.dataY,test_size=1/10, random_state=3, shuffle=True)
        # print("train indexes", train_x[:10])
        # self.dataX = np.array(self.dataX)
        # self.train_x = self.dataX[train_x]
        # self.test_x = self.dataX[test_x]
        # self.train_y = self.dataY[train_y]
        # self.test_y = self.dataY[test_y]
        if os.path.exists(f"../pickles/countVectorizer.p"):
            with open(f"../pickles/countVectorizer.p", "rb") as f:
                self.count_vect = pickle.load(f)
            with open(f"../pickles/tfidfVectorizer.p", "rb") as f2:
                self.tfidf_transformer = pickle.load(f2)
        
        self.count_vect = CountVectorizer()
        # create vectors for the train text
        # print("create vectors for the train text", self.train_x[:10], self.train_y[:10])
        
        X_train_counts = self.count_vect.fit_transform(self.data["cpt_input"])
        
        # transform vectors using TF-IDF
        print("transform vectors using TF-IDF")
        self.tfidf_transformer = TfidfTransformer()
        x_train = self.tfidf_transformer.fit_transform(X_train_counts)

        with open("countVectorizer.p", "wb") as file:
            pickle.dump(self.count_vect, file)

        with open("tfidfVectorizer.p", "wb") as file2:
            pickle.dump(self.count_vect, file2)


        # # fit test sentences to the bag of words vectors
        # print("fit xtest to the bag of words vectors")
        # x_test = self.tfidf_transformer.transform(
        #     self.count_vect.transform(self.test_x))

        # print("fit ytrain to the bag of words vectors")
        # # fit test sentences to the bag of words vectors
        # y_train = self.tfidf_transformer.transform(
        #     self.count_vect.transform(self.train_y))

        # print("fit ytest to the bag of words vectors")
        # # fit test sentences to the bag of words vectors
        # y_test = self.tfidf_transformer.transform(
        #     self.count_vect.transform(self.test_y))

        return self.count_vect, self.tfidf_transformer

    def split(a_list, train_test_ratio=9/10):
        """ ratio is trainsize/complete dataset"""
        split = int(len(a_list) * train_test_ratio)
        return a_list[:split], a_list[split:]

    def process_data(self):
        self.data
if __name__ == "__main__":
    data = Data(cpt_2=True)
    
    data.split_data(cpt_2=True)
    print(data.data)