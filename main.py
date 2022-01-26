import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Embedding
from keras.layers import RNN
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import pickle
import os
from numpy.random import choice
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from data.load_data import Data

FILENAME = input("What is your filename? (in data folder)")

#Neem de parameters over zoals ze op colab stonden 
with open(f"pickles/cpt_2_data.p", "rb") as f2:
    data = pickle.load(f2)

DATASIZE = 3600 #param: 3600

data = data.loc[0:DATASIZE-1]['cpt_input']

text = " ".join(data.to_numpy())

characters = sorted(list(set(" ".join(text)+'`')))
vocab_size = len(characters) #param

#parameters
seq_length = 100 #param: 100
vocab_size = 95 #param

n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}

#Copy of Colab model
model = Sequential()
model.add(GRU(600, input_shape=(seq_length, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(600))
model.add(Dropout(0.2))
model.add(Dense(vocab_size, activation='softmax'))


#Laad hier de goeie weights zoals je ze lokaal hebt opgeslagen
load_status = model.load_weights("data/" + FILENAME)

#vul hier de 10 prompts in een lijst
prompts = []
prompts.append(173*'`' + "Dawn FM")
prompts.append(173*'`' + "Test FM")

reviews = []
for prompt in prompts:
    string_mapped = [char_to_n[char] for char in prompt]

    full_string = list(prompt)

    print("Seed:")
    print("\"", ''.join(full_string), "\"")

    #Generating characters
    for i in range(400):
        x = np.reshape(string_mapped,(1,len(string_mapped), 1))
        x = x / float(len(characters))

        pred_index = np.argmax(model.predict(x, verbose=0))
        seq = [n_to_char[value] for value in string_mapped]
        full_string.append(n_to_char[pred_index])
    
        string_mapped.append(pred_index)  # add the predicted character to the end
        string_mapped = string_mapped[1:len(string_mapped)] # shift the string one character forward by removing pos. 0
    
    # combining text
    txt=""
    for char in full_string:
        txt = txt+char
    print(txt)    
    reviews.append(txt + '\n' + '\n')

with open("reviews.txt", "w") as file:
    for review in reviews:
        file.writelines(review)