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
with open(f"pickles/cpt_2_data_cd.p", "rb") as f2:
    data = pickle.load(f2)

DATASIZE = 3600 #param: 3600

data = data.loc[0:DATASIZE-1]['cpt_input']

text = " ".join(data.to_numpy())

#characters = sorted(list(set(" ".join(text)+'`')))
characters = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '~']
vocab_size = len(characters) #param

#parameters
seq_length = 100 #param: 100
#vocab_size = 95 #param

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
prompts.append(74*'`' + "Fix yourself not the world")
prompts.append(99*'`' + "=")
prompts.append(98*'`' + "30")
prompts.append(93*'`' + "Dawn FM")
prompts.append(91*'`' + "Fragments")
prompts.append(84*'`' + "The boy named If")
prompts.append(93*'`' + "Ds4Ever")
prompts.append(90*'`' + "Between us")
prompts.append(96*'`' + "Sour")
prompts.append(86*'`' + "The Highlights")

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

with open("reviews-15k.txt", "w") as file:
    for review in reviews:
        file.writelines(review)