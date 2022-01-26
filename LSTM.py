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
# import os
# from numpy.random import choice
# import string

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from data.load_data import Data

with open(f"pickles/cpt_2_data.p", "rb") as f2:
    data = pickle.load(f2)

DATASIZE = 36000

data = data.loc[0:DATASIZE-1]['cpt_input']

text = " ".join(data.to_numpy())

characters = sorted(list(set(" ".join(text)+'`')))
vocab_size = len(characters)

X = []   # extracted sequences
Y = []   # the target - the follow up character

seq_length = 100   #number of characters to consider before predicting the following character

n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}

length = len(text)

for r in data.to_numpy():

    for i in range(0, len(r) - seq_length, 5):
        sequence = text[i:i + seq_length]
        label = text[i + seq_length]
        X.append([char_to_n[char] for char in sequence])
        Y.append(char_to_n[label])
        
        if i == 100:
            break
    
print('Number of extracted sequences:', len(X))
    
X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(characters))
Y_modified = np_utils.to_categorical(Y)

model = Sequential()
model.add(GRU(600, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(600))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

filepath="model_weights/retrained-offfice-model-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X_modified, Y_modified, epochs=6, batch_size=256, callbacks = callbacks_list)

prompt = 93*'`' + "Dawn FM"
string_mapped = [char_to_n[char] for char in prompt]

#string_mapped = prompt

full_string = list(prompt)

print("Seed:")
print("\"", ''.join(full_string), "\"")

# generating characters
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
    
print(prompt)
print(txt)
