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


#amount of data to load
INPUT_NUMBER = 1000

#load data from file
X = []
full_text = '\n'

fp = open("dataX.txt", "r")
for i, line in enumerate(fp):
    if i < INPUT_NUMBER:
        line = line.lower()
        #line = line.translate(str.maketrans("","", string.punctuation))
        line = line.rstrip()
        X.append(line)
        full_text = full_text + ' '.join(line.split())
    else:
        break

print(X[0])
i= 0
Y = []
fp = open("dataY.txt", "r")
for line in fp:
    if i < INPUT_NUMBER:
        if line.strip() != '':
            Y.append(line.rstrip())
        else:
            Y.append('\n')    
        i+=1
        full_text = full_text + ' '.join(line.split())
    else:
        break

words = X[0].split()
words.append('\n')
for i in range(1, len(X)):
    sequence = X[i].split()
    for j in range(0, len(sequence)):
        words.append(sequence[j])


for i in range(0, len(Y)):
    words.append(Y[i])

#unique words
words_set = sorted(list(set(words)))

#unique characters
chars = sorted(list(set(full_text)))
num_chars = len(chars)

#create word mappings
n_to_char = {n:char for n, char in enumerate(chars)}
char_to_n = {char:n for n, char in enumerate(chars)}

#create Vectors for training


# create vectors for the train text
     
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X)
Y_train_counts = count_vect.transform(Y)
# transform vectors using TF-IDF

print("transform vectors using TF-IDF")
tfidf_vectorizer = TfidfVectorizer().fit(X)

x_vec_train = tfidf_vectorizer.transform(X).toarray()
y_vec_train = tfidf_vectorizer.transform(Y).toarray()
x_vec_train = x_vec_train[:, :, None]
y_vec_train = y_vec_train[:, :, None]

X_train, X_test, y_train, y_test = train_test_split(x_vec_train, y_vec_train, test_size = 0.2, random_state = 1)

# #create word mappings
# n_to_word = {n:word for n, word in enumerate(words_set)}
# word_to_n = {word:n for n, word in enumerate(words_set)}

vocab_size = len(words_set)
# print('Number of unique words: ', vocab_size)

# X_train = []   # extracted sequences
# Y_train = []   # the target: follow up character for each sequence in X

# #Create X and Y vectors
# for i in range(0, len(X)):
#     X_train.append([word_to_n[word] for word in X[i].split()])
#     Y_train.append(word_to_n[Y[i]])
    

# X_modified = np.empty((len(X_train), 6))

# for i in range(0, len(X_train)):
#     while(len(X_train[i]) < 6):
#         X_train[i].insert(0,0)
#     while(len(X_train[i]) > 6):
#         X_train[i].pop()
#     X_modified[i] = np.array(X_train[i])

# X_modified = X_modified / float(vocab_size)
# #X_modified = np.reshape(X_modified, (len(X_modified), 6, 1))


# Y_modified = np_utils.to_categorical(Y_train, num_classes=vocab_size)
#print(y_train.shape)
y_train = tf.squeeze(y_train, axis=-1)
#Model using LSTM layers
model = Sequential()
# model.add(Embedding(vocab_size, X_modified.shape[1], input_length=X_modified.shape[1]))
model.add(GRU(32, input_shape = X_train.shape[1:], return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(32, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(32))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], activation='softmax'))

# model = Sequential()
# model.add(LSTM(units=6, input_shape = X_train.shape[1:], return_sequences = True))
# model.add(LSTM(units=6, return_sequences=True))
# model.add(LSTM(units=6, return_sequences=True))
# model.add(LSTM(units=1, return_sequences=True, name='output'))
#model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model.fit(X_train, y_train, epochs=25, batch_size=128)
model.save('/content/drive/My Drive/Colab Notebooks/LSTM_V4')


