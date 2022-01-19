import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RNN
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

text = (open("data/wonderland.txt").read())
text = text.lower()

print("Downloaded Alice in Wonderland data with {} characters.".format(len(text)))
print("FIRST 1000 CHARACTERS: ")
print(text[:1000])

characters = sorted(list(set(text)))
words = sorted(list(set(text.split())))

text_as_words = text.split()

#create word mappings
n_to_word = {n:word for n, word in enumerate(words)}
word_to_n = {word:n for n, word in enumerate(words)}

vocab_size = len(words)
print('Number of unique words: ', vocab_size)

X = []   # extracted sequences
Y = []   # the target: follow up character for each sequence in X

length = len(text_as_words)
seq_length = 25

#Create X and Y vectors

for i in range(0, length - seq_length, 1):
    sequence = text_as_words[i:i + seq_length]
    label = text_as_words[i + seq_length]
    X.append([word_to_n[word] for word in sequence])
    Y.append(word_to_n[label])
    
print('Number of extracted sequences:', len(X))

X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(characters))
Y_modified = np_utils.to_categorical(Y)

X_modified.shape, Y_modified.shape

print("X[0].shape = {}, Y[0].shape = {}".format(X_modified[0].shape, Y_modified[0].shape))
print("X[0]: ", X_modified[0])
print("Y[0]: ", Y_modified[0])

#Model using LSTM layers
model = Sequential()
model.add(LSTM(400, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(400, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(400))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X_modified, Y_modified, epochs=10, batch_size=128)



start = np.random.randint(0, len(X)-1) # or generate random   #random row from the X array
string_mapped = list(X[start])
full_string = [n_to_word[value] for value in string_mapped]

# generating characters
for i in range(400):
    x = np.reshape(string_mapped,(1,len(string_mapped), 1))
    x = x / float(len(characters))

    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [n_to_word[value] for value in string_mapped]
    full_string.append(n_to_word[pred_index])

    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]

# combining text
txt=""
for char in full_string:
    txt = txt+char

print(start)
print(txt)