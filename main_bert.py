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

from evaluation_metrics import corpus_bert_score

from evaluation_metrics import sentence_bert_score

#FILENAME = input("What is your filename? (in data folder): ")
FILENAME = 'baseline-CDandVinyl-model-1.5m.hdf5'
# baseline-ton-60-0.0098.hdf5
#Neem de parameters over zoals ze op colab stonden 
with open(f"pickles/cpt_2_data.p", "rb") as f2:
    data = pickle.load(f2)

DATASIZE = 3600 #param: 3600

label_data = data.loc[0:DATASIZE-1]['startLabels']
review_data = data.loc[0:DATASIZE-1]['cpt_input']
STEP_SIZE = 70
reference_prompts = []
reference_reviews = []
previous_prompt = []
for i in range(0, 3600):
    if(label_data[i] != previous_prompt):
        reference_prompts.append(label_data[i])
        reference_reviews.append(review_data[i])

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
# model = Sequential()
# model.add(GRU(600, input_shape=(seq_length, 1), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(GRU(600))
# model.add(Dropout(0.2))
# model.add(Dense(vocab_size, activation='softmax'))


#Laad hier de goeie weights zoals je ze lokaal hebt opgeslagen
#load_status = model.load_weights("model_weights/" + FILENAME)

#vul hier de 10 prompts in een lijst
#prompts = []
#prompts.append("<ReviewPrompt> Dawn FM <Review>")
#prompts.append("<ReviewPrompt> 30 <Review>")
# prompts.append(91*'`' + "Fragments")
# prompts.append(84*'`' + "The boy named If")
# prompts.append(93*'`' + "Ds4Ever")
# prompts.append(90*'`' + "Between us")
# prompts.append(96*'`' + "Sour")
# prompts.append(86*'`' + "The Highlights")
# prompts.append(74*'`' + "Fix yourself not the world")
# prompts.append(99*'`' + "=")


reviews = []
#REVIEW_FILE = 'reviews-baseline-15k.txt'
#REVIEW_FILE = 'reviews-baseline-75k.txt'
#REVIEW_FILE = 'reviews-baseline-150k.txt'
#REVIEW_FILE = 'reviews-baseline-1.5M.txt'
#REVIEW_FILE = 'reviews-gpt2-360.txt'
#REVIEW_FILE = 'reviews-gpt2-3600.txt'
#REVIEW_FILE = 'reviews-gpt2-7200.txt'
REVIEW_FILE = 'reviews-gpt2-36000.txt'
#REVIEW_FILE = 'reviews-retrained-15k.txt'
#REVIEW_FILE = 'reviews-retrained-75k.txt'
#REVIEW_FILE = 'reviews-retrained-150k.txt'
#REVIEW_FILE = 'reviews-retrained-1.5M.txt'
with open("reviews/" + REVIEW_FILE, "r") as file:
    for line in file:
        if line.rstrip() != '':
            reviews.append(line.rstrip())

to_review = []
for i in range(0, 1000,  50):
    #to_review.append(reference_prompts[i])
    to_review.append(reference_reviews[i])

real_reviews = []
for i in range(2000, 3000,  100):
    real_reviews.append(reference_reviews[i])

reviews = real_reviews
# for prompt in to_review:
#     while len(prompt) < seq_length:
#         prompt = '`' + prompt
        
#     while len(prompt) > seq_length:
#         prompt = prompt[:len(prompt)-10] + ' <review>'

#     string_mapped = [char_to_n[char] for char in prompt]

#     full_string = list(prompt)

#     print("Seed:")
#     print("\"", ''.join(full_string), "\"")

#     #Generating characters
#     for i in range(400):
#         x = np.reshape(string_mapped,(1,len(string_mapped), 1))
#         x = x / float(len(characters))

#         pred_index = np.argmax(model.predict(x, verbose=0))
#         seq = [n_to_char[value] for value in string_mapped]
#         full_string.append(n_to_char[pred_index])
    
#         string_mapped.append(pred_index)  # add the predicted character to the end
#         string_mapped = string_mapped[1:len(string_mapped)] # shift the string one character forward by removing pos. 0
    
#     # combining text
#     txt=""
#     for char in full_string:
#         if char != '`':
#             txt = txt+char
#     print(txt)    
#     reviews.append(txt + '\n' + '\n')

# with open("reviews150k.txt", "w") as file:
#     for review in reviews:
#         file.writelines(review)
#reviews[0] = ""
#reviews[0] = "<ReviewPrompt> Songs for the Shepherd 5.0 Five Stars <review>Love it!!  Great seller!"
reference_corpus = []
for i in range(0, len(reviews)):
    reference_corpus.append(to_review)

bertscore = corpus_bert_score(reference_corpus, reviews, "p")
print(bertscore)

# for review in reviews:
#     bert_sentence = sentence_bert_score(to_review, review, 'p')
#     print(bert_sentence)
    