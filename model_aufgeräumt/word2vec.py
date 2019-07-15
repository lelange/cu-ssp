import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import dill as pickle

from utils import *

import gensim
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer

start_time = time.time()

n_tags = 8
n_words = 20
save_root = '../data/word2vec/'

file_train = 'train_700'
file_test = ['cb513_700', 'ts115_700', 'casp12_700']

#load data
X_train, y_train = get_data(file_train, False, False, False)

def seq2ngrams(seqs, n = 3):
    return np.array( ((seq[i : i + n] for i in range(len(seq))) for seq in seqs) )

n_grams = seq2ngrams(X_train)
print(n_grams.shape)
tokenizer_encoder = Tokenizer()
tokenizer_encoder.fit_on_texts(n_grams)
data = tokenizer_encoder.texts_to_sequences(n_grams)

n_words = len(tokenizer_encoder.word_index)
print('Number of words: ', n_words)

'''
model1 = gensim.models.Word2Vec(data, min_count = 1,
                              size = 100, window = 5)

# Create CBOW model
np.save(save_root+file_train+'_CBOW.npy', model1)

# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count=1, size=100,
                                window=5, sg=1)

np.save(save_root+file_train+'_skip_gram.npy', model2)

'''



'''
print("Cosine similarity between 'alice' " +
      "and 'wonderland' - CBOW : ",
      model1.similarity('alice', 'wonderland'))

print("Cosine similarity between 'alice' " +
      "and 'machines' - CBOW : ",
      model1.similarity('alice', 'machines'))

# Print results 
print("Cosine similarity between 'alice' " +
      "and 'wonderland' - Skip Gram : ",
      model2.similarity('alice', 'wonderland'))

print("Cosine similarity between 'alice' " +
      "and 'machines' - Skip Gram : ",
      model2.similarity('alice', 'machines')) 

'''

