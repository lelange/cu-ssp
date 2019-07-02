import sys
import os
import argparse
import time
import numpy as np
import dill as pickle
import pandas as pd
import tensorflow as tf
sys.path.append('keras-tcn')
from tcn import tcn
import h5py

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from keras import backend as K
from keras import regularizers, constraints, initializers, activations

from keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, CuDNNGRU
from keras.layers import Dropout, Flatten, Activation, RepeatVector, Permute

from keras.layers import Dropout
from keras.layers import merge
from keras.layers.core import Reshape
from keras.layers.merge import concatenate
from keras.layers.recurrent import Recurrent
from keras.metrics import categorical_accuracy
from keras.models import Model, Input, Sequential
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import fbchat
from fbchat.models import *
import emoji
from utils import *


maxlen_seq = 700
minlen_seq= 100

#inputs: primary structure
train_input_seqs = np.load('../data/train_input.npy')
test_input_seqs = np.load('../data/test_input.npy')
#labels: secondary structure
train_target_seqs = np.load('../data/train_q8.npy')
test_target_seqs = np.load('../data/test_q8.npy')

#transform sequence to n-grams, default n=3
train_input_grams = seq2ngrams(train_input_seqs)
test_input_grams = seq2ngrams(test_input_seqs)

# Use tokenizer to encode and decode the sequences
tokenizer_encoder = Tokenizer()
tokenizer_encoder.fit_on_texts(train_input_grams)
tokenizer_decoder = Tokenizer(char_level = True) #char_level=True means that every character is treated as a token
tokenizer_decoder.fit_on_texts(train_target_seqs)

#train
train_input_data = tokenizer_encoder.texts_to_sequences(train_input_grams)
train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)

#test
test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
test_target_data = tokenizer_decoder.texts_to_sequences(test_target_seqs)

# pad sequences to maxlen_seq
X_train = sequence.pad_sequences(train_input_data, maxlen = maxlen_seq, padding = 'post')
X_test = sequence.pad_sequences(test_input_data, maxlen = maxlen_seq, padding = 'post')
train_target_data = sequence.pad_sequences(train_target_data, maxlen = maxlen_seq, padding = 'post')
test_target_data = sequence.pad_sequences(test_target_data, maxlen = maxlen_seq, padding = 'post')

# Computing the number of words and number of tags to be passed as parameters to the keras model
n_words = len(tokenizer_encoder.word_index) + 1
n_tags = len(tokenizer_decoder.word_index) + 1

print("number words or endoder word index: ", n_words)
print("number tags or decoder word index: ", n_tags)

# labels to one-hot
y_test = to_categorical(test_target_data)
y_train = to_categorical(train_target_data)

np.save('../data/X_train_6133.npy', X_train)
np.save('../data/y_train_6133.npy', y_train)
np.save('../data/X_test_513.npy', X_test)
np.save('../data/y_test_513.npy', y_test)