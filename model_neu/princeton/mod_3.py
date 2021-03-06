import sys

sys.path.append('keras-tcn')
from tcn import tcn
import h5py
from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import dill as pickle
import pandas as pd
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, CuDNNGRU
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Activation, RepeatVector, Permute
import tensorflow as tf
from keras.layers.merge import concatenate
# from google.colab import files
from keras.layers import Dropout
from keras import regularizers
from keras.layers import merge
from keras.optimizers import Adam
from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.layers.recurrent import Recurrent
from keras.engine import InputSpec
from keras.engine.topology import Layer
import os
from keras.callbacks import EarlyStopping ,ModelCheckpoint

from utils import *

print("##device name:")
print(tf.test.gpu_device_name())
print("##gpu available:")
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

cb513filename = '../data/data_princeton/cb513.npy'
cb6133filteredfilename = '../data/data_princeton/cb6133filtered.npy'

maxlen_seq = 800

#load train and test and cut length to maxlen_seq
train_df, X_aug_train = load_augmented_data(cb6133filteredfilename  ,maxlen_seq)
train_input_seqs, train_target_seqs = train_df[['input', 'expected']][(train_df.len <= maxlen_seq)].values.T

test_df, X_aug_test = load_augmented_data(cb513filename,maxlen_seq)
test_input_seqs, test_target_seqs = test_df[['input','expected']][(test_df.len <= maxlen_seq)].values.T

# Using the tokenizer to encode and decode the sequences for use in training
# use preprocessing tools for text from keras to encode input sequence as word rank numbers and target sequence as one hot.
# To ensure easy to use training and testing, all sequences are padded with zeros to the maximum sequence length
# transform sequences to trigrams
train_input_grams = seq2ngrams(train_input_seqs)
#transform sequences
#fit alphabet on train basis
tokenizer_encoder = Tokenizer()
tokenizer_encoder.fit_on_texts(train_input_grams)

tokenizer_decoder = Tokenizer(char_level = True)
tokenizer_decoder.fit_on_texts(train_target_seqs)

#train
train_input_data = tokenizer_encoder.texts_to_sequences(train_input_grams)
X_train = sequence.pad_sequences(train_input_data, maxlen = maxlen_seq, padding = 'post')
#transform targets to one-hot
train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)
train_target_data = sequence.pad_sequences(train_target_data, maxlen = maxlen_seq, padding = 'post')
y_train = to_categorical(train_target_data)

#test
test_input_grams = seq2ngrams(test_input_seqs)
test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
X_test = sequence.pad_sequences(test_input_data, maxlen = maxlen_seq, padding = 'post')
test_target_data = tokenizer_decoder.texts_to_sequences(test_target_seqs)
test_target_data = sequence.pad_sequences(test_target_data, maxlen = maxlen_seq, padding = 'post')
y_test = to_categorical(test_target_data)

# Computing the number of words and number of tags to be passed as parameters to the keras model
n_words = len(tokenizer_encoder.word_index) + 1
n_tags = len(tokenizer_decoder.word_index) + 1

#### validation data

n_samples = len(train_df)
np.random.seed(0)
validation_idx = np.random.choice(np.arange(n_samples), size=300, replace=False)
training_idx = np.array(list(set(np.arange(n_samples))-set(validation_idx)))

X_val = X_train[validation_idx]
X_train = X_train[training_idx]
y_val = y_train[validation_idx]
y_train = y_train[training_idx]
X_aug_val = X_aug_train[validation_idx]
X_aug_train = X_aug_train[training_idx]
#### end validation


'''
Model

'''

def build_model():
    input = Input(shape=(None,))
    profiles_input = Input(shape=(None, 22))

    # Defining an embedding layer mapping from the words (n_words) to a vector of len 128
    x1 = Embedding(input_dim=n_words, output_dim=250, input_length=None)(input)
    x1 = concatenate([x1, profiles_input], axis=2)

    x2 = Embedding(input_dim=n_words, output_dim=125, input_length=None)(input)
    x2 = concatenate([x2, profiles_input], axis=2)

    x1 = Dense(1200, activation="relu")(x1)
    x1 = Dropout(0.5)(x1)

    # Defining a bidirectional LSTM using the embedded representation of the inputs
    x2 = Bidirectional(CuDNNGRU(units=500, return_sequences=True))(x2)
    x2 = Bidirectional(CuDNNGRU(units=100, return_sequences=True))(x2)
    COMBO_MOVE = concatenate([x1, x2])
    w = Dense(500, activation="relu")(COMBO_MOVE)  # try 500
    w = Dropout(0.4)(w)
    w = tcn.TCN(return_sequences=True)(w)

    y = TimeDistributed(Dense(n_tags, activation="softmax"))(w)

    # Defining the model as a whole and printing the summary
    model = Model([input, profiles_input], y)
    # model.summary()

    # Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
    adamOptimizer = Adam(lr=0.001, beta_1=0.8, beta_2=0.8, epsilon=None, decay=0.0001, amsgrad=False)
    model.compile(optimizer=adamOptimizer, loss="categorical_crossentropy", metrics=["accuracy", accuracy])
    return model

model = build_model()

load_file = "./model/mod_3-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+".h5"

#earlyStopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max')
checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose = 1, save_best_only=True, mode='max')
# Training the model on the training data and validating using the validation set
history=model.fit([X_train, X_aug_train], y_train, validation_data=([X_val, X_aug_val], y_val),
        epochs=5, batch_size=16, callbacks=[checkpointer], verbose=1, shuffle=True)

model.load_weights(load_file)
print("####evaluate:")
score = model.evaluate([X_test,X_aug_test], y_test, verbose=2, batch_size=1)
print(score)
print ('test loss:', score[0])
print ('test accuracy:', score[2])



