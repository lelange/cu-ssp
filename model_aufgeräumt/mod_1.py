"""
Cascaded Convolution Model

- Pranav Shrestha (ps2958)
- Jeffrey Wan (jw3468)

"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import *
from keras.layers import *
from sklearn.model_selection import train_test_split, KFold
from keras.metrics import categorical_accuracy
from keras import backend as K
from keras.regularizers import l1, l2
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping ,ModelCheckpoint

from utils import *

### Data Retrieval
cb513filename = '../data/cb513.npy'
cb6133filteredfilename = '../data/cb6133filtered.npy'

maxlen_seq = r = 700  # protein residues padded to 700
f = 57  # number of features for each residue

#load train
train_df, X_aug_train = load_augmented_data(cb6133filteredfilename  ,maxlen_seq)
train_input_seqs, train_target_seqs = train_df[['input', 'expected']][(train_df.len <= maxlen_seq)].values.T
#load test
test_df, X_aug_test = load_augmented_data(cb513filename,maxlen_seq)
test_input_seqs, test_target_seqs = test_df[['input','expected']][(test_df.len <= maxlen_seq)].values.T

#tokenizer
# Converting the inputs to trigrams
train_input_grams = seq2ngrams(train_input_seqs)
# Initializing and defining the tokenizer encoders and decoders based on the train set
tokenizer_encoder = Tokenizer()
tokenizer_encoder.fit_on_texts(train_input_grams)
tokenizer_decoder = Tokenizer(char_level = True)
tokenizer_decoder.fit_on_texts(train_target_seqs)


# Using the tokenizer to encode and decode the sequences for use in training

#train inputs
train_input_data = tokenizer_encoder.texts_to_sequences(train_input_grams)
X_train = sequence.pad_sequences(train_input_data, maxlen = maxlen_seq, padding = 'post')
#train targets
train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)
train_target_data = sequence.pad_sequences(train_target_data, maxlen = maxlen_seq, padding = 'post')
y_train = to_categorical(train_target_data)
#test inputs
test_input_grams = seq2ngrams(test_input_seqs)
test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
X_test = sequence.pad_sequences(test_input_data, maxlen = maxlen_seq, padding = 'post')
#test targets
test_target_data = tokenizer_decoder.texts_to_sequences(test_target_seqs)
test_target_data = sequence.pad_sequences(test_target_data, maxlen = maxlen_seq, padding = 'post')
y_test = to_categorical(test_target_data)

# Computing the number of words and number of tags
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


p = {'activation1':[relu, softmax],
     'activation2':[relu, softmax],
     'optimizer': ['Nadam', "RMSprop"],
     'losses': ['categorical_crossentropy', keras.losses.binary_crossentropy],
     'first_hidden_layer': [10, 8, 6],
     'second_hidden_layer': [2, 4, 6],
     'batch_size': [64, 128, 10000],
     'epochs': [50, 75]}

def train(X_train, y_train, X_val=None, y_val=None):
    """
    Main Training function with the following properties:
        Optimizer - Nadam
        Loss function - Categorical Crossentropy
        Batch Size - 128 (any more will exceed Collab GPU RAM)
        Epochs - 50
    """
    model = CNN_BIGRU()
    model.compile(
        optimizer="Nadam",
        loss="categorical_crossentropy",
        metrics=["accuracy", accuracy])

    if X_val is not None and y_val is not None:


        earlyStopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max')
        checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose=1, save_best_only=True,
                                       mode='max')
        # Training the model on the training data and validating using the validation set
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=75, batch_size=128, callbacks=[checkpointer, earlyStopping], verbose=1, shuffle=True)

    else:
        history = model.fit(X_train, y_train,
                            batch_size=128, epochs=75)

    return history, model


""" Build model """


def conv_block(x, activation=True, batch_norm=True, drop_out=True, res=True):
    cnn = Conv1D(64, 11, padding="same")(x)
    if activation: cnn = TimeDistributed(Activation("relu"))(cnn)
    if batch_norm: cnn = TimeDistributed(BatchNormalization())(cnn)
    if drop_out:   cnn = TimeDistributed(Dropout(0.5))(cnn)
    if res:        cnn = Concatenate(axis=-1)([x, cnn])

    return cnn


def super_conv_block(x):
    c3 = Conv1D(32, 1, padding="same")(x)
    c3 = TimeDistributed(Activation("relu"))(c3)
    c3 = TimeDistributed(BatchNormalization())(c3)

    c7 = Conv1D(64, 3, padding="same")(x)
    c7 = TimeDistributed(Activation("relu"))(c7)
    c7 = TimeDistributed(BatchNormalization())(c7)

    c11 = Conv1D(128, 5, padding="same")(x)
    c11 = TimeDistributed(Activation("relu"))(c11)
    c11 = TimeDistributed(BatchNormalization())(c11)

    x = Concatenate(axis=-1)([x, c3, c7, c11])
    x = TimeDistributed(Dropout(0.5))(x)
    return x


def CNN_BIGRU():

    input = Input(shape=(maxlen_seq,))
    embed_out = Embedding(input_dim=n_words, output_dim=128, input_length=maxlen_seq)(input)
    profile_input = Input(shape=(maxlen_seq, 22))
    x = concatenate([embed_out, profile_input])  # 5600, 700, 150

    x = super_conv_block(x)
    x = conv_block(x)
    x = super_conv_block(x)
    x = conv_block(x)
    x = super_conv_block(x)
    x = conv_block(x)

    x = Bidirectional(CuDNNGRU(units=256, return_sequences=True, recurrent_regularizer=l2(0.2)))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(256, activation="relu"))(x)
    x = TimeDistributed(Dropout(0.5))(x)

    y = TimeDistributed(Dense(n_tags, activation="softmax"))(x)

    model = Model([input, profile_input], y)
    model.summary()

    return model

load_file = "./model/mod_1-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+".h5"
history, model = train([X_train, X_aug_train], y_train, X_val=[X_val, X_aug_val], y_val=y_val)

model.load_weights(load_file)

print("####evaluate:")
score = model.evaluate([X_test,X_aug_test], y_test, verbose=2, batch_size=1)
print(score)
print ('test loss:', score[0])
print ('test accuracy:', score[2])