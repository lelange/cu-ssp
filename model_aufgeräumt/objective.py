import sys
import os
import time
import dill as pickle
import pprint

import numpy as np
import pandas as pd
import tensorflow as tf
sys.path.append('keras-tcn')
from tcn import tcn
import h5py

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from keras import backend as K
from keras import regularizers, constraints, initializers, activations

from keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
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
from utils import *

from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.mongoexp import MongoTrials

data_root = '../data/netsurfp/'

file_train = 'train'
file_test = ['cb513', 'ts115', 'casp12']

X_train_aug, y_train = get_data('train', True, False, True)

def build_model_ho_3(params, epochs = 20, verbose=2, hmm=True):
    model = None
    print('----------------------')
    print('----------------------')
    if verbose > 0:
        print('Params testing: ', params)
        print('\n ')

    if hmm:
        input = Input(shape=(600, 20,))
        profiles_input = Input(shape=(600, 30,))
        x1 = concatenate([input, profiles_input])
        x2 = concatenate([input, profiles_input])
        inp = [input, profiles_input]
    else:
        input = Input(shape=(600, 20,))
        x1 = input
        x2 = input
        inp = input

    x1 = Dense(params['dense1'], activation="relu")(x1)
    x1 = Dropout(params['dropout1'])(x1)
    # x1 = Bidirectional(CuDNNGRU(units=100, return_sequences=True))(x1)
    # Defining a bidirectional LSTM using the embedded representation of the inputs
    x2 = Bidirectional(CuDNNGRU(units=params['gru1'], return_sequences=True))(x2)
    # x2 = Dropout(0.5)(x2)
    if params['gru2']:
        x2 = Bidirectional(CuDNNGRU(units=params['gru2']['gru2_units'], return_sequences=True))(x2)
    if params['gru2'] and params['gru2']['gru3']:
        x2 = Bidirectional(CuDNNGRU(units=params['gru2']['gru3']['gru3_units'], return_sequences=True))(x2)
    # x2 = Dropout(0.5)(x2)
    COMBO_MOVE = concatenate([x1, x2])
    w = Dense(params['dense2'], activation="relu")(COMBO_MOVE)  # try 500
    w = Dropout(params['dropout2'])(w)
    w = tcn.TCN(return_sequences=True)(w)

    y = TimeDistributed(Dense(n_tags, activation="softmax"))(w)

    # Defining the model as a whole and printing the summary
    model = Model(inp, y)

    # Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
    adamOptimizer = Adam(lr=params['lr'], beta_1=0.8, beta_2=0.8, epsilon=None, decay=params['decay'], amsgrad=False)
    model.compile(optimizer=adamOptimizer, loss="categorical_crossentropy", metrics=["accuracy", accuracy])

    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=3, verbose=verbose, mode='max')
    checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose=1, save_best_only=True,
                                   mode='max')
    history = model.fit(X_train_aug, y_train, validation_data=(X_val_aug, y_val),
                        epochs=epochs, batch_size=params['batch_size'], callbacks=[checkpointer, earlyStopping],
                        verbose=verbose, shuffle=True)

    #test and evaluate performance
    X_test_aug, y_test = get_data(file_test[0], hmm, normalize, standardize)
    model.load_weights(load_file)
    print('\n----------------------')
    print('----------------------')
    print("evaluate " + file_test[0] + ":")
    score = model.evaluate(X_test_aug, y_test)
    print(file_test[0] + ' test accuracy:', score[2])

    result = {'loss': -score[2], 'status': STATUS_OK, 'space': params}

    return result