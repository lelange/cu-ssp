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

normalize = False
standardize = True
hmm = True
embedding = False
epochs = 20
plot = False
no_input = False

batch_size = 16

n_tags = 8
n_words = 20
data_root = '../data/netsurfp/'

file_train = 'train'
file_test = ['cb513', 'ts115', 'casp12']

#load data
X_train_aug, y_train = get_data(file_train, hmm, normalize, standardize)

if hmm:
    print("X train shape: ", X_train_aug[0].shape)
    print("X aug train shape: ", X_train_aug[1].shape)
else:
    print("X train shape: ", X_train_aug.shape)
print("y train shape: ", y_train.shape)

DROPOUT_CHOICES = np.arange(0.0, 0.9, 0.1)
UNIT_CHOICES = [100, 200, 500, 800, 1000, 1200]
GRU_CHOICES = [100, 200, 300, 400, 500, 600]
BATCH_CHOICES = [16, 32]
LR_CHOICES = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01]
space = {
    'dense1': hp.choice('dense1', UNIT_CHOICES),
    'dropout1': hp.choice('dropout1', DROPOUT_CHOICES),
    'gru1': hp.choice('gru1', GRU_CHOICES),
    # nesting the layers ensures they're only un-rolled sequentially
    'gru2': hp.choice('gru2', [False, {
        'gru2_units': hp.choice('gru2_units', GRU_CHOICES),
        # only make the 3rd layer availabile if the 2nd one is
        'gru3': hp.choice('gru3', [False, {
            'gru3_units': hp.choice('gru3_units', GRU_CHOICES)
        }]),
    }]),
    'dense2': hp.choice('dense2', UNIT_CHOICES),
    'dropout2': hp.choice('dropout2', DROPOUT_CHOICES),
    'lr': hp.choice('lr', LR_CHOICES),
    'decay': hp.choice('decay', LR_CHOICES),
    'batch_size': hp.choice('batch_size', BATCH_CHOICES)
}
#load_file = "./model/mod_3-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+".h5"
load_file = "./model/mod_3-CB513-test.h5"
def build_model_ho_3(params, epochs = epochs, verbose=2, hmm=hmm):
    model = None
    print('----------------------')
    print('----------------------')
    if verbose > 0:
        print('Params testing: ', params)
        print('\n ')

    if hmm:
        input = Input(shape=(X_train_aug[0].shape[1], X_train_aug[0].shape[2],))
        profiles_input = Input(shape=(X_train_aug[1].shape[1], X_train_aug[1].shape[2],))
        x1 = concatenate([input, profiles_input])
        x2 = concatenate([input, profiles_input])
        inp = [input, profiles_input]
    else:
        input = Input(shape=(X_train_aug.shape[1], X_train_aug.shape[2],))
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

