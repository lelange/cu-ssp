import numpy as np
from numpy import array
import pandas as pd
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import *
from keras.layers import *
from keras.regularizers import l2
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.metrics import categorical_accuracy
from keras import backend as K
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import tensorflow as tf
from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from datetime import datetime

import sys
import os
import time
import dill as pickle

from utils import *

start_time = time.time()

args = parse_arguments(default_epochs=15)

data_root = '../data/netsurfp/'
load_file = "./model/mod_4-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+".h5"

normalize = args.normalize
standardize = args.standardize
hmm = args.hmm
embedding = args.embedding
epochs = args.epochs
plot = args.plot
no_input = args.no_input
optimize = args.optimize
cross_validate = args.cv
batch_size = 64

n_tags = 8
n_words = 20

file_train = 'train'
file_test = ['cb513', 'ts115', 'casp12']

#load data
X_train_aug, y_train = get_data(file_train, hmm, normalize, standardize)

time_data = time.time() - start_time

'''

Model

'''
def build_model():
    model = None
    if hmm:
        input = Input(shape=(X_train_aug[0].shape[1], X_train_aug[0].shape[2],))
        profiles_input = Input(shape=(X_train_aug[1].shape[1], X_train_aug[1].shape[2],))
        x = concatenate([input, profiles_input])
        inp = [input, profiles_input]
    else:
        input = Input(shape=(X_train_aug.shape[1], X_train_aug.shape[2],))
        x = input
        inp = input

    # one dense layer to remove sparsity
    x = Dense(128, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = Reshape([maxlen_seq, 128, 1])(x)

    # Defining 3 convolutional layers with different kernel sizes
    # kernel size = 3
    conv1 = ZeroPadding2D((3 // 2, 0), data_format='channels_last')(x)
    conv1 = Conv2D(filters=64,
                   kernel_size=(3, 128),
                   input_shape=(1, maxlen_seq, 128),
                   data_format='channels_last',
                   strides=(1, 1),
                   dilation_rate=(1, 1),
                   activation='relu',
                   use_bias=True,
                   kernel_initializer='glorot_uniform',
                   bias_initializer='zeros')(conv1)
    conv1 = BatchNormalization(axis=-1)(conv1)

    # kernel size = 7
    conv2 = ZeroPadding2D((7 // 2, 0), data_format='channels_last')(x)
    conv2 = Conv2D(filters=64,
                   kernel_size=(7, 128),
                   input_shape=(1, maxlen_seq, 128),
                   data_format='channels_last',
                   strides=(1, 1),
                   padding='valid',
                   dilation_rate=(1, 1),
                   activation='relu',
                   use_bias=True,
                   kernel_initializer='glorot_uniform',
                   bias_initializer='zeros')(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)

    # kernel size = 11
    conv3 = ZeroPadding2D((11 // 2, 0), data_format='channels_last')(x)
    conv3 = Conv2D(filters=64,
                   kernel_size=(11, 128),
                   input_shape=(1, maxlen_seq, 128),
                   data_format='channels_last',
                   strides=(1, 1),
                   padding='valid',
                   dilation_rate=(1, 1),
                   activation='relu',
                   use_bias=True,
                   kernel_initializer='glorot_uniform',
                   bias_initializer='zeros')(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv = concatenate([conv1, conv2, conv3])
    conv = Reshape([maxlen_seq, 3 * 64])(conv)

    # Defining 3 bidirectional GRU layers; taking the concatenation of outputs
    gru1 = Bidirectional(GRU(32,
                             return_sequences='True',
                             activation='tanh',
                             recurrent_activation='hard_sigmoid',
                             use_bias=True,
                             kernel_initializer='glorot_uniform',
                             recurrent_initializer='orthogonal',
                             bias_initializer='zeros',
                             dropout=0.0,
                             recurrent_dropout=0.1,
                             implementation=1))(conv)

    gru2 = Bidirectional(GRU(32,
                             return_sequences='True',
                             activation='tanh',
                             recurrent_activation='hard_sigmoid',
                             use_bias=True,
                             kernel_initializer='glorot_uniform',
                             recurrent_initializer='orthogonal',
                             bias_initializer='zeros',
                             dropout=0.0,
                             recurrent_dropout=0.1,
                             implementation=1))(gru1)

    gru3 = Bidirectional(GRU(32,
                             return_sequences='True',
                             activation='tanh',
                             recurrent_activation='hard_sigmoid',
                             use_bias=True,
                             kernel_initializer='glorot_uniform',
                             recurrent_initializer='orthogonal',
                             bias_initializer='zeros',
                             dropout=0.0,
                             recurrent_dropout=0.1,
                             implementation=1))(gru2)

    comb = concatenate([gru1, gru2, gru3, conv])

    # Defining two fully-connected layers with dropout
    x = TimeDistributed(Dense(256,
                              activation='relu',
                              use_bias=True,
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros'))(comb)
    x = Dropout(0.1)(x)

    x = TimeDistributed(Dense(128,
                              activation='relu',
                              use_bias=True,
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros'))(x)
    x = Dropout(0.1)(x)

    # Defining the output layer
    y = TimeDistributed(Dense(n_tags,
                              activation='softmax',
                              use_bias=False,
                              kernel_initializer='glorot_uniform'))(x)

    # Defining the model as a whole and printing the summary
    model = Model(inp, y)

    model.compile(optimizer="nadam", loss="categorical_crossentropy", metrics=["accuracy", accuracy])
    model.summary()

    return model


'''
Fitting and Predicting

'''

def train_model(X_train_aug, y_train, X_val_aug, y_val, epochs = epochs):
    model = build_model()

    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='max')
    checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose = 1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=4, verbose=1, mode='max')

    history = model.fit(X_train_aug, y_train, validation_data=(X_val_aug, y_val),
            epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, earlyStopping, reduce_lr], verbose=1, shuffle=True)

    # plot accuracy during training
    return model

def evaluate_model(model, load_file, test_ind = None):
    if test_ind is None:
        test_ind = range(len(file_test))
    for i in test_ind:
        X_test_aug, y_test = get_data(file_test[i], hmm, normalize, standardize)
        model.load_weights(load_file)
        print("####evaluate" + file_test[i] +":")
        score = model.evaluate(X_test_aug, y_test, verbose=2, batch_size=1)
        print(file_test[i] +' test loss:', score[0])
        print(file_test[i] +' test accuracy:', score[2])
    return score[2]


if cross_validate :
    cv_scores, model_history = crossValidation(load_file, X_train_aug, y_train)
    test_acc = np.mean(cv_scores)
    print('Estimated accuracy %.3f (%.3f)' % (test_acc, np.std(cv_scores)))
else:
    X_train_aug, y_train, X_val_aug, y_val = train_val_split(hmm, X_train_aug, y_train)
    model = train_model(X_train_aug, y_train, X_val_aug, y_val, epochs=epochs)
    test_acc = evaluate_model(model, load_file)

time_end = time.time() - start_time
m, s = divmod(time_end, 60)
print("The program needed {:.0f}s to load the data and {:.0f}min {:.0f}s in total.".format(time_data, m, s))

telegram_me(m, s, sys.argv[0], test_acc, hmm, standardize, normalize, no_input)
