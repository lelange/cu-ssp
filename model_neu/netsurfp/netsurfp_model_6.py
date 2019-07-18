import numpy as np
from numpy import array
import pandas as pd
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, GRU, Conv1D, CuDNNLSTM, concatenate
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
import tensorflow as tf
from keras import optimizers, initializers, constraints, regularizers
from keras.engine.topology import Layer
from tensorflow.keras.layers import Activation
from tensorflow.layers import Flatten
from keras.callbacks import EarlyStopping ,ModelCheckpoint
from keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau

import sys
import os
import time
import dill as pickle

from utils import *

start_time = time.time()

args = parse_arguments(default_epochs=80)

data_root = '/nosave/lange/cu-ssp/data/netsurfp/'
load_file = "./model/mod_6-CB513-" + datetime.now().strftime("%Y_%m_%d-%H_%M") + ".h5"

normalize = args.normalize
standardize = args.standardize
hmm = args.hmm
embedding = args.embedding
epochs = args.epochs
plot = args.plot
no_input = args.no_input
optimize = args.optimize
cross_validate = args.cv
tv_perc = args.tv_perc
batch_size = 128

n_tags = 3
n_words = 20

file_train = 'train_700'
file_test = ['cb513_700', 'ts115_700', 'casp12_700']

#load data
X_train_aug, y_q8, y_q3 = get_data(file_train, hmm, normalize, standardize)

time_data = time.time() - start_time

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

    z = Conv1D(64, 11, strides=1, padding='same')(x)
    w = Conv1D(64, 7, strides=1, padding='same')(x)
    x = concatenate([x, z], axis=2)
    x = concatenate([x, w], axis=2)
    z = Conv1D(64, 5, strides=1, padding='same')(x)
    w = Conv1D(64, 3, strides=1, padding='same')(x)
    x = concatenate([x, z], axis=2)
    x = concatenate([x, w], axis=2)
    x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(x)

    y_q8 = TimeDistributed(Dense(8, activation="softmax"), name="y_q8")(x)
    y_q3 = TimeDistributed(Dense(3, activation="softmax"), name="y_q3")(x)

    model = Model(inp, [y_q8, y_q3])
    model.compile(optimizer='RMSprop', loss="categorical_crossentropy", metrics=["accuracy", accuracy])
    model.summary()

    return model

def train_model(X_train_aug, y_train, X_val_aug, y_val, epochs = epochs):
    model = build_model()

    earlyStopping = EarlyStopping(monitor='val_y_q8_accuracy', patience=10, verbose=1, mode='max')
    checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_y_q8_accuracy', verbose = 1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_y_q8_accuracy', factor=0.2, patience=6, verbose=1, mode='max')

    history = model.fit(X_train_aug, y_train, validation_data=(X_val_aug, y_val),
            epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, earlyStopping], verbose=1, shuffle=True)

    # plot accuracy during training
    return model

def evaluate_model(model, load_file, test_ind = None):
    if test_ind is None:
        test_ind = range(len(file_test))
    test_accs = []
    names = []
    for i in test_ind:
        X_test_aug, y_test_q8, y_test_q3 = get_data(file_test[i], hmm, normalize, standardize)
        y_test = [y_test_q8, y_test_q3]
        model.load_weights(load_file)
        print("####evaluate " + file_test[i] +":")
        score = model.evaluate(X_test_aug, y_test, verbose=2, batch_size=1)
        print(score)
        print(file_test[i] +' test loss:', score[0])
        print(file_test[i] +' test accuracy:', score[2])
        test_accs.append(score[2])
        names.append(file_test[i])
    return dict(zip(names, test_accs))

if cross_validate :
    cv_scores, model_history = crossValidation(load_file, X_train_aug, y_train)
    test_acc = np.mean(cv_scores)
    print('Estimated accuracy %.3f (%.3f)' % (test_acc, np.std(cv_scores)))
else:
    y_train = [y_q8, y_q3]
    X_train_aug, y_train, X_val_aug, y_val = train_val_split(hmm, X_train_aug, y_train, tv_perc)
    model = train_model(X_train_aug, y_train, X_val_aug, y_val, epochs=epochs)
    test_acc = evaluate_model(model, load_file)

time_end = time.time() - start_time
m, s = divmod(time_end, 60)
print("The program needed {:.0f}s to load the data and {:.0f}min {:.0f}s in total.".format(time_data, m, s))

telegram_me(m, s, sys.argv[0], test_acc, hmm, standardize, normalize, no_input)
