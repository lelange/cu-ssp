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
import fbchat
from fbchat.models import *
import telegram
from utils import *

from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval

start_time = time.time()

args = parse_arguments(default_epochs=10)

normalize = args.normalize
standardize = args.standardize
hmm = args.hmm
embedding = args.embedding
epochs = args.epochs
plot = args.plot
no_input = args.no_input
hyperopt = args.hyperopt

batch_size = 16

n_tags = 8
n_words = 20
data_root = '../data/netsurfp/'

file_train = 'train'
file_test = ['cb513', 'ts115', 'casp12']

def get_data(filename, hmm, normalize, standardize):

    print('Load ' + filename + ' data...')
    if embedding:
        input_seq = np.load(data_root + filename + '_netsurfp_input_embedding_residue.npy')
    else:
        if no_input:
            input_seq = np.load(data_root + filename + '_hmm.npy')
            if normalize:
                input_seq = normal(input_seq)
            if standardize:
                input_seq = standard(input_seq)

        else:
            input_seq =  np.load(data_root+filename+'_input.npy')
    q8 = np.load(data_root + filename + '_q8.npy')
    if hmm:
        profiles = np.load(data_root+filename+'_hmm.npy')
        if normalize:
            profiles = normal(profiles)
        if standardize:
            profiles = standard(profiles)
        input_aug = [input_seq, profiles]
    else:
        input_aug = input_seq
    return  input_aug, q8

#load data
X_train_aug, y_train = get_data(file_train, hmm, normalize, standardize)

if hmm:
    print("X train shape: ", X_train_aug[0].shape)
    print("X aug train shape: ", X_train_aug[1].shape)
else:
    print("X train shape: ", X_train_aug.shape)
print("y train shape: ", y_train.shape)

time_data = time.time() - start_time

def build_model():
    model = None

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

    x1 = Dense(1200, activation="relu")(x1)
    x1 = Dropout(0.5)(x1)

    #x1 = Bidirectional(CuDNNGRU(units=100, return_sequences=True))(x1)
    # Defining a bidirectional LSTM using the embedded representation of the inputs
    x2 = Bidirectional(CuDNNGRU(units=500, return_sequences=True))(x2)
    #x2 = Dropout(0.5)(x2)
    x2 = Bidirectional(CuDNNGRU(units=100, return_sequences=True))(x2)
    #x2 = Dropout(0.5)(x2)
    COMBO_MOVE = concatenate([x1, x2])
    w = Dense(500, activation="relu")(COMBO_MOVE)  # try 500
    w = Dropout(0.4)(w)
    w = tcn.TCN(return_sequences=True)(w)

    y = TimeDistributed(Dense(n_tags, activation="softmax"))(w)

    # Defining the model as a whole and printing the summary
    model = Model(inp, y)
    #model.summary()

    # Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
    adamOptimizer = Adam(lr=0.001, beta_1=0.8, beta_2=0.8, epsilon=None, decay=0.0001, amsgrad=False)
    model.compile(optimizer=adamOptimizer, loss="categorical_crossentropy", metrics=["accuracy", accuracy])
    return model


DROPOUT_CHOICES = np.arange(0.0, 0.9, 0.1)
UNIT_CHOICES = np.arange(100, 1201, 100, dtype=int)
BATCH_CHOICES = np.arange(16, 129, 16, dtype=int)
LR_CHOICES = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01]
space = {
    'dense1': hp.choice('dense1', UNIT_CHOICES),
    'dropout1': hp.choice('dropout1', DROPOUT_CHOICES),
    'gru1': hp.choice('gru1', UNIT_CHOICES),
    # nesting the layers ensures they're only un-rolled sequentially
    'gru2': hp.choice('gru2', [False, {
        'gru2_units': hp.choice('gru2_units', UNIT_CHOICES),
        # only make the 3rd layer availabile if the 2nd one is
        'gru3': hp.choice('gru3', [False, {
            'gru3_units': hp.choice('gru3_units', UNIT_CHOICES)
        }]),
    }]),
    'dense2': hp.choice('dense2', UNIT_CHOICES),
    'dropout2': hp.choice('dropout2', DROPOUT_CHOICES),
    'lr': hp.choice('lr', LR_CHOICES),
    'decay': hp.choice('decay', LR_CHOICES),
    'batch_size': hp.choice('batch_size', UNIT_CHOICES)
}

def build_model_ho(load_file, X_train_aug, y_train, X_val_aug, y_val,
                   epochs = epochs, params, verbose=0):
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

    X_test_aug, y_test = get_data(file_test[0], hmm, normalize, standardize)
    model.load_weights(load_file)
    print('\n----------------------')
    print('----------------------')
    print("evaluate" + file_test[i] + ":")
    score = model.evaluate(X_test_aug, y_test, verbose=verbose, batch_size=1)
    print(file_test[0] + ' test accuracy:', score[2])

    return {'accuracy': score[2], 'status': STATUS_OK}

load_file = "./model/mod_3-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+".h5"

def train_model(X_train_aug, y_train, X_val_aug, y_val, epochs = epochs):
    model = build_model()

    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, mode='max')
    checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose = 1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=1, verbose=1, mode='max', cooldown = 2)

    #tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False,
                             # write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                             # embeddings_metadata=None, embeddings_data=None, update_freq='batch')
    # Training the model on the training data and validating using the validation set
    history = model.fit(X_train_aug, y_train, validation_data=(X_val_aug, y_val),
            epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, earlyStopping, reduce_lr], verbose=1, shuffle=True)

    # plot accuracy during training
    if plot:
        plt.title('Accuracy')
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='val')
        plt.legend()
        plt.savefig('./plots/mod_3-CB513-' + datetime.now().strftime("%m_%d-%H_%M") + '_accuracy.png')

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

if args.cv :
    cv_scores, model_history = crossValidation(load_file, X_train_aug, y_train)
    test_acc = np.mean(cv_scores)
    print('Estimated accuracy %.3f (%.3f)' % (test_acc, np.std(cv_scores)))
else:
    if hmm:
        n_samples = len(X_train_aug[0])
    else:
        n_samples = len(X_train_aug)
    np.random.seed(0)
    validation_idx = np.random.choice(np.arange(n_samples), size=300, replace=False)
    training_idx = np.array(list(set(np.arange(n_samples)) - set(validation_idx)))

    y_val = y_train[validation_idx]
    y_train = y_train[training_idx]

    if hmm:
        X_val_aug = [X_train_aug[0][validation_idx], X_train_aug[1][validation_idx]]
        X_train_aug = [X_train_aug[0][training_idx], X_train_aug[1][training_idx]]
    else:
        X_val_aug = X_train_aug[validation_idx]
        X_train_aug = X_train_aug[training_idx]

    if hyperopt:
        test_params = {
            'dense1': 1200,
            'dropout1': 0.5,
            'gru1': 500,
            'gru2': {
                'gru2_units': 100,
                'gru3':  {
                    'gru3_units': 100
                },
            },
            'dense2': 500,
            'dropout2': 0.4,
            'lr': 0.001,
            'decay': 0.0001,
            'batch_size': 16,
        }
        build_model_ho(load_file, X_train_aug, y_train, X_val_aug, y_val,
                       epochs=epochs,params=test_params, verbose=1)
    else:
        model = train_model(X_train_aug, y_train, X_val_aug, y_val, epochs=epochs)
        test_acc = evaluate_model(model, load_file, [0])

time_end = time.time() - start_time
m, s = divmod(time_end, 60)
print("The program needed {:.0f}s to load the data and {:.0f}min {:.0f}s in total.".format(time_data, m, s))

telegram_me(m, s, sys.argv[0], test_acc, hmm, standardize, normalize, no_input)