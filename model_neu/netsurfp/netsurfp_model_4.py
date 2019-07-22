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
from collections import defaultdict
from datetime import datetime


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


args = parse_arguments(default_epochs=15)
batch_size = 64

start_time = time.time()
MODEL_NAME = 'mod_3'
save_pred_file = "_pred_3.npy"

N_FOLDS = 10 # for cross validation
MAXLEN_SEQ = 700 # only use sequences to this length and pad to this length, choose from 600, 608, 700
NB_CLASSES_Q8 = 9 # number Q8 classes, used in final layer for classification (one extra for empty slots)
NB_CLASSES_Q3 = 3 # number Q3 classes
NB_AS = 20 # number of amino acids, length of one-hot endoded amino acids vectors
NB_FEATURES = 30 # feature dimension

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
test_mode = args.test_mode
predict_only = args.predict

if test_mode:
    N_FOLDS = 2
    epochs = 2

data_root = '../data/netsurfp/'
weights_file = MODEL_NAME+"-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+".h5"
load_file = "./model/"+weights_file
file_scores = "logs/cv_results.txt"
file_scores_mean = "logs/cv_results_mean.txt"

file_train = 'train_' + str(MAXLEN_SEQ)
file_test = ['cb513_'+ str(MAXLEN_SEQ), 'ts115_'+ str(MAXLEN_SEQ), 'casp12_'+ str(MAXLEN_SEQ)]


'''

Model

'''
len_seq = 600

def build_model():
    model = None
    input = Input(shape=(MAXLEN_SEQ, NB_AS,))
    if hmm:
        profiles_input = Input(shape=(MAXLEN_SEQ, NB_FEATURES,))
        x = concatenate([input, profiles_input])
        inp = [input, profiles_input]
    else:
        x = input
        inp = input

    # one dense layer to remove sparsity
    x = Dense(128, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)

    x = Reshape([len_seq, 128, 1])(x)

    # Defining 3 convolutional layers with different kernel sizes
    # kernel size = 3
    conv1 = ZeroPadding2D((3 // 2, 0), data_format='channels_last')(x)
    conv1 = Conv2D(filters=64,
                   kernel_size=(3, 128),
                   input_shape=(1, len_seq, 128),
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
                   input_shape=(1, len_seq, 128),
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
                   input_shape=(1, len_seq, 128),
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
    conv = Reshape([len_seq, 3 * 64])(conv)

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
    y = TimeDistributed(Dense(NB_CLASSES_Q8,
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

def build_and_train(X_train_aug, y_train, X_val_aug, y_val, epochs = epochs):
    model = build_model()

    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='max')
    checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose = 1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=4, verbose=1, mode='max')

    history = model.fit(X_train_aug, y_train, validation_data=(X_val_aug, y_val),
            epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, earlyStopping, reduce_lr], verbose=1, shuffle=True)

    # plot accuracy during training
    return model, history

def evaluate_model(model, load_file, test_ind = None):
    if test_ind is None:
        test_ind = range(len(file_test))
    test_accs = []
    names = []
    for i in test_ind:
        X_test_aug, y_test = get_data(file_test[i], hmm, normalize, standardize)
        model.load_weights(load_file)
        print("####evaluate" + file_test[i] +":")
        score = model.evaluate(X_test_aug, y_test, verbose=2, batch_size=1)
        print(file_test[i] +' test loss:', score[0])
        print(file_test[i] +' test accuracy:', score[2])
        test_accs.append(score[2])
        names.append(file_test[i])
    return dict(zip(names, test_accs))


def crossValidation(load_file, X_train_aug, y_train, n_folds=N_FOLDS):
    X_train, X_aug_train = X_train_aug
    # Instantiate the cross validator
    kfold_splits = n_folds
    kf = KFold(n_splits=kfold_splits, shuffle=True)

    cv_scores = defaultdict(list)
    model_history = []

    # Loop through the indices the split() method returns
    for index, (train_indices, val_indices) in enumerate(kf.split(X_train, y_train)):
        print('\n\n-----------------------')
        print("Training on fold " + str(index + 1) + "/" + str(kfold_splits) +"...")
        print('-----------------------\n')

        # Generate batches from indices
        X_train_fold, X_val_fold = X_train[train_indices], X_train[val_indices]
        X_aug_train_fold, X_aug_val_fold = X_aug_train[train_indices], X_aug_train[val_indices]
        y_train_fold, y_val_fold = y_train[train_indices], y_train[val_indices]

        print("Training new iteration on " + str(X_train_fold.shape[0]) + " training samples, " + str(
            X_val_fold.shape[0]) + " validation samples...")

        model, history = build_and_train([X_train_fold, X_aug_train_fold], y_train_fold,
                                  [X_val_fold, X_aug_val_fold], y_val_fold)

        print(history.history)

        test_acc = evaluate_model(model, load_file, test_ind = [0, 1, 2])

        cv_scores['val_accuracy'].append(max(history.history['val_accuracy']))

        for k, v in test_acc.items():
            cv_scores[k].append(v)

        model_history.append(model)

    return cv_scores, model_history

# write best weight models in file and look for model (eg. mod_1) name in weight name
best_weights = "model/mod_4-CB513-2019_07_22-15_08.h5"



#--------------------------------- main ---------------------------------

if predict_only:
    build_and_predict(build_model(), best_weights, save_pred_file, MODEL_NAME, file_test)
    test_acc = None
    time_data = time.time() - start_time
    save_results = False
else:
    # load data
    X_train_aug, y_train = get_data(file_train, hmm, normalize, standardize)

    if hmm:
        print("X train shape: ", X_train_aug[0].shape)
        print("X aug train shape: ", X_train_aug[1].shape)
    else:
        print("X train shape: ", X_train_aug.shape)
    print("y train shape: ", y_train.shape)

    time_data = time.time() - start_time
    save_results = True

    if cross_validate:

        cv_scores, model_history = crossValidation(load_file, X_train_aug, y_train)
        test_accs = save_cv(weights_file, cv_scores, file_scores, file_scores_mean, N_FOLDS)
        test_acc = test_accs[file_test[0] + '_mean']

    else:
        X_train_aug, y_train, X_val_aug, y_val = train_val_split(hmm, X_train_aug, y_train, tv_perc)
        model, history = build_and_train(X_train_aug, y_train, X_val_aug, y_val, epochs=epochs)
        test_acc = evaluate_model(model, load_file)

time_end = time.time() - start_time
m, s = divmod(time_end, 60)
print("The program needed {:.0f}s to load the data and {:.0f}min {:.0f}s in total.".format(time_data, m, s))

telegram_me(m, s, sys.argv[0], test_acc, hmm, standardize)

if save_results:
    save_results_to_file(time_end, MODEL_NAME, weights_file, test_acc, hmm, standardize, normalize)
