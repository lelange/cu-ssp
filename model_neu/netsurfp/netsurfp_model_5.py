############################################
#
# LSTMs with Luang attention
#
############################################

##### Load .npy data file and generate sequence csv and profile csv files  #####
import numpy as np
import pandas as pd
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, CuDNNLSTM
from keras.layers import Activation, BatchNormalization, dot, concatenate
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

import sys
import os
import time
import dill as pickle

from utils import *
from collections import defaultdict
from datetime import datetime


start_time = time.time()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_NAME = 'mod_5'
save_pred_file = "_pred_5.npy"

N_FOLDS = 10 # for cross validation
MAXLEN_SEQ = 700 # only use sequences to this length and pad to this length, choose from 600, 608, 700
NB_CLASSES_Q8 = 9 # number Q8 classes, used in final layer for classification (one extra for empty slots)
NB_CLASSES_Q3 = 3 # number Q3 classes
NB_AS = 20 # number of amino acids, length of one-hot endoded amino acids vectors
NB_FEATURES = 30 # feature dimension

args = parse_arguments(default_epochs=25)

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

batch_size = 48

data_root = '../data/netsurfp/'
weights_file = MODEL_NAME+"-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+".h5"
load_file = "./model/"+weights_file
file_scores = "logs/cv_results.txt"
file_scores_mean = "logs/cv_results_mean.txt"

file_train = 'train_' + str(MAXLEN_SEQ)
file_test = ['cb513_'+ str(MAXLEN_SEQ), 'ts115_'+ str(MAXLEN_SEQ), 'casp12_'+ str(MAXLEN_SEQ)]

############################### Model starts here ##############################

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

    x1_out = Bidirectional(CuDNNLSTM(units=75, return_sequences=True), merge_mode='concat')(x)
    x1_out_last = x1_out[:, -1, :]

    x2_out = CuDNNLSTM(units=150, return_sequences=True)(x1_out,
                                                                           initial_state=[x1_out_last, x1_out_last])
    x2_out_last = x2_out[:, -1, :]

    attention = dot([x2_out, x1_out], axes=[2, 2])
    attention = Activation('softmax')(attention)
    context = dot([attention, x1_out], axes=[2, 1])
    x2_out_combined_context = concatenate([context, x2_out])

    x3_out = CuDNNLSTM(units=150, return_sequences=True)(x2_out,
                                                                           initial_state=[x2_out_last, x2_out_last])
    x3_out_last = x3_out[:, -1, :]

    attention_2 = dot([x3_out, x2_out], axes=[2, 2])
    attention_2 = Activation('softmax')(attention_2)
    context_2 = dot([attention_2, x2_out], axes=[2, 1])
    x3_out_combined_context = concatenate([context_2, x3_out])

    attention_2_1 = dot([x3_out, x1_out], axes=[2, 2])
    attention_2_1 = Activation('softmax')(attention_2_1)
    context_2_1 = dot([attention_2_1, x1_out], axes=[2, 1])
    x3_1_out_combined_context = concatenate([context_2_1, x3_out])

    x4_out = CuDNNLSTM(units=150, return_sequences=True)(x3_out,
                                                                           initial_state=[x3_out_last, x3_out_last])
    x4_out_last = x4_out[:, -1, :]

    attention_3 = dot([x4_out, x3_out], axes=[2, 2])
    attention_3 = Activation('softmax')(attention_3)
    context_3 = dot([attention_3, x3_out], axes=[2, 1])
    x4_out_combined_context = concatenate([context_3, x4_out])

    attention_3_1 = dot([x4_out, x2_out], axes=[2, 2])
    attention_3_1 = Activation('softmax')(attention_3_1)
    context_3_1 = dot([attention_3_1, x2_out], axes=[2, 1])
    x4_1_out_combined_context = concatenate([context_3_1, x4_out])

    attention_3_2 = dot([x4_out, x1_out], axes=[2, 2])
    attention_3_2 = Activation('softmax')(attention_3_2)
    context_3_2 = dot([attention_3_2, x1_out], axes=[2, 1])
    x4_2_out_combined_context = concatenate([context_3_2, x4_out])

    x5_out = CuDNNLSTM(units=150, return_sequences=True)(x4_out,
                                                                           initial_state=[x4_out_last, x4_out_last])
    x5_out_last = x5_out[:, -1, :]

    attention_4 = dot([x5_out, x4_out], axes=[2, 2])
    attention_4 = Activation('softmax')(attention_4)
    context_4 = dot([attention_4, x4_out], axes=[2, 1])
    x5_out_combined_context = concatenate([context_4, x5_out])

    attention_4_1 = dot([x5_out, x3_out], axes=[2, 2])
    attention_4_1 = Activation('softmax')(attention_4_1)
    context_4_1 = dot([attention_4_1, x3_out], axes=[2, 1])
    x5_1_out_combined_context = concatenate([context_4_1, x5_out])

    attention_4_2 = dot([x5_out, x2_out], axes=[2, 2])
    attention_4_2 = Activation('softmax')(attention_4_2)
    context_4_2 = dot([attention_4_2, x2_out], axes=[2, 1])
    x5_2_out_combined_context = concatenate([context_4_2, x5_out])

    attention_4_3 = dot([x5_out, x1_out], axes=[2, 2])
    attention_4_3 = Activation('softmax')(attention_4_3)
    context_4_3 = dot([attention_4_3, x1_out], axes=[2, 1])
    x5_3_out_combined_context = concatenate([context_4_3, x5_out])

    out = keras.layers.Add()([x2_out_combined_context, \
                              x3_out_combined_context, x3_1_out_combined_context, \
                              x4_out_combined_context, x4_1_out_combined_context, x4_2_out_combined_context, \
                              x5_out_combined_context, x5_1_out_combined_context, x5_2_out_combined_context, \
                              x5_3_out_combined_context])

    fc1_out = TimeDistributed(Dense(150, activation="relu"))(out)  # equation (5) of the paper
    output = TimeDistributed(Dense(NB_CLASSES_Q8, activation="softmax"))(fc1_out)  # equation (6) of the paper

    model = Model(inp, output)
    #model.summary()

    # Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
    rmsprop = keras.optimizers.RMSprop(lr=0.003, rho=0.9, epsilon=None, decay=0.0)  # add decay=0.5 after 15 epochs
    model.compile(optimizer=rmsprop, loss="categorical_crossentropy", metrics=["accuracy", accuracy])

    return model


################################################################################

def build_and_train(X_train_aug, y_train, X_val_aug, y_val, epochs = epochs):
    model = build_model()

    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='max')
    checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose = 1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=4, verbose=1, mode='max', cooldown = 2)

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
    K.clear_session()
    del model
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
best_weights = "model/mod_5-CB513-2019_07_23-23_14.h5"



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
        #korrigiere name und return
        model, history = build_and_train(X_train_aug, y_train, X_val_aug, y_val, epochs=epochs)
        test_acc = evaluate_model(model, load_file)

K.clear_session()
del model

time_end = time.time() - start_time
m, s = divmod(time_end, 60)
print("The program needed {:.0f}s to load the data and {:.0f}min {:.0f}s in total.".format(time_data, m, s))

telegram_me(m, s, sys.argv[0], test_acc, hmm, standardize)

if save_results:
    save_results_to_file(time_end, MODEL_NAME, weights_file, test_acc, hmm, standardize, normalize)
