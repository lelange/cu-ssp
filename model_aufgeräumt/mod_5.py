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
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional
from keras.layers import Activation, BatchNormalization, dot, concatenate
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau

import sys
import os
import time
import dill as pickle

from utils import *

start_time = time.time()

args = parse_arguments(default_epochs=25)

data_root = '../data/netsurfp/'
load_file = "./model/mod_5-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+".h5"

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
############################### Model starts here ##############################

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

    x1_out = Bidirectional(LSTM(units=75, return_sequences=True, recurrent_dropout=0.2), merge_mode='concat')(x)
    x1_out_last = x1_out[:, -1, :]

    x2_out = LSTM(units=150, return_sequences=True, recurrent_dropout=0.2)(x1_out,
                                                                           initial_state=[x1_out_last, x1_out_last])
    x2_out_last = x2_out[:, -1, :]

    attention = dot([x2_out, x1_out], axes=[2, 2])
    attention = Activation('softmax')(attention)
    context = dot([attention, x1_out], axes=[2, 1])
    x2_out_combined_context = concatenate([context, x2_out])

    x3_out = LSTM(units=150, return_sequences=True, recurrent_dropout=0.2)(x2_out,
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

    x4_out = LSTM(units=150, return_sequences=True, recurrent_dropout=0.2)(x3_out,
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

    x5_out = LSTM(units=150, return_sequences=True, recurrent_dropout=0.2)(x4_out,
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
    output = TimeDistributed(Dense(n_tags, activation="softmax"))(fc1_out)  # equation (6) of the paper

    model = Model(inp, output)
    #model.summary()

    # Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
    rmsprop = keras.optimizers.RMSprop(lr=0.003, rho=0.9, epsilon=None, decay=0.0)  # add decay=0.5 after 15 epochs
    model.compile(optimizer=rmsprop, loss="categorical_crossentropy", metrics=["accuracy", accuracy])

    return model


################################################################################

def train_model(X_train_aug, y_train, X_val_aug, y_val, epochs = epochs):
    model = build_model()

    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='max')
    checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose = 1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=1, verbose=1, mode='max', cooldown = 2)

    history = model.fit(X_train_aug, y_train, validation_data=(X_val_aug, y_val),
            epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, earlyStopping, reduce_lr], verbose=1, shuffle=True)

    # plot accuracy during training
    return model

def evaluate_model(model, load_file, test_ind = None):
    if test_ind is None:
        test_ind = range(len(file_test))
    for i in test_ind:
        X_test_aug, y_test = get_data(file_test[i], hmm, normalize, standardize)
        #model.load_weights(load_file)
        print("####evaluate" + file_test[i] +":")
        score = model.evaluate(X_test_aug, y_test, verbose=2, batch_size=1)
        print(file_test[i] +' test loss:', score[0])
        print(file_test[i] +' test accuracy:', score[2])
    return score[2]

def train_val_split(hmm, X_train_aug, y_train):
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

    return X_train_aug, y_train, X_val_aug, y_val

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
