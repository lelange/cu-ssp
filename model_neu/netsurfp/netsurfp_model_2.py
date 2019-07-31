import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
import tensorflow as tf
from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import sys

import time
import dill as pickle
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval

from utils import parse_arguments, get_data, train_val_split, \
    save_cv, telegram_me, accuracy, get_acc, build_and_predict, \
    save_results_to_file

from collections import defaultdict
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_NAME = 'mod_2'
save_pred_file = "_pred_2.npy"

N_FOLDS = 10 # for cross validation
MAXLEN_SEQ = 768 # only use sequences to this length and pad to this length, choose from 600, 608, 700
NB_CLASSES_Q8 = 9 # number Q8 classes, used in final layer for classification (one extra for empty slots)
NB_CLASSES_Q3 = 3 # number Q3 classes
NB_AS = 20 # number of amino acids, length of one-hot endoded amino acids vectors
NB_FEATURES = 30 # feature dimension


start_time = time.time()

args = parse_arguments(default_epochs=20)

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

batch_size = 128

data_root = '../data/netsurfp/'
weights_file = MODEL_NAME+"-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+".h5"
load_file = "./model/"+weights_file
file_scores = "logs/cv_results.txt"
file_scores_mean = "logs/cv_results_mean.txt"

file_train = 'train_' + str(MAXLEN_SEQ)
file_test = ['cb513_'+ str(MAXLEN_SEQ), 'ts115_'+ str(MAXLEN_SEQ), 'casp12_'+ str(MAXLEN_SEQ)]

# Dropout to prevent overfitting.
droprate = 0.3

#### model

def conv_block(x, n_channels, droprate):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(n_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Dropout(droprate)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(n_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    return x

def up_block(x, n_channels):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling1D(size = 2)(x)
    x = Conv1D(n_channels, 2, padding = 'same', kernel_initializer = 'he_normal')(x)
    return x

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


def build_model():
    model = None
    input = Input(shape=(MAXLEN_SEQ, NB_AS,))

    if hmm:
        profiles_input = Input(shape=(MAXLEN_SEQ, NB_FEATURES,))
        merged_input = concatenate([input, profiles_input])
        inp = [input, profiles_input]
    else:
        merged_input = input
        inp = input

    merged_input = Conv1D(128, 3, padding='same', kernel_initializer='he_normal')(merged_input)

    conv1 = conv_block(merged_input, 128, droprate)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = conv_block(pool1, 192, droprate)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = conv_block(pool2, 384, droprate)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = conv_block(pool3, 768, droprate)
    pool4 = MaxPooling1D(pool_size=2)(conv4)

    conv5 = conv_block(pool4, 1536, droprate)

    up4 = up_block(conv5, 768)
    up4 = concatenate([conv4, up4], axis=2)
    up4 = conv_block(up4, 768, droprate)

    up3 = up_block(up4, 384)
    up3 = concatenate([conv3, up3], axis=2)
    up3 = conv_block(up3, 384, droprate)

    up2 = up_block(up3, 192)
    up2 = concatenate([conv2, up2], axis=2)
    up2 = conv_block(up2, 192, droprate)

    up1 = up_block(up2, 128)
    up1 = concatenate([conv1, up1], axis=2)
    up1 = conv_block(up1, 128, droprate)

    up1 = BatchNormalization()(up1)
    up1 = ReLU()(up1)

    # the following it equivalent to Conv1D with kernel size 1
    # A dense layer to output from the LSTM's64 units to the appropriate number of tags to be fed into the decoder
    y = TimeDistributed(Dense(NB_CLASSES_Q8, activation="softmax"))(up1)

    # Defining the model as a whole and printing the summary
    model = Model(inp, y)
    # Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy

    optim = RMSprop(lr=0.002)

    model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=["accuracy", accuracy])
    model.summary()

    return model


def build_and_train(X_train_aug, y_train, X_val_aug, y_val, epochs = epochs):
    model = build_model()

    ####callbacks for fitting
    def scheduler(i, lr):
        if i in [60]:
            return lr * 0.5
        return lr
    reduce_lr = LearningRateScheduler(schedule=scheduler, verbose=1)
    # reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
    #                             patience=8, min_lr=0.0005, verbose=1)

    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=15, verbose=1, mode='max')
    checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose=1, save_best_only=True,
                                   mode='max')
    # Training the model on the training data and validating using the validation set
    history = model.fit(X_train_aug, y_train, validation_data=(X_val_aug, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, reduce_lr, earlyStopping],
                        verbose=1, shuffle=True)

    # plot accuracy during training

    return model, history

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


best_weights = "model/mod_2-CB513-2019_07_23-10_35.h5"
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

time_end = time.time() - start_time
m, s = divmod(time_end, 60)
print("The program needed {:.0f}s to load the data and {:.0f}min {:.0f}s in total.".format(time_data, m, s))

telegram_me(m, s, sys.argv[0], test_acc, hmm, standardize)

if save_results:
    save_results_to_file(time_end, MODEL_NAME, weights_file, test_acc, hmm, standardize, normalize)
