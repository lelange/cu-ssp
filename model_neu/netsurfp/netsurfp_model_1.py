"""
Cascaded Convolution Model

- Pranav Shrestha (ps2958)
- Jeffrey Wan (jw3468)

"""
import sys
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
from keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
import os
import time
import dill as pickle
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval
import json
from bson import json_util

from utils import *
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_NAME = 'mod_1'
start_time = time.time()

args = parse_arguments(default_epochs=75)

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

n_tags = 8
n_words = 20
data_root = '../data/netsurfp/'
weights_file = "mod_1-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+".h5"
load_file = "./model/"+weights_file

file_train = 'train_700'
file_test = ['cb513_700', 'ts115_700', 'casp12_700']

#load data
X_train_aug, y_train = get_data(file_train, hmm, normalize, standardize)

if hmm:
    print("X train shape: ", X_train_aug[0].shape)
    print("X aug train shape: ", X_train_aug[1].shape)
else:
    print("X train shape: ", X_train_aug.shape)
print("y train shape: ", y_train.shape)

time_data = time.time() - start_time

'''
p = {'activation1':[relu, softmax],
     'activation2':[relu, softmax],
     'optimizer': ['Nadam', "RMSprop"],
     'losses': ['categorical_crossentropy', keras.losses.binary_crossentropy],
     'first_hidden_layer': [10, 8, 6],
     'second_hidden_layer': [2, 4, 6],
     'batch_size': [64, 128, 10000],
     'epochs': [50, 75]}
'''


def train_model (X_train_aug, y_train, X_val_aug, y_val, epochs = epochs):
    """
    Main Training function with the following properties:
        Optimizer - Nadam
        Loss function - Categorical Crossentropy
        Batch Size - 128 (any more will exceed Collab GPU RAM)
        Epochs - 50
    """
    model = CNN_BIGRU()

    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max')
    checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose=1, save_best_only=True,
                                   mode='max')
    # Training the model on the training data and validating using the validation set
    history = model.fit(X_train_aug, y_train, validation_data=(X_val_aug, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, earlyStopping],
                        verbose=1, shuffle=True)

    return model, history


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
    if hmm:
        input = Input(shape=(X_train_aug[0].shape[1], X_train_aug[0].shape[2],))
        profiles_input = Input(shape=(X_train_aug[1].shape[1], X_train_aug[1].shape[2],))
        x = concatenate([input, profiles_input])
        inp = [input, profiles_input]
    else:
        input = Input(shape=(X_train_aug.shape[1], X_train_aug.shape[2],))
        x = input
        inp = input

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

    model = Model(inp, y)

    model.compile(
        optimizer="Nadam",
        loss="categorical_crossentropy",
        metrics=["accuracy", accuracy])

    return model

def evaluate_model(model, load_file, test_ind = None):
    if test_ind is None:
        test_ind = range(len(file_test))
    test_accs = []
    names = []
    for i in test_ind:
        X_test_aug, y_test = get_data(file_test[i], hmm, normalize, standardize)
        model.load_weights(load_file)
        print("\nevaluate " + file_test[i] +":")
        score = model.evaluate(X_test_aug, y_test, verbose=2, batch_size=1)
        print(file_test[i] +' test loss:', score[0])
        print(file_test[i] +' test accuracy:', score[2])
        test_accs.append(score[2])
        names.append(file_test[i])
    return dict(zip(names, test_accs))

def crossValidation(load_file, X_train_aug, y_train, n_folds=2):
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

        model, history = train_model([X_train_fold, X_aug_train_fold], y_train_fold,
                                  [X_val_fold, X_aug_val_fold], y_val_fold)


        test_acc = evaluate_model(model, load_file, test_ind = [0, 1, 2])
        print(test_acc.keys())
        print(test_acc.values())
        for k, v in test_acc.items():
            print(k+ ' >%.3f' % v)

        try:
            cv_scores['val_accuracy'].append(history.history['val_accuracy'][0])
        except:
            cv_scores['val_accuracy'].append(history.history['val_accuracy'])

        for k, v in test_acc.items():
            cv_scores[k].append(v)

        print(cv_scores)
        print('history:')
        print(history.history)
        print(history.history['val_accuracy'])

        model_history.append(model)

    return cv_scores, model_history


if cross_validate :
    cv_scores, model_history = crossValidation(load_file, X_train_aug, y_train)
    test_acc = {}
    for k, v in cv_scores.items():
        print(k)
        print(type(v))
        test_acc[k+'_mean']=np.mean(v)
        test_acc[k+'_std']=np.std(v)
        print('Estimated accuracy %.3f (%.3f)' % (np.mean(v)*100, np.std(v)*100))
        print('Estimated accuracy %.3f (%.3f)' % (np.mean(v), np.std(v)))

    print('print normal:')
    print(test_acc)
    print('json dumps:')
    print(json.dumps(
        test_acc,
        default=json_util.default, sort_keys=True,
        indent=1, separators=(',', ': ')
    ))
    # save mean of cross validation results
    if not os.path.exists("logs/cv_results_mean.txt"):
        f = open("logs/cv_results_mean.txt", "a+")
        f.write('### Log file for tests on ' +sys.argv[0]+ ' with standardized hmm profiles. \n\n')
        f.close()

    f = open("logs/cv_results_mean.txt", "a")

    for k, v in test_acc.items():
        f.write(str(k) + ": "+"%.5f\t"%v)
    f.write('\n')
    f.write("Weights are saved to: " + weights_file + "\n")
    f.write('-----------------------\n\n')
    f.close()

    #save all history of scores used to calculate cross validation score
    if not os.path.exists("logs/cv_results.txt"):
        f = open("logs/cv_results.txt", "a+")
        f.write('### Log file for tests on ' +sys.argv[0]+ ' with standardized hmm profiles. \n\n')
        f.close()

    f = open("logs/cv_results.txt", "a+")

    for i, k, v in enumerate(cv_scores.items()):
        f.write(str(k) + ": " + str(v))
        if i%2==0:
            f.write('\n')
    f.write("\n")
    f.write("Weights are saved to: " + weights_file + "\n")
    f.write('-----------------------\n\n')

    f.close()

    '''
        for k, v in test_acc:
        print(k, v)

        f.write('%.3f (%.3f)\t'+ weights_file + "\n" % (test_acc, np.std(cv_scores)))
        f = open("logs/cv_results.txt", "a+")
        score_text = ""
        for score in cv_scores:
            score_text += score+"\t"
        f.write(weights_file+"\t"+score_text+"\n")
    '''

else:
    X_train_aug, y_train, X_val_aug, y_val = train_val_split(hmm, X_train_aug, y_train, tv_perc)

    model, history = train_model(X_train_aug, y_train, X_val_aug, y_val, epochs=epochs)
    test_acc = evaluate_model(model, load_file)

time_end = time.time() - start_time
m, s = divmod(time_end, 60)
print("The program needed {:.0f}s to load the data and {:.0f}min {:.0f}s in total.".format(time_data, m, s))

#telegram_me(m, s, sys.argv[0], test_acc, hmm, standardize, normalize, no_input)

