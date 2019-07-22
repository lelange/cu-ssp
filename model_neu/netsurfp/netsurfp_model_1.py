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
from datetime import datetime

from utils import parse_arguments, get_data, train_val_split, save_cv, telegram_me, accuracy, get_acc
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_NAME = 'mod_1'
N_FOLDS = 10 # for cross validation
MAXLEN_SEQ = 700 # only use sequences to this length and pad to this length, choose from 600, 608, 700
NB_CLASSES_Q8 = 9 # number Q8 classes, used in final layer for classification (one extra for empty slots)
NB_CLASSES_Q3 = 3 # number Q3 classes
NB_AS = 20 # number of amino acids, length of one-hot endoded amino acids vectors
NB_FEATURES = 30 # feature dimension

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


def build_and_train (X_train_aug, y_train, X_val_aug, y_val, epochs = epochs):
    """
    Main Training function with the following properties:
        Optimizer - Nadam
        Loss function - Categorical Crossentropy
        Batch Size - 128 (any more will exceed Collab GPU RAM)
        Epochs - 50
    """
    model = build_model()

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

def build_model():
    input = Input(shape=(MAXLEN_SEQ, NB_AS,))
    if hmm:
        profiles_input = Input(shape=(MAXLEN_SEQ, NB_FEATURES,))
        x = concatenate([input, profiles_input])
        inp = [input, profiles_input]
    else:

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

    y = TimeDistributed(Dense(NB_CLASSES_Q8, activation="softmax"))(x)

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
best_weights = "model/mod_1-CB513-2019_07_22-12_53.h5"
save_pred_file = "_pred_1.npy"
PRED_DIR = "preds/"
q8_list = list('-GHIBESTC')
q3_list = list('-HHHEECCC')

def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        if np.max(o)!=0:
            i = np.argmax(o)
            s += index[i]
        else:
            #s += index[0]
            return s
    return s

def onehot_to_seq2(oh_seq, index):
    s = ''
    for o in oh_seq:
        i = np.argmax(o)
        s += index[i]

        if i ==0:
            return s
    return s


def accuracy2(y_true, y_predicted):
    print("understand metric:")
    #turn onehot to seq
    y = tf.argmax(y_true, axis =- 1)
    print("y shape:")
    print(y.eval().shape)
    #turn onehot to seq
    y_ = tf.argmax(y_predicted, axis =- 1)
    print("y_:")
    print(y_.eval().shape)
    mask = tf.greater(y, 0)
    print('mask len:')
    print(len(mask.eval()))
    print("tf boolean mask y len")
    print(len(tf.boolean_mask(y, mask).eval()))
    print("tf boolean mask y_ len")
    print(len(tf.boolean_mask(y_, mask).eval()))
    print(len(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)).eval()))

    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())


def build_and_predict(model, best_weights, save_pred_file, file_test=['ts115_700']):
    if model is None:
        model = build_model()

    for test in file_test:
        i=True
        X_test_aug, y_test = get_data(test, hmm=True, normalize=False, standardize=True)
        model.load_weights(best_weights)

        print("\nPredict " + test +"...")

        y_test_pred = model.predict(X_test_aug)
        score = model.evaluate(X_test_aug, y_test)
        print("Accuracy from model evaluate: "+str(score[2]))
        np.save(PRED_DIR+test+save_pred_file, y_test_pred)

        '''
        sess = tf.Session()
        with sess.as_default():
            acc = accuracy2(y_test, y_test_pred)
            print("Accuracy2: ")
            print(acc.eval()[:30])
            print(np.sum(acc.eval()))
            print(len(acc.eval()))
            print(np.sum(acc.eval())/len(acc.eval()))
            print("Test argmax (len 5, max at 3): "+str(tf.argmax(input=[2,0,1,0,0]).eval()))
            print("Test argmax (len 2): " + str(tf.argmax(input=[0]).eval()))
            
        '''

        print("Saved predictions to "+PRED_DIR+test+save_pred_file+".")
        q3_pred = 0
        q8_pred = 0
        q3_len = 0
        q8_len = 0

        f = open(PRED_DIR+"q4_pred_mod_1.txt", "a+")
        g = open(PRED_DIR+"q9_pred_mod_1.txt", "a+")
        for true, pred in zip(y_test, y_test_pred):
            seq3 = onehot_to_seq(pred, q3_list)
            seq8 = onehot_to_seq(pred, q8_list)
            seq_true_3 = onehot_to_seq(true, q3_list)
            seq_true_8 = onehot_to_seq(true, q8_list)

            if i:
                print('Q3 prediction, first pred then true: ')
                print(seq3[:60])
                print(seq_true_3[:60])
                print(seq3[-60:])
                print(seq_true_3[-60:])

                print('Q8 prediction, first pred then true: ')
                print(seq8[:60])
                print(seq_true_8[:60])
                print(seq8[-60:])
                print(seq_true_8[-60:])
                i=False

            f.write(seq3)
            g.write(seq8)
            f.write("\n")
            g.write("\n")

            corr3, len3 = get_acc(seq_true_3, seq3)
            corr8, len8 = get_acc(seq_true_8, seq8)
            q3_pred+=corr3
            q8_pred+=corr8
            q3_len+= len3
            q8_len+=len8
        f.close()
        g.close()

        print(q8_pred)
        print(q8_len)
        print("Q3 " +test+ " test accuracy: "+str(q3_pred/q3_len))
        print("Q8 " +test+ " test accuracy: "+str(q8_pred/q8_len))

        f = open(PRED_DIR+"prediction_accuracy.txt", "a+")
        f.write("Results for "+MODEL_NAME+" and weights "+best_weights)
        f.write("\n")
        f.write("Netsurf data were used with standardized hhblits profiles.")
        f.write("\n")
        f.write("Q3 " +test+ " test accuracy: "+str(q3_pred/q3_len))
        f.write("\n")
        f.write("Q8 " +test+ " test accuracy: "+str(q8_pred/q8_len))
        f.write("\n")
        f.write("Predictions are saved to: "+PRED_DIR+test+save_pred_file)
        f.write("----------------------------\n")
        f.close()


#--------------------------------- main ---------------------------------

if predict_only:
    build_and_predict(build_model(), best_weights, save_pred_file)
    test_acc = None
    time_data = time.time() - start_time
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

#telegram_me(m, s, sys.argv[0], test_acc, hmm=True, standardize=True)

