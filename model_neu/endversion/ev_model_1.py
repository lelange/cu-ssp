"""
model description

"""

""" imports """

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

from utils import *
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
start_time = time.time()

""" important variables """

# ------ adjust

best_weights = "model/mod_1-CB513-2019_07_23-17_11.h5"
save_pred_file = "_pred_1.npy"
MODEL_NAME = 'mod_1'
MAXLEN_SEQ = 700 # only use sequences to this length and pad to this length, choose from 600, 608, 700

batch_size = 128
default_epochs = 75

# ------ do not change

data_root = '../data/netsurfp/'
weights_file = MODEL_NAME+"-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+".h5"
load_file = "./model/"+weights_file
file_scores = "logs/cv_results.txt"
file_scores_mean = "logs/cv_results_mean.txt"

if MAXLEN_SEQ is None:
    ending = "full"
else:
    ending= str(MAXLEN_SEQ)

file_train = 'train_' + ending
file_test = ['cb513_'+ ending, 'ts115_'+ ending, 'casp12_'+ ending]


N_FOLDS = 10 # for cross validation
NB_CLASSES_Q8 = 9 # number Q8 classes, used in final layer for classification (one extra for empty slots)
NB_CLASSES_Q3 = 3 # number Q3 classes
NB_AS = 20 # number of amino acids, length of one-hot endoded amino acids vectors
NB_FEATURES = 30 # feature dimension

args = parse_arguments(default_epochs=default_epochs)

normalize = args.normalize
standardize = args.standardize
hmm = args.hmm
pssm = args.pssm
embedding = args.embedding
epochs = args.epochs
primary = not args.no_input
optimize = args.optimize
cross_validate = args.cv
#tv_perc = args.tv_perc
#plot = args.plot
test_mode = args.test_mode
predict_only = args.predict

if test_mode:
    N_FOLDS = 2
    epochs = 2


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
    input = Input(shape=(None, NB_AS,))

    if hmm or pssm:
        profiles_input = Input(shape=(None, NB_FEATURES,))
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

# --------------------------------- main ---------------------------------