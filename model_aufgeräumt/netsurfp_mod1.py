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

import time
import dill as pickle
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval

from utils import *

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
load_file = "./model/mod_1-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+".h5"

file_train = 'train_608'
file_test = ['cb513_608', 'ts115_608', 'casp12_608']

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

    return model


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
    model.summary()

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
        print("####evaluate" + file_test[i] +":")
        score = model.evaluate(X_test_aug, y_test, verbose=2, batch_size=1)
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
    X_train_aug, y_train, X_val_aug, y_val = train_val_split(hmm, X_train_aug, y_train, tv_perc)

    if optimize:

        '''
        #---- create a Trials database to store experiment results
        trials = Trials()
        #---- use that Trials database for fmin
        best = fmin(build_model_ho, space, algo=tpe.suggest, trials=trials, max_evals=100, rstate=np.random.RandomState(99))
        #---- save trials
        pickle.dump(trials, open("./trials/mod_3-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+"-hyperopt.p", "wb"))
        #trials = pickle.load(open("./trials/mod_3-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+"-hyperopt.p", "rb"))
        print('Space evaluation: ')
        space_eval(space, best)
        data = list(map(_flatten, extract_params(trials)))
        df = pd.DataFrame(list(data))
        df = df.fillna(0)  # missing values occur when the object is not populated
        corr = df.corr()
        print(corr)
        '''

    else:
        model = train_model(X_train_aug, y_train, X_val_aug, y_val, epochs=epochs)
        test_acc = evaluate_model(model, load_file)

time_end = time.time() - start_time
m, s = divmod(time_end, 60)
print("The program needed {:.0f}s to load the data and {:.0f}min {:.0f}s in total.".format(time_data, m, s))

telegram_me(m, s, sys.argv[0], test_acc, hmm, standardize, normalize, no_input)

