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

from utils import print_json, train_val_split, accuracy, get_test_data
import numpy as np
import keras
from keras.layers.core import K  # import keras.backend as K
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, GRU, Conv1D, CuDNNLSTM, concatenate
from keras.optimizers import Adam, Nadam, RMSprop
import tensorflow as tf
from hyperopt import STATUS_OK, STATUS_FAIL
from datetime import datetime
import uuid
import traceback
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TENSORBOARD_DIR = "TensorBoard/"
WEIGHTS_DIR = "weights/"

MAXLEN_SEQ = 700
NB_CLASSES_FINE = 8
NB_CLASSES_COARSE = 3
NB_FEATURES = 50
MODEL_NAME = "mod_1"

def data():
    data_root = '/nosave/lange/cu-ssp/data/netsurfp/'
    file_train = 'train_700'
    file_test = ['cb513_700', 'ts115_700', 'casp12_700']

    X_test = np.load(data_root + file_test[0] + '_input.npy')
    profiles = np.load(data_root + file_test[0] + '_hmm.npy')
    mean = np.mean(profiles)
    std = np.std(profiles)
    X_aug_test = (profiles - mean) / std
    X_test_aug = np.concatenate((X_test, X_aug_test), axis = 2)
    y_test = np.load(data_root + file_test[0] + '_q8.npy')

    X_train = np.load(data_root + file_train + '_input.npy')
    profiles = np.load(data_root + file_train + '_hmm.npy')
    mean = np.mean(profiles)
    std = np.std(profiles)
    X_aug_train = (profiles - mean) / std
    X_train_aug = [X_train, X_aug_train]
    y_train = np.load(data_root + file_train + '_q8.npy')

    X_train_aug, y_train, X_val_aug, y_val = train_val_split(X_train_aug, y_train)

    return X_train_aug, y_train, X_val_aug, y_val, X_test_aug, y_test

X_train_aug, y_train, X_val_aug, y_val, X_test_aug, y_test = data()


# You may want to reduce this considerably if you don't have a killer GPU:
EPOCHS = 60
STARTING_L2_REG = 0.0007

OPTIMIZER_STR_TO_CLASS = {
    'Adam': Adam,
    'Nadam': Nadam,
    'RMSprop': RMSprop
}

def evaluate_model(model, load_file, test_ind = None):
    file_test = ['cb513_700', 'ts115_700', 'casp12_700']
    if test_ind is None:
        test_ind = range(len(file_test))
    test_accs = []
    names = []
    for i in test_ind:
        X_test_aug, y_test = get_test_data(file_test[i])
        model.load_weights(load_file)
        score = model.evaluate(X_test_aug, y_test, verbose=2, batch_size=1)
        #print(file_test[i] +' test accuracy: ' + str(score[1]))
        test_accs.append(score[1])
        names.append(file_test[i])
    return dict(zip(names, test_accs))


def build_and_train(hype_space, save_best_weights=True):
    """Build the model and train it."""

    K.set_learning_phase(1)
    model = build_model(hype_space)

    time_str = datetime.now().strftime("%Y_%m_%d-%H_%M")
    model_weight_name = MODEL_NAME+"-" + time_str

    callbacks = []

    # Weight saving callback:
    if save_best_weights:
        weights_save_path = os.path.join(
            WEIGHTS_DIR, '{}.hdf5'.format(model_weight_name))
        print("Model's weights will be saved to: {}".format(weights_save_path))
        if not os.path.exists(WEIGHTS_DIR):
            os.makedirs(WEIGHTS_DIR)

        callbacks.append(ModelCheckpoint(
            weights_save_path,
            monitor='val_accuracy',
            save_best_only=True, mode='max'))

    callbacks.append(EarlyStopping(
        monitor='val_accuracy',
        patience=10, verbose=1, mode='max'))

    # TensorBoard logging callback (see model 6):
    log_path = None

    # Train net:
    history = model.fit(
        X_train_aug,
        y_train,
        batch_size=int(hype_space['batch_size']),
        epochs=EPOCHS,
        shuffle=True,
        verbose=2,
        callbacks=callbacks,
        validation_data=(X_val_aug, y_val)
    ).history

    # Test net:
    K.set_learning_phase(0)
    score = evaluate_model(model, weights_save_path)
    print("\n\n")
    max_acc = max(history['val_accuracy'])

    model_name = MODEL_NAME+"_{}_{}".format(str(max_acc), time_str)
    print("Model name: {}".format(model_name))

    print(history.keys())
    print(history)
    print('Score: ', score)
    result = {
        # We plug "-val_accuracy" as a minimizing metric named 'loss' by Hyperopt.
        'loss': -max_acc,
        # Misc:
        'model_name': model_name,
        'space': hype_space,
        'status': STATUS_OK
    }

    print("RESULT:")
    print_json(result)

    # save test results to logfile
    f = open("/nosave/lange/cu-ssp/model_neu/optimized/logs/test_results_mod1.txt", "a+")
    res = ""
    for k, v in score.items():
        res += str(k)+": "+str(v)+"\t"
    f.write("\n"+str(model_weight_name)+"\t"+ res)
    f.close()

    return model, model_name, result, log_path


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




""" Build model """



def build_model(hype_space):
    """Create model according to the hyperparameter space given."""
    print("Hyperspace:")
    print(hype_space)

    def super_conv_block(x):
        # kennt er den hype_space?
        conv = int(hype_space['super_conv_filter_size'])

        c3 = Conv1D(conv, 1, padding="same")(x)
        c3 = TimeDistributed(Activation("relu"))(c3)
        c3 = TimeDistributed(BatchNormalization())(c3)

        c7 = Conv1D(conv * 2, 3, padding="same")(x)
        c7 = TimeDistributed(Activation("relu"))(c7)
        c7 = TimeDistributed(BatchNormalization())(c7)

        c11 = Conv1D(conv * 4, 5, padding="same")(x)
        c11 = TimeDistributed(Activation("relu"))(c11)
        c11 = TimeDistributed(BatchNormalization())(c11)

        x = Concatenate(axis=-1)([x, c3, c7, c11])
        x = TimeDistributed(Dropout(0.5))(x)
        return x

    def conv_block(x, activation=True, batch_norm=True, drop_out=True, res=True):
        conv = int(hype_space['super_conv_filter_size'])

        cnn = Conv1D(conv * 2, 11, padding="same")(x)
        if activation: cnn = TimeDistributed(Activation("relu"))(cnn)
        if batch_norm: cnn = TimeDistributed(BatchNormalization())(cnn)
        if drop_out:   cnn = TimeDistributed(Dropout(0.5))(cnn)
        if res:        cnn = Concatenate(axis=-1)([x, cnn])

        return cnn

    input_layer = keras.layers.Input((MAXLEN_SEQ, NB_FEATURES))

    for i in range(hype_space['nb_conv_super_layers']):
        x = super_conv_block(input_layer)
        x = conv_block(x)

    x = Bidirectional(CuDNNGRU(units=int(256*hype_space['GRU_units_mult']), return_sequences=True, recurrent_regularizer=l2(0.2)))(x)
    x = TimeDistributed(Dropout(hype_space['dropout']))(x)
    x = TimeDistributed(Dense(units=int(256*hype_space['GRU_units_mult']), activation="relu"))(x)
    x = TimeDistributed(Dropout(hype_space['dropout']))(x)

    q8_output = TimeDistributed(Dense(NB_CLASSES_FINE, activation="softmax"))(x)

    # Finalize model:
    model = Model(
        inputs=[input_layer],
        outputs=[q8_output]
    )
    model.compile(
        optimizer=OPTIMIZER_STR_TO_CLASS[hype_space['optimizer']](
            lr=0.001 * hype_space['lr_rate_mult']),
        loss='categorical_crossentropy',
        metrics=[accuracy]
    )

    return model
