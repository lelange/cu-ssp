import sys
from keras.models import *
from keras.layers import *

from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
sys.path.append('keras-tcn')
from tcn import tcn

import time
import dill as pickle
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval

from hyperutils import *
import numpy as np
import keras
from keras.layers.core import K  # import keras.backend as K
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, GRU, Conv1D, CuDNNLSTM, concatenate, Dropout
from keras.optimizers import Adam, Nadam, RMSprop
from hyperopt import STATUS_OK, STATUS_FAIL
from datetime import datetime
import traceback
import os


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

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from keras import regularizers, constraints, initializers, activations

from keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, CuDNNGRU, Concatenate
from keras.layers import Dropout, Flatten, Activation, RepeatVector, Permute, Conv1D, BatchNormalization

from keras.layers import Dropout
from keras.layers import merge
from keras.layers.merge import concatenate
from keras.layers.recurrent import Recurrent
from keras.metrics import categorical_accuracy
from keras.models import Model, Input, Sequential
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

WEIGHTS_DIR = "weights_super_v2/"
DATA_ROOT = '/nosave/lange/cu-ssp/data/data_princeton/'
MODEL_NAME = "mod_super_v2"

MAXLEN_SEQ = 700

NB_CLASSES_Q8 = 8
#NB_CLASSES_RSA = 4

STARTING_L2_REG = 0.0007
EPOCHS = 100

OPTIMIZER_STR_TO_CLASS = {
    'Adam': Adam,
    'Nadam': Nadam,
    'RMSprop': RMSprop
}

LOSS_STR_TO_CLASS = {
    'categorical_crossentropy': 'categorical_crossentropy',
    'mean_squared_error': 'mean_squared_error',
    'nll': nll
}



def build_and_train(hype_space, save_best_weights=True):
    """Build the model and train it."""
    start_time = time.time()
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
            filepath=weights_save_path,
            monitor='val_accuracy',
            verbose = 1,
            save_best_only=True, mode='max'))

    callbacks.append(EarlyStopping(
        monitor='val_accuracy',
        patience=10, verbose=1, mode='max'))

    callbacks.append(ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5,
        patience=10, verbose=1, mode='max', cooldown = 2))

    #standardize train and val profiles
    X_train, y_train, X_test, y_test = get_data()

    # Train net:
    history = model.fit(
        X_train,
        y_train,
        batch_size=int(hype_space['batch_size']),
        epochs=EPOCHS,
        shuffle=True,
        verbose=2,
        callbacks=callbacks,
        validation_split=0.1
    ).history

    end_time = time.time() - start_time

    # evaluate on cb513:
    score = evaluate_model(model, weights_save_path, hype_space, X_test, y_test)
    K.set_learning_phase(0)

    print("\n\n")

    min_loss = min(history['val_loss'])
    max_acc = max(history['val_accuracy'])
    number_of_epochs_it_ran = len(history['loss'])

    model_name = MODEL_NAME+"_{}_{}".format(str(score['cb513']), time_str)
    print("Model name: {}".format(model_name))

    result = {
        # We plug "-val_accuracy" as a minimizing metric named 'loss' by Hyperopt.
        'loss': -max_acc,
        'real_loss': min_loss,
        'cb513': score['cb513'],
        'nb_epochs': number_of_epochs_it_ran,
        'accuracy_history':history['val_accuracy'],
        'time_in_sec':end_time,
        # Misc:
        'model_name': model_name,
        'weight_path' : weights_save_path,
        'space': hype_space,
        'status': STATUS_OK
    }

    print("RESULT:")
    print_json(result)

    return model, model_name, result


""" Build model """

n_words = 22
n_tags = 9

def build_model(hype_space):
    """Create model according to the hyperparameter space given."""
    print("Hyperspace:")
    print(hype_space)

    #use different inputs according to choice of hyperparameter 'input', 'use profiles'
    input_onehot = Input(shape=(None, n_words))
    input_seqs = Input(shape=(None,))
    input_pssm = Input(shape=(None, 22))
    input_hmm = Input(shape=(None, 30))
    inp = [input_onehot, input_seqs, input_pssm, input_hmm]
    x0=None
    if hype_space['input']=='onehot':
        x0 = input_onehot
    if hype_space['input']=='seqs':
        # have to use embedding
        x0 = Embedding(input_dim=n_words, output_dim=int(hype_space['dense_output']), input_length=None)(input_seqs)
    if hype_space['input']=='both':
        x_seq = Embedding(input_dim=n_words, output_dim=int(hype_space['dense_output']), input_length=None)(input_seqs)
        x0 = concatenate([input_onehot, x_seq])


    if hype_space['use_profiles'] is not None:
        if hype_space['use_profiles']=='pssm':
            x0 = concatenate([x0, input_pssm])
        if hype_space['use_profiles']=='hmm':
            x0 = concatenate([x0, input_hmm])
        if hype_space['use_profiles']=='both':
            x0 = concatenate([x0, input_pssm, input_hmm])

    # NN starts here:
    if hype_space['tcn_position']=='first':
        x0 = tcn.TCN(return_sequences=True)(x0)

    x1 = x0
    if hype_space['first_layer']['type'] == 'LSTM':
        for i in range(hype_space['first_layer']['lstm_nb']):
            i=i+1
            print(i)
            x1 = Bidirectional(CuDNNLSTM(units=int(hype_space['first_layer']['lstm_units']/i), return_sequences=True))(x1)
            x1 = Dropout(int(hype_space['dropout']))(x1)
    x2=x1

    x1 = x0
    if hype_space['first_layer']['type']=='GRU':
        x1 = Bidirectional(CuDNNGRU(units=int(hype_space['first_layer']['gru1']*100), return_sequences=True))(x1)
        if hype_space['first_layer']['gru2']:
            x1 = Bidirectional(CuDNNGRU(units=int(hype_space['first_layer']['gru2']['gru2_units']*100), return_sequences=True))(x1)
        if hype_space['first_layer']['gru2'] and hype_space['first_layer']['gru2']['gru3']:
            x1 = Bidirectional(CuDNNGRU(units=int(hype_space['first_layer']['gru2']['gru3']['gru3_units']*100), return_sequences=True))(x1)
        x2=x1
    x1 = x0
    if hype_space['first_layer']['type'] == 'conv':
        for i in range(hype_space['first_layer']['nb_conv_layers']):
            i = i + 1
            print(i)
            x1 = keras.layers.convolutional.Conv1D(
            filters=int(hype_space['first_layer']['nb_filter']), kernel_size=int(hype_space['first_layer']['conv_filter_size']), strides=i,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(
                STARTING_L2_REG * hype_space['l2_weight_reg_mult']))(x1)
            x1 = Dropout(int(hype_space['dropout']))(x1)
        x2 = x1

    COMBO_MOVE = concatenate([x0, x2])

    x0=COMBO_MOVE
    if hype_space['second_layer']:
        x1 = x0
        if hype_space['second_layer']['type'] == 'LSTM':
            for i in range(hype_space['second_layer']['lstm_nb_2']):
                i = i + 1
                print(i)
                x1 = Bidirectional(
                    CuDNNLSTM(units=int(hype_space['second_layer']['lstm_units_2'] / i), return_sequences=True))(x1)
                x1 = Dropout(int(hype_space['dropout']))(x1)
        x2 = x1

        x1 = x0
        if hype_space['second_layer']['type'] == 'GRU':
            x1 = Bidirectional(CuDNNGRU(units=int(hype_space['second_layer']['gru1_2'] * 100), return_sequences=True))(
                x1)
            if hype_space['second_layer']['gru2_2']:
                x1 = Bidirectional(
                    CuDNNGRU(units=int(hype_space['second_layer']['gru2_2']['gru2_units_2'] * 100),
                             return_sequences=True))(x1)
            if hype_space['second_layer']['gru2_2'] and hype_space['second_layer']['gru2_2']['gru3_2']:
                x1 = Bidirectional(
                    CuDNNGRU(units=int(hype_space['second_layer']['gru2_2']['gru3_2']['gru3_units_2'] * 100),
                             return_sequences=True))(x1)
            x2 = x1
        x1 = x0
        if hype_space['second_layer']['type'] == 'conv':
            for i in range(hype_space['second_layer']['nb_conv_layers_2']):
                i = i + 1
                print(i)
                x1 = keras.layers.convolutional.Conv1D(
                    filters=int(hype_space['second_layer']['nb_filter_2']),
                    kernel_size=int(hype_space['second_layer']['conv_filter_size_2']) * i, strides=i,
                    padding='same',
                    kernel_regularizer=keras.regularizers.l2(
                        STARTING_L2_REG * hype_space['l2_weight_reg_mult']))(x1)
                x1 = Dropout(int(hype_space['dropout']))(x1)
            x2 = x1

        COMBO_MOVE = concatenate([x0, x2])

    '''
    current_layer = input

    if hype_space['first_conv'] is not None:
        k = hype_space['first_conv']
        current_layer = keras.layers.convolutional.Conv1D(
            filters=16, kernel_size=k, strides=1,
            padding='same', activation=hype_space['activation'],
            kernel_regularizer=keras.regularizers.l2(
                STARTING_L2_REG * hype_space['l2_weight_reg_mult'])
        )(current_layer)

    # Core loop that stacks multiple conv+pool layers, with maybe some
    # residual connections and other fluffs:
    n_filters = int(40 * hype_space['conv_hiddn_units_mult'])
    for i in range(hype_space['nb_conv_pool_layers']):
        print(i)
        print(n_filters)
        print(current_layer._keras_shape)

        current_layer = convolution(current_layer, n_filters, hype_space)
        if hype_space['use_BN']:
            current_layer = bn(current_layer)
        print(current_layer._keras_shape)
        n_filters *= 2
    # Fully Connected (FC) part:

    current_layer = TimeDistributed(Dense(
        units=int(1000 * hype_space['fc_units_1_mult']),
        activation=hype_space['activation'],
        kernel_regularizer=keras.regularizers.l2(
            STARTING_L2_REG * hype_space['l2_weight_reg_mult'])
    ))(current_layer)
    print(current_layer._keras_shape)

    current_layer = dropout(
        current_layer, hype_space, for_convolution_else_fc=False)

    if hype_space['one_more_fc'] is not None:
        current_layer = TimeDistributed(Dense(
            units=int(750 * hype_space['one_more_fc']),
            activation=hype_space['activation'],
            kernel_regularizer=keras.regularizers.l2(
                STARTING_L2_REG * hype_space['l2_weight_reg_mult'])
        ))(current_layer)
        print(current_layer._keras_shape)

        current_layer = dropout(
            current_layer, hype_space, for_convolution_else_fc=False)

    y = TimeDistributed(Dense(
        units=NB_CLASSES_Q8,
        activation="softmax",
        kernel_regularizer=keras.regularizers.l2(
            STARTING_L2_REG * hype_space['l2_weight_reg_mult']),
        name='y'
    ))(current_layer)

    print(y._keras_shape)

    # Finalize model:
    inp = [input, profiles_input]
    model = Model(inp, y)

    model.compile(
        optimizer=OPTIMIZER_STR_TO_CLASS[hype_space['optimizer']](
            #lr=0.001 * hype_space['lr_rate_mult']
        ),
        loss=LOSS_STR_TO_CLASS[hype_space['loss']],
        metrics=[accuracy] #noch andere dazu
    )
    '''
    w = Dense(int(hype_space['dense_output'])*2, activation="relu")(COMBO_MOVE)  # try 500
    w = Dropout(int(hype_space['dropout']))(w)
    if hype_space['tcn_position']=='last':
        w = tcn.TCN(return_sequences=True)(w)

    y = TimeDistributed(Dense(n_tags, activation="softmax"))(w)

    # Defining the model as a whole and printing the summary
    model = Model(inp, y)
    # model.summary()

    # Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
    #adamOptimizer = Adam(lr=0.001, beta_1=0.8, beta_2=0.8, epsilon=None, decay=0.0001, amsgrad=False)
    model.compile(
        optimizer=OPTIMIZER_STR_TO_CLASS[hype_space['optimizer']](
            lr=0.001 * hype_space['lr_rate_mult']
        ),
        loss=LOSS_STR_TO_CLASS[hype_space['loss']],
        metrics=[accuracy,
                 #weighted_accuracy,
                 kullback_leibler_divergence,
                 matthews_correlation,
                 precision,
                 recall,
                 fbeta_score
                 ]
    )
    return model


'''
model = None
    if embedding_layer:
        input = Input(shape=(None,))
    else:
        input = Input(shape=(None,n_words))
    profiles_input = Input(shape=(None, X_aug_train.shape[2]))

    # Defining an embedding layer mapping from the words (n_words) to a vector of len 250
    if embedding_layer:
        x1 = Embedding(input_dim=n_words, output_dim=250, input_length=None)(input)
    else:
        x1 = input

    x1 = concatenate([x1, profiles_input])
    x1 = Dense(1200, activation="relu")(x1)
    x1 = Dropout(0.5)(x1)

    # Defining a bidirectional GRU using the embedded representation of the inputs
    x1 = Bidirectional(CuDNNGRU(units=500, return_sequences=True))(x1)
    x1 = Bidirectional(CuDNNGRU(units=100, return_sequences=True))(x1)

    if embedding_layer:
        x2 = Embedding(input_dim=n_words, output_dim=125, input_length=None)(input)
    else:
        x2 = input
    x2 = concatenate([x2, profiles_input])

    # Defining a bidirectional GRU using the embedded representation of the inputs
    x2 = Bidirectional(CuDNNGRU(units=500, return_sequences=True))(x2)
    x2 = Bidirectional(CuDNNGRU(units=100, return_sequences=True))(x2)
    COMBO_MOVE = concatenate([x1, x2])

    w = Dense(500, activation="relu")(COMBO_MOVE)  # try 500
    w = Dropout(0.4)(w)
    w = tcn.TCN(return_sequences=True)(w)

    y = TimeDistributed(Dense(n_tags, activation="softmax"))(w)

    # Defining the model as a whole and printing the summary
    model = Model([input, profiles_input], y)
    # model.summary()

    # Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
    adamOptimizer = Adam(lr=0.001, beta_1=0.8, beta_2=0.8, epsilon=None, decay=0.0001, amsgrad=False)
    model.compile(optimizer=adamOptimizer, loss=nll1, metrics=["accuracy", accuracy])
    return model

'''