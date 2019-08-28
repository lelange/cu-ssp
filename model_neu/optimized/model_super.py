from keras.models import *
from keras.layers import *
from keras import backend as K
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler

import time
import dill as pickle
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval

from hyperutils import *
import numpy as np
import keras
from keras.layers.core import K  # import keras.backend as K
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, GRU, Conv1D, CuDNNLSTM, concatenate
from keras.optimizers import Adam, Nadam, RMSprop
from hyperopt import STATUS_OK, STATUS_FAIL
from datetime import datetime
import traceback
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

WEIGHTS_DIR = "weights_super/"
DATA_ROOT = '/nosave/lange/cu-ssp/data/data_princeton/'
MODEL_NAME = "mod_super"

MAXLEN_SEQ = 700

NB_CLASSES_Q8 = 8
NB_CLASSES_RSA = 4


EPOCHS = 100

OPTIMIZER_STR_TO_CLASS = {
    'Adam': Adam,
    'Nadam': Nadam,
    'RMSprop': RMSprop
}

LOSS_STR_TO_CLASS = {
    'categorical_crossentropy': 'categorical_crossentropy',
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
    X_train, y_train = get_data('')

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
    score = evaluate_model(model, weights_save_path, hype_space)
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

n_words = 9

def build_model(hype_space):
    """Create model according to the hyperparameter space given."""
    print("Hyperspace:")
    print(hype_space)

    #get all inputs but only use according to hyperparameter
    input_onehot = Input(shape=(None, n_words))
    input_seqs = Input(shape=(None,))
    input_pssm = Input(shape=(None, 21))

    if hype_space['input']=='onehot':
        x0 = input_onehot
    if hype_space['input']=='seqs':
        # have to use embedding
        x0 = Embedding(input_dim=n_words, output_dim=250, input_length=None)(input_seqs)

    if hype_space['use_profiles'] is not None:
        if hype_space['use_profiles']=='pssm':
            x0 = concatenate([x0, input_pssm])

    x1 = x0
    if hype_space['first_layer']['type'] == 'LSTM':
        for i in range(hype_space['first_layer']['nb']):
            print(i)
            x1 = Bidirectional(CuDNNLSTM(units=int(hype_space['first_layer']['units']/i), return_sequences=True))(x1)

    # wenn model gestackt x2=x1, sonst x2 = x0
    x2 = x0
    if hype_space['first_layer']['type']=='GRU':
        x2 = Bidirectional(CuDNNGRU(units=int(hype_space['gru1']), return_sequences=True))(x2)
        if hype_space['first_layer']['gru2']:
            x2 = Bidirectional(CuDNNGRU(units=int(hype_space['gru2']['gru2_units']), return_sequences=True))(x2)
        if hype_space['first_layer']['gru2'] and hype_space['first_layer']['gru2']['gru3']:
            x2 = Bidirectional(CuDNNGRU(units=int(hype_space['gru2']['gru3']['gru3_units']), return_sequences=True))(x2)



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
            lr=0.001 * hype_space['lr_rate_mult']
        ),
        loss=hype_space['loss'],
        metrics=[accuracy]
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