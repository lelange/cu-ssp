from keras.models import *
from keras.layers import *
from keras import backend as K
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler

import dill as pickle
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval

from utils import *

import numpy as np
import keras
from keras.layers.core import K  # import keras.backend as K
from keras.optimizers import Adam, Nadam, RMSprop
from hyperopt import STATUS_OK, STATUS_FAIL
from datetime import datetime
import traceback
import sys
import os
import time
import tensorflow as tf
sys.path.append('keras-tcn')
from tcn import tcn


data_root = '/nosave/lange/cu-ssp/data/'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TENSORBOARD_DIR = "TensorBoard/"
WEIGHTS_DIR = "weights/"

MAXLEN_SEQ = 700
NB_CLASSES_Q8 = 9
NB_FEATURES = 30
MODEL_NAME = "mod_6_w2v"

epochs = 75
batch_size = 128


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

    # TensorBoard logging callback (see model 6):
    log_path = None

    emb_dim = int(hype_space['embed_dim'])
    window_size = int(hype_space['window_size'])
    nb_neg = int(hype_space['negative'])
    nb_iter = int(hype_space['iter'])
    n_gram = int(hype_space['n_gram'])
    mod = int(hype_space['model'])
    #tokens = int(hype_space['tokens'])

    #standardize train and val profiles
    X_train, y_train, X_aug = get_netsurf_data('train_full')

    X_train, y_train, X_aug, X_val, y_val, X_aug_val = train_val_split_(X_train, y_train, X_aug)

    ## load data and get embedding form train data, embed train+val
    index2embed = get_embedding(emb_dim, window_size, nb_neg, nb_iter, n_gram, mod,
                                seqs=X_train)

    X_train_embed = embed_data(X_train, index2embed, emb_dim, n_gram)
    X_val_embed = embed_data(X_val, index2embed, emb_dim, n_gram)

    X_train_aug = [X_train_embed, X_aug]
    X_val_aug = [X_val_embed, X_aug_val]

    print('We have '+str(len(callbacks))+' callbacks.')

    # Train net:
    history = model.fit(
        X_train_aug,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        verbose=2,
        callbacks=callbacks,
        validation_data=(X_val_aug, y_val)
    ).history


    # Test net:
    score = evaluate_model(model, weights_save_path,
                           emb_dim, n_gram, index2embed)
    K.set_learning_phase(0)
    print("\n\n")
    min_loss = min(history['val_loss'])
    max_acc = max(history['val_accuracy'])
    number_of_epochs_it_ran = len(history['loss'])

    model_name = MODEL_NAME+"_{}_{}".format(str(max_acc), time_str)
    print("Model name: {}".format(model_name))

    print('Score: ', score)
    result = {
        # We plug "-val_accuracy" as a minimizing metric named 'loss' by Hyperopt.
        'loss': -max_acc,
        'real_loss': min_loss,
        'cb513': score['cb513_full'],
        'casp12':score['casp12_full'],
        'ts115':score['ts115_full'],
        'nb_epochs': number_of_epochs_it_ran,
        # Misc:
        'model_name': model_name,
        'space': hype_space,
        'status': STATUS_OK
    }

    print("RESULT:")
    print_json(result)

    # save test results to logfile
    f = open("/nosave/lange/cu-ssp/model_neu/optimized/logs/test_results_mod6_w2v.txt", "a+")
    res = ""
    for k, v in score.items():
        res += str(k)+": "+str(v)+"\t"
    f.write("\n"+str(model_weight_name)+"\t"+ res)
    f.close()

    return model, model_name, result, log_path


""" Build model """

def build_model(hype_space):
    """Create model according to the hyperparameter space given."""
    print("Hyperspace:")
    print(hype_space)

    input = Input(shape=(MAXLEN_SEQ, int(hype_space['embed_dim']) ))
    profiles_input = Input(shape=(MAXLEN_SEQ, NB_FEATURES,))
    x = input
    #conv = int(hype_space['embed_dim'])
    z = Conv1D(64, 11, strides=1, padding='same')(x)
    w = Conv1D(64, 7, strides=1, padding='same')(x)
    x = concatenate([x, z], axis=2)
    x = concatenate([x, w], axis=2)

    z = Conv1D(64, 5, strides=1, padding='same')(x)
    w = Conv1D(64, 3, strides=1, padding='same')(x)
    x = concatenate([x, z], axis=2)
    x = concatenate([x, w], axis=2)

    x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(units=64, return_sequences=True))(x)
    y = TimeDistributed(Dense(NB_CLASSES_Q8, activation="softmax"))(x)

    # y_q3 = TimeDistributed(Dense(3, activation="softmax"), name="y_q3")(x)
    inp = [input, profiles_input]
    model = Model(inp, y)
    model.compile(optimizer='RMSprop', loss="categorical_crossentropy", metrics=[accuracy])
    # model.summary()

    return model
