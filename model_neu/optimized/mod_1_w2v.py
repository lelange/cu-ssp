from keras.models import *
from keras.layers import *
from keras import backend as K
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler

import time
import dill as pickle
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval

from utils import *

from utils import print_json, train_val_split, accuracy, get_test_data
import numpy as np
import keras
from keras.layers.core import K  # import keras.backend as K
from keras.optimizers import Adam, Nadam, RMSprop
from hyperopt import STATUS_OK, STATUS_FAIL
from datetime import datetime
import traceback
import os

import multiprocessing
from gensim.models import Word2Vec

data_root = '/nosave/lange/cu-ssp/data/'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TENSORBOARD_DIR = "TensorBoard/"
WEIGHTS_DIR = "weights/"

MAXLEN_SEQ = 700
NB_CLASSES_Q8 = 9
NB_FEATURES = 30
MODEL_NAME = "mod_1_w2v"
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

    # TensorBoard logging callback (see model 6):
    log_path = None
    emb_dim = int(hype_space['embed_dim'])
    window_size = int(hype_space['window_size'])
    nb_neg = int(hype_space['negative'])
    nb_iter = int(hype_space['iter'])
    n_gram = int(hype_space['n_gram'])
    #model = int(hype_space['model'])
    #tokens = int(hype_space['tokens'])
    mod = 0

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
    f = open("/nosave/lange/cu-ssp/model_neu/optimized/logs/test_results_mod1_w2v.txt", "a+")
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

    def conv_block(x, activation=True, batch_norm=True, drop_out=True, res=True):
        cnn = Conv1D(64, 11, padding="same")(x)
        if activation: cnn = TimeDistributed(Activation("relu"))(cnn)
        if batch_norm: cnn = TimeDistributed(BatchNormalization())(cnn)
        if drop_out:   cnn = TimeDistributed(Dropout(0.5))(cnn)
        if res:        cnn = Concatenate(axis=-1)([x, cnn])

        return cnn

    input = Input(shape=(MAXLEN_SEQ, int(hype_space['embed_dim']) ))
    profiles_input = Input(shape=(MAXLEN_SEQ, NB_FEATURES))
    x = concatenate([input, profiles_input])
    inp = [input, profiles_input]

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
        metrics=[accuracy])

    return model
