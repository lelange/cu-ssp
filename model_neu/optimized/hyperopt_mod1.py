import numpy as np
from numpy import array
import pandas as pd
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, GRU, Conv1D, CuDNNLSTM, concatenate
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
import tensorflow as tf
from keras import optimizers, initializers, constraints, regularizers
from keras.engine.topology import Layer
from tensorflow.keras.layers import Activation
from tensorflow.layers import Flatten
from keras.callbacks import EarlyStopping ,ModelCheckpoint
from keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau
from datetime import datetime

import sys
import os
import traceback
import time
import dill as pickle

from utils import *

from keras.utils import plot_model
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, space_eval, STATUS_FAIL

from mod_1 import build_and_train, build_model, MODEL_NAME

SAVE_RESULTS = "results_mod1.pkl" # save trials of optimization
SAVE_BEST_PLOT = "model_1_best" # save best NN graph

space = {
    # This loguniform scale will multiply the learning rate, so as to make
    # it vary exponentially, in a multiplicative fashion rather than in
    # a linear fashion, to handle his exponentialy varying nature:
    'lr_rate_mult': hp.loguniform('lr_rate_mult', -0.5, 0.5),
    # L2 weight decay:
    'batch_size': hp.quniform('batch_size', 100, 450, 2),
    # Choice of optimizer:
    'optimizer': hp.choice('optimizer', ['Adam', 'Nadam', 'RMSprop']),
    # Kernel size for convolutions:
    'super_conv_filter_size': hp.quniform('super_conv_filter_size', 8, 128, 8),
    # LSTM units:
    'GRU_units_mult': hp.loguniform('GRU_units_mult', -0.6, 0.6),
    # Number of super_conv+conv layers stacked:
    'nb_conv_super_layers': hp.choice('nb_conv_super_layers', [2, 3, 4]),
    # Uniform distribution in finding appropriate dropout values, conv layers
    'dropout': hp.uniform('dropout', 0.0, 0.7),

}

'''
     # Use batch normalisation at more places?
    'use_BN': hp.choice('use_BN', [False, True]),
    # Uniform distribution in finding appropriate dropout values, conv layers
    'conv_dropout_drop_proba': hp.uniform('conv_dropout_proba', 0.0, 0.35),
    # Uniform distribution in finding appropriate dropout values, FC layers
    'fc_dropout_drop_proba': hp.uniform('fc_dropout_proba', 0.0, 0.6),
    '''

def plot(hyperspace, file_name_prefix):
    """Plot a model from it's hyperspace."""
    model = build_model(hyperspace)
    plot_model(
        model,
        to_file='{}.png'.format(file_name_prefix),
        show_shapes=True
    )
    print("Saved model visualization to {}.png.".format(file_name_prefix))
    K.clear_session()
    del model

def plot_best_model():
    """Plot the best model found yet."""
    space_best_model = load_best_hyperspace(name=MODEL_NAME)
    if space_best_model is None:
        print("No best model to plot. Continuing...")
        return

    print("Best hyperspace yet:")
    print_json(space_best_model)
    plot(space_best_model, SAVE_BEST_PLOT)

def optimize_model(hyperspace):
    """Build model 1 and train it."""

    try:
        model, model_name, result, _ = build_and_train(hyperspace)

        # Save training results to disks with unique filenames
        save_json_result(model_name, result)

        K.clear_session()
        del model

        return result

    except Exception as err:
        try:
            K.clear_session()
        except:
            pass
        err_str = str(err)
        print(err_str)
        traceback_str = str(traceback.format_exc())
        print(traceback_str)
        return {
            'status': STATUS_FAIL,
            'err': err_str,
            'traceback': traceback_str
        }
    print("\n\n")


def run_a_trial():
    """Run one TPE meta optimisation step and save its results."""
    max_evals = nb_evals = 1

    print("Attempt to resume a past training if it exists:")

    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open(SAVE_RESULTS, "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        # empty results.pkl
        trials = Trials()
        print("Starting from scratch: new trials.")

    best = fmin(
        optimize_model,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    pickle.dump(trials, open(SAVE_RESULTS, "wb"))

    print("\nOPTIMIZATION STEP COMPLETE.\n")

if __name__ == "__main__":
    """Plot the model and run the optimisation forever (and saves results)."""

    while True:

        # Optimize a new model with the TPE Algorithm:
        print("OPTIMIZING NEW MODEL:")
        try:
            run_a_trial()
        except Exception as err:
            err_str = str(err)
            print(err_str)
            traceback_str = str(traceback.format_exc())
            print(traceback_str)

        print("PLOTTING BEST MODEL:")
        plot_best_model()

