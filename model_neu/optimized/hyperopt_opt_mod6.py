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

from mod_6 import build_and_train, build_model

MODEL_NAME = "mod_6" #f or identifing the results and load the best for this model


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
    plot(space_best_model, "model_best")

space = {
    # This loguniform scale will multiply the learning rate, so as to make
    # it vary exponentially, in a multiplicative fashion rather than in
    # a linear fashion, to handle his exponentialy varying nature:
    'lr_rate_mult': hp.loguniform('lr_rate_mult', -0.5, 0.5),
    # L2 weight decay:
    'batch_size': hp.quniform('batch_size', 100, 450, 5),
    # Choice of optimizer:
    'optimizer': hp.choice('optimizer', ['Adam', 'Nadam', 'RMSprop']),
    # Kernel size for convolutions:
    'conv_filter_size': hp.quniform('conv_filter_size', 32, 128, 32),
    # LSTM units:
    'LSTM_units_mult': hp.loguniform('LSTM_units_mult', -0.6, 0.6),
    # Use batch normalisation at more places?
    'use_BN': hp.choice('use_BN', [False, True]),
}

def optimize_model(hyperspace):
    """Build model 6 and train it."""

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
        trials = pickle.load(open("results.pkl", "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new trials.")

    best = fmin(
        optimize_model,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    pickle.dump(trials, open("results.pkl", "wb"))

    print("\nOPTIMIZATION STEP COMPLETE.\n")

if __name__ == "__main__":
    """Plot the model and run the optimisation forever (and saves results)."""

    print("Now, we train many models, one after the other. "
          "Note that hyperopt has support for cloud "
          "distributed training using MongoDB.")

    print("\nThe results will be saved in the folder named 'results/'. "
          "You can sort that alphabetically and take the greatest one. "
          "As you run the optimization, results are consinuously saved into a "
          "'results.pkl' file, too. Re-running optimize.py will resume "
          "the meta-optimization.\n")

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
        #plot_best_model()





