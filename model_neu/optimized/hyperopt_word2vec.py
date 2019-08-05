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

import argparse
import sys
import os
import traceback
import time
import dill as pickle

from utils import *

from keras.utils import plot_model
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, space_eval, STATUS_FAIL

def parse_arguments():
    """
    :return: command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help='which model should be optimized', type=int, required=True)
    return parser.parse_args()

args = parse_arguments()
model = args.model

if model == 1:
    from mod_1_w2v import build_and_train, build_model, MODEL_NAME

    SAVE_RESULTS = "results_mod1_w2v.pkl" # save trials of optimization
    SAVE_BEST_PLOT = "model_1_w2v_best" # save best NN graph

if model == 3:
    from mod_3_w2v import build_and_train, build_model, MODEL_NAME

    SAVE_RESULTS = "results_mod3_w2v.pkl" # save trials of optimization
    SAVE_BEST_PLOT = "model_3_w2v_best" # save best NN graph

space = {
    'embed_dim': hp.quniform('embed_dim', 20, 300, 10),
    'window_size': hp.quniform('window_size', 3, 100, 1),
    'negative': hp.quniform('negative', 1, 20, 1),
    'iter': hp.quniform('iter', 3, 30, 1),
    'n_gram': hp.quniform('n_gram', 1, 2, 1)
}


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
    #plot(space_best_model, SAVE_BEST_PLOT)

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

        print("BEST MODEL:")
        plot_best_model()

