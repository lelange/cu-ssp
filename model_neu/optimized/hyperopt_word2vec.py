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

space = {
    'embed_dim': hp.quniform('embed_dim', 20, 300, 10),
    'window_size': hp.quniform('window_size', 3, 100, 1),
    'negative': hp.quniform('negative', 1, 30, 1),
    'iter': hp.quniform('iter', 3, 30, 1),
    'n_gram': hp.quniform('n_gram', 1, 5, 1),
    'model':hp.choice('model',(0,1))
}

if model == 1:
    from mod_1_w2v import build_and_train, build_model, MODEL_NAME

    SAVE_RESULTS = "results_mod1_w2v.pkl" # save trials of optimization
    SAVE_BEST_PLOT = "model_1_w2v_best" # save best NN graph

if model == 3:
    from mod_3_w2v import build_and_train, build_model, MODEL_NAME

    SAVE_RESULTS = "results_mod3_w2v.pkl" # save trials of optimization
    SAVE_BEST_PLOT = "model_3_w2v_best" # save best NN graph


if model == 6:
    from mod_6_w2v import build_and_train, build_model, MODEL_NAME

    SAVE_RESULTS = "results_mod6_w2v.pkl"  # save trials of optimization
    SAVE_BEST_PLOT = "model_6_w2v_best"  # save best NN graph
    '''
    space.update({

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
        # Uniform distribution in finding appropriate dropout values, conv layers
        'dropout': hp.uniform('dropout', 0.0, 0.7),
        # Uniform distribution in finding appropriate dropout values, conv layers
        'dropout2': hp.uniform('dropout2', 0.0, 0.7),
        'super_conv_filter_size': hp.quniform('super_conv_filter_size', 8, 128, 8),
        # LSTM units:
        'GRU_units_mult': hp.loguniform('GRU_units_mult', -0.6, 0.6),
        # Number of super_conv+conv layers stacked:
        'nb_conv_super_layers': hp.choice('nb_conv_super_layers', [2, 3, 4]),

    })
    '''

if model == 7:
    from mod_7_w2v import build_and_train, build_model, MODEL_NAME

    SAVE_RESULTS = "results_mod7_w2v.pkl"  # save trials of optimization
    SAVE_BEST_PLOT = "model_7_w2v_best"  # save best NN graph

    space.update({
        # This loguniform scale will multiply the learning rate, so as to make
        # it vary exponentially, in a multiplicative fashion rather than in
        # a linear fashion, to handle his exponentialy varying nature:
        'lr_rate_mult': hp.loguniform('lr_rate_mult', -0.5, 0.5),
        # L2 weight decay:
        'l2_weight_reg_mult': hp.loguniform('l2_weight_reg_mult', -1.3, 1.3),
        # Batch size fed for each gradient update
        'batch_size': hp.quniform('batch_size', 100, 450, 5),
        # Choice of optimizer:
        'optimizer': hp.choice('optimizer', ['Adam', 'Nadam', 'RMSprop']),
        # Coarse labels importance for weights updates:
        'coarse_labels_weight': hp.uniform('coarse_labels_weight', 0.1, 0.7),
        # Uniform distribution in finding appropriate dropout values, conv layers
        'conv_dropout_drop_proba': hp.uniform('conv_dropout_proba', 0.0, 0.35),
        # Uniform distribution in finding appropriate dropout values, FC layers
        'fc_dropout_drop_proba': hp.uniform('fc_dropout_proba', 0.0, 0.6),
        # Use batch normalisation at more places?
        'use_BN': hp.choice('use_BN', [False, True]),

        # Use a first convolution which is special?
        'first_conv': hp.choice(
            'first_conv', [None, hp.choice('first_conv_size', [3, 4])]
        ),
        # Use residual connections? If so, how many more to stack?
        'residual': hp.choice(
            'residual', [None, hp.quniform(
                'residual_units', 1 - 0.499, 4 + 0.499, 1)]
        ),
        # Let's multiply the "default" number of hidden units:
        'conv_hiddn_units_mult': hp.loguniform('conv_hiddn_units_mult', -0.6, 0.6),
        # Number of conv+pool layers stacked:
        'nb_conv_pool_layers': hp.choice('nb_conv_pool_layers', [2, 3]),
        # Starting conv+pool layer for residual connections:
        'conv_pool_res_start_idx': hp.quniform('conv_pool_res_start_idx', 0, 2, 1),
        # The type of pooling used at each subsampling step:
        'pooling_type': hp.choice('pooling_type', [
            'max',  # Max pooling
            'avg',  # Average pooling
            'all_conv',  # All-convolutionnal: https://arxiv.org/pdf/1412.6806.pdf
            'inception'  # Inspired from: https://arxiv.org/pdf/1602.07261.pdf
        ]),
        # The kernel_size for convolutions:
        'conv_kernel_size': hp.quniform('conv_kernel_size', 2, 4, 1),
        # The kernel_size for residual convolutions:
        'res_conv_kernel_size': hp.quniform('res_conv_kernel_size', 2, 4, 1),

        # Amount of fully-connected units after convolution feature map
        'fc_units_1_mult': hp.loguniform('fc_units_1_mult', -0.6, 0.6),
        # Use one more FC layer at output
        'one_more_fc': hp.choice(
            'one_more_fc', [None, hp.loguniform('fc_units_2_mult', -0.6, 0.6)]
        ),
        # Activations that are used everywhere
        'activation': hp.choice('activation', ['relu', 'elu'])
    })




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
        max_evals=20
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

