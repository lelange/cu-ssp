from keras import backend as K
import numpy as np
#from keras.layers.core import K
import traceback
from datetime import datetime
import traceback
import dill as pickle

from hyperutils import *

from keras.utils import plot_model
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, space_eval, STATUS_FAIL

from model_super import build_and_train, build_model, MODEL_NAME

SAVE_RESULTS = "results_mod_super_v2.pkl" # save trials of optimization
SAVE_BEST_PLOT = "model_super_best_v2" # save best NN graph

DROPOUT_CHOICES = np.arange(0.0, 0.9, 0.1)
UNIT_CHOICES = [100, 200, 500, 800, 1000, 1200]
GRU_CHOICES = [100, 200, 300, 400, 500, 600]
BATCH_CHOICES = [16, 32]
LR_CHOICES = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01]

space = {
    # This loguniform scale will multiply the learning rate, so as to make
    # it vary exponentially, in a multiplicative fashion rather than in
    # a linear fashion, to handle his exponentialy varying nature:
    'lr_rate_mult': hp.loguniform('lr_rate_mult', -0.5, 0.5),
    # max weight contraint:
    # L2 weight decay:
    'l2_weight_reg_mult': hp.loguniform('l2_weight_reg_mult', -1.3, 1.3),
    # batch size
    'batch_size': hp.quniform('batch_size', 20, 450, 2),
    # Choice of optimizer:
    'optimizer': hp.choice('optimizer', ['Adam', 'Nadam', 'RMSprop']),
    # LSTM units:
    'dense_output': hp.quniform('dense_output', 100, 500, 50),
    # Uniform distribution in finding appropriate dropout values, conv layers
    'dropout': hp.uniform('dropout', 0.0, 0.9),
    #decide input
    'input': hp.choice('input', ['onehot','seqs','both'] ),
    'use_profiles': hp.choice('use_profiles', [hp.choice('which_profiles', ['pssm', 'hmm', 'both']), None]),
    #position tcn layer
    'tcn_position':hp.choice('tcn_position', [hp.choice('position', ['first', 'last']), None]),
    #loss function to be evaluated
    'loss':hp.choice('loss',['categorical_crossentropy', 'nll', 'mean_squared_error']),
    # decide about first layer: recurrent or conv
    'first_layer':hp.choice('first_layer', [
        { #use LSTM
            'type': 'LSTM',
            'lstm_units': hp.quniform('lstm_units', 50, 800, 10 ),
            'lstm_nb': hp.choice('lstm_nb', [1, 2, 3])
        },
        { #or use GRU
            'type': 'GRU',
            'gru1': hp.loguniform('gru1', -0.6, 0.6),
            # nesting the layers ensures they're only un-rolled sequentially
            'gru2': hp.choice('gru2', [False, {
                'gru2_units': hp.loguniform('gru2_units', -0.6, 0.6),
                # only make the 3rd layer availabile if the 2nd one is
                'gru3': hp.choice('gru3', [False, {
                    'gru3_units': hp.loguniform('gru3_units', -0.6, 0.6)
                }]),
            }]),

        },
        { # or use convolutional Layer
            'type': 'conv',
            'conv_filter_size': hp.quniform('conv_filter_size', 8, 128, 8),
            'nb_filter': hp.quniform('nb_filter', 8, 128, 8),
            'nb_conv_layers': hp.choice('nb_conv_layers', [1, 2, 3]),
        },
    ]),
    #same for the second layer
    'second_layer':hp.choice('second_layer', [
        { #use LSTM
            'type': 'LSTM',
            'lstm_units_2': hp.quniform('lstm_units_2', 50, 800, 10 ),
            'lstm_nb_2': hp.choice('lstm_nb_2', [1, 2, 3])
        },
        { #or use GRU
            'type': 'GRU',
            'gru1_2': hp.loguniform('gru1_2', -0.6, 0.6),
            # nesting the layers ensures they're only un-rolled sequentially
            'gru2_2': hp.choice('gru2_2', [False, {
                'gru2_units_2': hp.loguniform('gru2_units_2', -0.6, 0.6),
                # only make the 3rd layer availabile if the 2nd one is
                'gru3_2': hp.choice('gru3_2', [False, {
                    'gru3_units_2': hp.loguniform('gru3_units_2', -0.6, 0.6)
                }]),
            }]),

        },
        { # or use convolutional Layer
            'type': 'conv',
            'conv_filter_size_2': hp.quniform('conv_filter_size_2', 8, 128, 8),
            'nb_filter_2': hp.quniform('nb_filter_2', 2, 32, 2),
            'nb_conv_layers_2': hp.choice('nb_conv_layers_2', [1, 2, 3]),
        },
        #no second layer
        False

    ])


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
    plot(space_best_model, SAVE_BEST_PLOT)


def optimize_model(hyperspace):
    """Build model 1 and train it."""

    try:
        model, model_name, result = build_and_train(hyperspace)

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
        max_evals = 21 #test mode = 1
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

