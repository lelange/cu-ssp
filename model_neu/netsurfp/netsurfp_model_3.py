import sys
import os
import time
import dill as pickle
import pprint

import numpy as np
import pandas as pd
import tensorflow as tf
sys.path.append('keras-tcn')
from tcn import tcn
import h5py

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from keras import backend as K
from keras import regularizers, constraints, initializers, activations

from keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, CuDNNGRU, Reshape
from keras.layers import Dropout, Flatten, Activation, RepeatVector, Permute

from keras.layers import Dropout
from keras.layers import merge
from keras.layers.core import Reshape
from keras.layers.merge import concatenate
from keras.layers.recurrent import Recurrent
from keras.metrics import categorical_accuracy
from keras.models import Model, Input, Sequential
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from utils import *

from collections import defaultdict
from datetime import datetime

from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.mongoexp import MongoTrials

#import objective

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_NAME = 'mod_3'
save_pred_file = "_pred_3.npy"

N_FOLDS = 10 # for cross validation
MAXLEN_SEQ = 700 # only use sequences to this length and pad to this length, choose from 600, 608, 700
NB_CLASSES_Q8 = 9 # number Q8 classes, used in final layer for classification (one extra for empty slots)
NB_CLASSES_Q3 = 3 # number Q3 classes
NB_AS = 20 # number of amino acids, length of one-hot endoded amino acids vectors
NB_FEATURES = 30 # feature dimension


start_time = time.time()

args = parse_arguments(default_epochs=25)

normalize = args.normalize
standardize = args.standardize
hmm = args.hmm
embedding = args.embedding
epochs = args.epochs
plot = args.plot
no_input = args.no_input
optimize = args.optimize
cross_validate = args.cv
tv_perc = args.tv_perc
test_mode = args.test_mode
predict_only = args.predict


if test_mode:
    N_FOLDS = 2
    epochs = 2

batch_size = 128

data_root = '../data/netsurfp/'
weights_file = MODEL_NAME+"-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+".h5"
load_file = "./model/"+weights_file
file_scores = "logs/cv_results.txt"
file_scores_mean = "logs/cv_results_mean.txt"

file_train = 'train_' + str(MAXLEN_SEQ)
file_test = ['cb513_'+ str(MAXLEN_SEQ), 'ts115_'+ str(MAXLEN_SEQ), 'casp12_'+ str(MAXLEN_SEQ)]


def build_model():
    model = None

    input = Input(shape=(MAXLEN_SEQ, NB_AS,))
    inp = input
    x1 = input
    x2 = input

    if hmm:
        profiles_input = Input(shape=(MAXLEN_SEQ, NB_FEATURES,))
        inp = [input, profiles_input]
        x1 = concatenate([input, profiles_input])
        x2 = concatenate([input, profiles_input])

    x1 = Dense(200, activation="relu")(x1)
    print(x1._keras_shape)
    x1 = Dropout(0.5)(x1)

    #x1 = Bidirectional(CuDNNGRU(units=100, return_sequences=True))(x1)
    # Defining a bidirectional LSTM using the embedded representation of the inputs
    x2 = Bidirectional(CuDNNGRU(units=100, return_sequences=True))(x2)
    print(x2._keras_shape)
    #x2 = Dropout(0.5)(x2)
    x2 = Bidirectional(CuDNNGRU(units=50, return_sequences=True))(x2)
    print(x2._keras_shape)
    #x2 = Dropout(0.5)(x2)
    #COMBO_MOVE = concatenate([x1, x2])
    #print(COMBO_MOVE._keras_shape)
    #w = Dense(24, activation="relu")(COMBO_MOVE)  #try 500
    w = Dense(24, activation="relu")(x2)
    print(w._keras_shape)
    w = Dropout(0.2)(w)
    #w = tcn.TCN(return_sequences=True)(w)
    print(w._keras_shape)
    #w = TimeDistributed(Dense(64, activation="relu"))(w)
    #print(w._keras_shape)
    #w2 = tcn.TCN(return_sequences=True)(x3)

    #w2 = TimeDistributed(Dense(180, activation="relu"))(w2)

    y = TimeDistributed(Dense(NB_CLASSES_Q8, activation="softmax"))(w)

    # Defining the model as a whole and printing the summary
    model = Model(inp, y)
    #model.summary()

    # Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
    #adamOptimizer = Adam(lr=0.001, beta_1=0.8, beta_2=0.8, epsilon=None, decay=0.0001, amsgrad=False)
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy", accuracy, tf_pearson, tf_accuracy])
    return model


DROPOUT_CHOICES = np.arange(0.0, 0.9, 0.1)
UNIT_CHOICES = [100, 200, 500, 800, 1000, 1200]
GRU_CHOICES = [100, 200, 300, 400, 500, 600]
BATCH_CHOICES = [16, 32]
LR_CHOICES = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01]
space = {
    'dense1': hp.quniform('dense1', UNIT_CHOICES),
    'dropout1': hp.uniform('dropout1', 0.0, 0.9),
    'gru1': hp.quniform('gru1', 100, 700, 100),
    # nesting the layers ensures they're only un-rolled sequentially
    'gru2': hp.choice('gru2', [False, {
        'gru2_units': hp.choice('gru2_units', GRU_CHOICES),
        # only make the 3rd layer availabile if the 2nd one is
        'gru3': hp.choice('gru3', [False, {
            'gru3_units': hp.choice('gru3_units', GRU_CHOICES)
        }]),
    }]),
    'dense2': hp.uniform('dense2', 0.0, 0.9),
    'dropout2': hp.choice('dropout2', DROPOUT_CHOICES),
    'lr': hp.choice('lr', LR_CHOICES),
    'decay': hp.choice('decay', LR_CHOICES),
    'batch_size': hp.choice('batch_size', BATCH_CHOICES)
}


def data():
    data_root = '/nosave/lange/cu-ssp/data/netsurfp/'
    file_train = 'train'
    file_test = ['cb513', 'ts115', 'casp12']

    X_test = np.load(data_root + file_test[0] + '_input.npy')
    profiles = np.load(data_root + file_test[0] + '_hmm.npy')
    mean = np.mean(profiles)
    std = np.std(profiles)
    X_aug_test = (profiles - mean) / std
    X_test_aug = [X_test, X_aug_test]
    y_test = np.load(data_root + file_test[0] + '_q8.npy')

    X_train = np.load(data_root + file_train + '_input.npy')
    profiles = np.load(data_root + file_train + '_hmm.npy')
    mean = np.mean(profiles)
    std = np.std(profiles)
    X_aug_train = (profiles - mean) / std
    X_train_aug = [X_train, X_aug_train]
    y_train = np.load(data_root + file_train + '_q8.npy')

    X_train_aug, y_train, X_val_aug, y_val = train_val_split(True, X_train_aug, y_train)

    return X_train_aug, y_train, X_val_aug, y_val, X_test_aug, y_test


def build_model_ho_3(params):
    print(params)
    input = Input(shape=(X_train_aug[0].shape[1], X_train_aug[0].shape[2],))
    profiles_input = Input(shape=(X_train_aug[1].shape[1], X_train_aug[1].shape[2],))
    x1 = concatenate([input, profiles_input])
    x2 = concatenate([input, profiles_input])
    x1 = Dense(params['dense1'], activation="relu")(x1)
    x1 = Dropout(params['dropout1'])(x1)
    x2 = Bidirectional(CuDNNGRU(units=params['gru1'], return_sequences=True))(x2)
    if params['gru2']:
        x2 = Bidirectional(CuDNNGRU(units=params['gru2']['gru2_units'], return_sequences=True))(x2)
    if params['gru2'] and params['gru2']['gru3']:
        x2 = Bidirectional(CuDNNGRU(units=params['gru2']['gru3']['gru3_units'], return_sequences=True))(x2)
    COMBO_MOVE = concatenate([x1, x2])
    w = Dense(params['dense2'], activation="relu")(COMBO_MOVE)
    w = Dropout(params['dropout2'])(w)
    w = tcn.TCN(return_sequences=True)(w)
    y = TimeDistributed(Dense(8, activation="softmax"))(w)
    model = Model([input, profiles_input], y)

    adamOptimizer = Adam(lr=params['lr'], beta_1=0.8, beta_2=0.8, epsilon=None, decay=params['decay'], amsgrad=False)
    model.compile(optimizer=adamOptimizer, loss="categorical_crossentropy", metrics=["accuracy", accuracy])

    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, mode='max')
    checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose=1, save_best_only=True,
                                   mode='max')

    model.fit(X_train_aug, y_train, validation_data=(X_val_aug, y_val),
              epochs=20, batch_size=params['batch_size'], callbacks=[checkpointer, earlyStopping],
              verbose=1, shuffle=True)

    model.load_weights(load_file)
    score = model.evaluate(X_test_aug, y_test)
    K.clear_session()
    result = {'loss': -score[2], 'status': STATUS_OK}

    return result

def build_and_train(X_train_aug, y_train, X_val_aug, y_val, epochs = epochs):
    model = build_model()

    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, mode='max')
    checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose = 1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, mode='max', cooldown = 2)

    #tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False,
                             # write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                             # embeddings_metadata=None, embeddings_data=None, update_freq='batch')
    # Training the model on the training data and validating using the validation set
    K.get_session().run(tf.local_variables_initializer())

    history = model.fit(X_train_aug, y_train, validation_data=(X_val_aug, y_val),
            epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, reduce_lr], verbose=1, shuffle=True)

    # plot accuracy during training
    if plot:
        plt.title('Accuracy')
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='val')
        plt.legend()
        plt.savefig('./plots/mod_3-CB513-' + datetime.now().strftime("%m_%d-%H_%M") + '_accuracy.png')

    return model, history

def evaluate_model(model, load_file, test_ind = None):
    if test_ind is None:
        test_ind = range(len(file_test))
    test_accs = []
    names = []
    for i in test_ind:
        X_test_aug, y_test = get_data(file_test[i], hmm, normalize, standardize, embedding)
        model.load_weights(load_file)
        print("####evaluate " + file_test[i] +":")
        score = model.evaluate(X_test_aug, y_test, verbose=2, batch_size=1)
        print(file_test[i] +' test loss:', score[0])
        print(file_test[i] +' test accuracy:', score[2])
        test_accs.append(score[2])
        names.append(file_test[i])
    return dict(zip(names, test_accs))

def crossValidation(load_file, X_train_aug, y_train, n_folds=N_FOLDS):
    X_train, X_aug_train = X_train_aug
    # Instantiate the cross validator
    kfold_splits = n_folds
    kf = KFold(n_splits=kfold_splits, shuffle=True)

    cv_scores = defaultdict(list)
    model_history = []

    # Loop through the indices the split() method returns
    for index, (train_indices, val_indices) in enumerate(kf.split(X_train, y_train)):
        print('\n\n-----------------------')
        print("Training on fold " + str(index + 1) + "/" + str(kfold_splits) +"...")
        print('-----------------------\n')

        # Generate batches from indices
        X_train_fold, X_val_fold = X_train[train_indices], X_train[val_indices]
        X_aug_train_fold, X_aug_val_fold = X_aug_train[train_indices], X_aug_train[val_indices]
        y_train_fold, y_val_fold = y_train[train_indices], y_train[val_indices]

        print("Training new iteration on " + str(X_train_fold.shape[0]) + " training samples, " + str(
            X_val_fold.shape[0]) + " validation samples...")

        model, history = build_and_train([X_train_fold, X_aug_train_fold], y_train_fold,
                                  [X_val_fold, X_aug_val_fold], y_val_fold)

        print(history.history)

        test_acc = evaluate_model(model, load_file, test_ind = [0, 1, 2])

        cv_scores['val_accuracy'].append(max(history.history['val_accuracy']))

        for k, v in test_acc.items():
            cv_scores[k].append(v)

        model_history.append(model)

    return cv_scores, model_history

# write best weight models in file and look for model (eg. mod_1) name in weight name
best_weights = "model/mod_3-CB513-2019_07_31-23_37.h5"

#--------------------------------- main ---------------------------------

if predict_only:
    build_and_predict(build_model(), best_weights, save_pred_file, MODEL_NAME, file_test)
    test_acc = None
    time_data = time.time() - start_time
    save_results = False
else:
    # load data
    X_train_aug, y_train = get_data(file_train, hmm, normalize, standardize, embedding)

    if hmm:
        print("X train shape: ", X_train_aug[0].shape)
        print("X aug train shape: ", X_train_aug[1].shape)
    else:
        print("X train shape: ", X_train_aug.shape)
    print("y train shape: ", y_train.shape)

    time_data = time.time() - start_time
    save_results = True

    if cross_validate:

        cv_scores, model_history = crossValidation(load_file, X_train_aug, y_train)
        test_accs = save_cv(weights_file, cv_scores, file_scores, file_scores_mean, N_FOLDS)
        test_acc = test_accs[file_test[0] + '_mean']

    else:
        X_train_aug, y_train, X_val_aug, y_val = train_val_split(hmm, X_train_aug, y_train, tv_perc)

        model, history = build_and_train(X_train_aug, y_train, X_val_aug, y_val, epochs=epochs)
        test_acc = evaluate_model(model, load_file)

time_end = time.time() - start_time
m, s = divmod(time_end, 60)
print("The program needed {:.0f}s to load the data and {:.0f}min {:.0f}s in total.".format(time_data, m, s))

telegram_me(m, s, sys.argv[0], test_acc, hmm, standardize)

if save_results:
    save_results_to_file(time_end, MODEL_NAME, weights_file, test_acc, hmm, standardize, normalize, embedding=embedding)

'''
if cross_validate :
    cv_scores, model_history = crossValidation(load_file, X_train_aug, y_train)
    test_acc = np.mean(cv_scores)
    print('Estimated accuracy %.3f (%.3f)' % (test_acc, np.std(cv_scores)))
else:
    X_train_aug, y_train, X_val_aug, y_val = train_val_split(hmm, X_train_aug, y_train, tv_perc)

    if optimize:
        #---- create a Trials database to store experiment results
        trials = MongoTrials('mongo://localhost:27017/jobs/jobs', exp_key='exp2')
        #---- use that Trials database for fmin
        best = fmin(objective.build_model_ho_3, space, algo=tpe.suggest, trials=trials, max_evals=10)
        #---- save trials
        pickle.dump(trials, open("./trials/mod_3-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+"-hyperopt.p", "wb"))
        #trials = pickle.load(open("./trials/mod_3-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+"-hyperopt.p", "rb"))
        print("Found minimum:")
        print(best)

    else:
        model = train_model(X_train_aug, y_train, X_val_aug, y_val, epochs=epochs)
        test_acc = evaluate_model(model, load_file, [0])

time_end = time.time() - start_time
m, s = divmod(time_end, 60)
print("The program needed {:.0f}s to load the data and {:.0f}min {:.0f}s in total.".format(time_data, m, s))

telegram_me(m, s, sys.argv[0], test_acc, hmm, standardize, normalize, no_input)
'''

