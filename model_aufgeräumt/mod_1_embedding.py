"""
Cascaded Convolution Model

- Pranav Shrestha (ps2958)
- Jeffrey Wan (jw3468)

"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import *
from keras.layers import *
from sklearn.model_selection import train_test_split, KFold
from keras.metrics import categorical_accuracy
from keras import backend as K
from keras.optimizers import Nadam
from keras.regularizers import l1, l2
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau

import sys
import time

from utils import *


start_time = time.time()

args = parse_arguments(default_epochs=75) #=50

normalize = args.normalize
standardize = args.standardize
pssm = args.pssm
hmm = args.hmm
embedding = args.embedding
epochs = args.epochs

n_tags = 9
n_words = 24

if not pssm and not hmm:
    raise Exception('you should use one of the profiles!')

#inputs: primary structure
if embedding:
    X_train = np.load('../data/train_input_embedding_residue.npy')
    X_test = np.load('../data/test_input_embedding_residue.npy')
else:
    X_train = np.load('../data/X_train_6133.npy')
    X_test = np.load('../data/X_test_513.npy')

#labels: secondary structure
y_train = np.load('../data/y_train_6133.npy')
y_test = np.load('../data/y_test_513.npy')

X_aug_train, X_aug_test = prepare_profiles(pssm, hmm, normalize, standardize)

time_data = time.time() - start_time


def train_model(X_train_aug, y_train,
                X_val_aug, y_val,
                X_test_aug, y_test,
                epochs = epochs):
    """
    Main Training function with the following properties:
        Optimizer - Nadam
        Loss function - Categorical Crossentropy
        Batch Size - 128 (any more will exceed Collab GPU RAM)
        Epochs - 50
    """
    model = CNN_BIGRU()
    optim = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(
        optimizer=optim,
        loss="categorical_crossentropy",
        metrics=["accuracy", accuracy])

    load_file = "./model/mod_1-CB513-" + datetime.now().strftime("%Y_%m_%d-%H_%M") + ".h5"
    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max')
    checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose=1, save_best_only=True,
                                   mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=8, verbose=1, mode='max')

    # Training the model on the training data and validating using the validation set
    history = model.fit(X_train_aug, y_train, validation_data=(X_val_aug, y_val),
            epochs=epochs, batch_size=64, callbacks=[checkpointer, earlyStopping, reduce_lr], verbose=1, shuffle=True)

    model.load_weights(load_file)
    print('\n----------------------')
    print('----------------------')
    print("evaluate:")
    score = model.evaluate(X_test_aug, y_test, verbose=2, batch_size=1)
    print(score)
    print('test loss:', score[0])
    print('test accuracy:', score[2])

    return model, score[2]


""" Build model """


def conv_block(x, activation=True, batch_norm=True, drop_out=True, res=True):
    cnn = Conv1D(64, 11, padding="same")(x)
    if activation: cnn = TimeDistributed(Activation("relu"))(cnn)
    if batch_norm: cnn = TimeDistributed(BatchNormalization())(cnn)
    if drop_out:   cnn = TimeDistributed(Dropout(0.5))(cnn)
    if res:        cnn = Concatenate(axis=-1)([x, cnn])

    return cnn


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


def CNN_BIGRU():
    model = None
    input = Input(shape=(X_train.shape[1], X_train.shape[2],))
    profile_input = Input(shape=(X_aug_train.shape[1], X_aug_train.shape[2],))
    x = concatenate([input, profile_input])
    '''
    x = super_conv_block(x)
    x = conv_block(x)
    x = TimeDistributed(Dropout(0.5))(x)
    '''

    x = super_conv_block(x)
    x = conv_block(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = super_conv_block(x)
    x = conv_block(x)
    x = TimeDistributed(Dropout(0.5))(x)

    x = Bidirectional(CuDNNGRU(units=256, return_sequences=True, recurrent_regularizer=l2(0.2)))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(256, activation="relu"))(x)
    x = TimeDistributed(Dropout(0.5))(x)

    y = TimeDistributed(Dense(n_tags, activation="softmax"))(x)

    model = Model([input, profile_input], y)
    model.summary()

    return model

if args.cv :
    cv_scores, model_history = crossValidation(X_train, X_aug_train, y_train, X_test, X_aug_test, y_test)
    print('Estimated accuracy %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))
else:
    n_samples = len(X_train)
    np.random.seed(0)
    validation_idx = np.random.choice(np.arange(n_samples), size=300, replace=False)
    training_idx = np.array(list(set(np.arange(n_samples)) - set(validation_idx)))

    X_val = X_train[validation_idx]
    X_train = X_train[training_idx]
    y_val = y_train[validation_idx]
    y_train = y_train[training_idx]
    X_aug_val = X_aug_train[validation_idx]
    X_aug_train = X_aug_train[training_idx]

    X_train_aug = [X_train, X_aug_train]
    X_val_aug = [X_val, X_aug_val]
    X_test_aug = [X_test, X_aug_test]
    model, test_acc = train_model(X_train_aug, y_train, X_val_aug, y_val, X_test_aug, y_test, epochs=epochs)

time_end = time.time() - start_time
m, s = divmod(time_end, 60)
print("The program needed {:.0f}s to load the data and {:.0f}min {:.0f}s in total.".format(time_data, m, s))

telegram_me(m, s, sys.argv[0])
