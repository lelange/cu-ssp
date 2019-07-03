import sys
import os
import argparse
import time
import numpy as np
import dill as pickle
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

from keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, CuDNNGRU
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
import fbchat
from fbchat.models import *
import telegram
from utils import *

start_time = time.time()

args = parse_arguments(default_epochs=6)

normalize = args.normalize
standardize = args.standardize
pssm = args.pssm
hmm = args.hmm
embedding = args.embedding
epochs = args.epochs
batch_size = 16

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

print("X train shape: ", X_train.shape)
print("y train shape: ", y_train.shape)
print("X aug train shape: ", X_aug_train.shape)

time_data = time.time() - start_time

def build_model():
    model = None
    input = Input(shape=(X_train.shape[1], X_train.shape[2],))
    profiles_input = Input(shape=(X_aug_train.shape[1], X_aug_train.shape[2],))
    #reshaped = Reshape((None,))(profiles_input)
    #reshaped = Flatten()(profiles_input)
    # Defining an embedding layer mapping from the words (n_words) to a vector of len 128
    #x1 = Embedding(input_dim=24, output_dim=350, input_length=None)(input)
    #x1 = Dense(250, activation="relu")(reshaped)
    x1 = concatenate([input, profiles_input])
    #experiment with dense layer
    #x2 = Dense(125, activation= "relu")(reshaped)
    x2 = concatenate([input, profiles_input])

    x1 = Dense(1200, activation="relu")(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Bidirectional(CuDNNGRU(units=100, return_sequences=True))(x1)
    #x1 = Dropout(0.5)(x1)
    # Defining a bidirectional LSTM using the embedded representation of the inputs
    x2 = Bidirectional(CuDNNGRU(units=500, return_sequences=True))(x2)
    #x2 = Dropout(0.5)(x2)
    x2 = Bidirectional(CuDNNGRU(units=100, return_sequences=True))(x2)
    #x2 = Dropout(0.5)(x2)
    COMBO_MOVE = concatenate([x1, x2])
    w = Dense(500, activation="relu")(COMBO_MOVE)  # try 500
    w = Dropout(0.4)(w)
    w = tcn.TCN(return_sequences=True)(w)

    y = TimeDistributed(Dense(n_tags, activation="softmax"))(w)

    # Defining the model as a whole and printing the summary
    model = Model([input, profiles_input], y)
    model.summary()

    # Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
    adamOptimizer = Adam(lr=0.001, beta_1=0.8, beta_2=0.8, epsilon=None, decay=0.0001, amsgrad=False)
    model.compile(optimizer=adamOptimizer, loss="categorical_crossentropy", metrics=["accuracy", accuracy])
    return model

def train_model(X_train_aug, y_train, X_val_aug, y_val, X_test_aug, y_test, epochs = epochs):

    model = build_model()
    load_file = "./model/mod_3-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+".h5"

    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, mode='max')
    checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose = 1, save_best_only=True, mode='max')
    #tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False,
                             # write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                             # embeddings_metadata=None, embeddings_data=None, update_freq='batch')
    # Training the model on the training data and validating using the validation set
    history = model.fit(X_train_aug, y_train, validation_data=(X_val_aug, y_val),
            epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, earlyStopping], verbose=1, shuffle=True)

    # plot accuracy during training
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.legend()
    plt.savefig('./plots/mod_3-CB513-'+datetime.now().strftime("%m_%d-%H_%M")+'_accuracy.png')

    model.load_weights(load_file)
    print("####evaluate:")
    score = model.evaluate(X_test_aug, y_test, verbose=2, batch_size=1)
    print(score)
    print ('test loss:', score[0])
    print ('test accuracy:', score[2])
    return model, score[2]

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