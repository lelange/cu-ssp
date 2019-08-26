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

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from keras import backend as K
from keras import regularizers, constraints, initializers, activations

from keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, CuDNNGRU, Concatenate
from keras.layers import Dropout, Flatten, Activation, RepeatVector, Permute, Conv1D, BatchNormalization

from keras.layers import Dropout
from keras.layers import merge
from keras.layers.merge import concatenate
from keras.layers.recurrent import Recurrent
from keras.metrics import categorical_accuracy
from keras.models import Model, Input, Sequential
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from scipy.stats import logistic
from utils import *

start_time = time.time()
'''
print("##device name:")
print(tf.test.gpu_device_name())
print("##gpu available:")
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

'''

def parse_arguments():
    """
    :return: command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-pssm', help='use pssm profiles', action='store_true')
    parser.add_argument('-hmm', help='use hmm profiles', action='store_true')
    parser.add_argument('-normalize', help='nomalize profiles', action='store_true')
    parser.add_argument('-standardize',  help='standardize profiles', action='store_true')
    parser.add_argument('-cv', help='use crossvalidation' , action= 'store_true')
    return parser.parse_args()

args = parse_arguments()

maxlen_seq = 700
minlen_seq= 100
normalize = args.normalize
standardize = args.standardize
pssm = args.pssm
hmm = args.hmm

if not pssm and not hmm:
    raise Exception('you should use one of the profiles!')

#inputs: primary structure
train_input_seqs = np.load('../data/data_qzlshy/train_input.npy')
test_input_seqs = np.load('../data/data_qzlshy/test_input.npy')
#labels: secondary structure
train_target_seqs = np.load('../data/data_qzlshy/train_q8.npy')
test_target_seqs = np.load('../data/data_qzlshy/test_q8.npy')

'''
X_train = np.load('../data/data_qzlshy/X_train_6133.npy')
X_test = np.load('../data/data_qzlshy/X_test_513.npy')
y_train = np.load('../data/data_qzlshy/y_train_6133.npy')
y_test = np.load('../data/data_qzlshy/y_test_513.npy')

'''

#profiles
if normalize:
    # load normalized profiles
    print("load normalized profiles... ")
    if pssm == True:
        train_pssm = np.load('../data/data_qzlshy/train_pssm.npy')
        test_pssm = np.load('../data/data_qzlshy/test_pssm.npy')
        #train_pssm = normal(train_pssm)
        #test_pssm= normal(test_pssm)

    if hmm == True:
        train_hmm = np.load('../data/data_qzlshy/train_hmm.npy')
        test_hmm = np.load('../data/data_qzlshy/test_hmm.npy')
        train_hmm = normal(train_hmm)
        test_hmm= normal(test_hmm)

elif standardize:
    print("load standardized profiles... ")
    if pssm == True:
        train_pssm = np.load('../data/data_qzlshy/train_pssm.npy')
        test_pssm = np.load('../data/data_qzlshy/test_pssm.npy')
        train_pssm = logistic.cdf(train_pssm)
        test_pssm = logistic.cdf(test_pssm)

    if hmm == True:
        train_hmm = np.load('../data/data_qzlshy/train_hmm.npy')
        test_hmm = np.load('../data/data_qzlshy/test_hmm.npy')
        train_hmm = logistic.cdf(train_hmm)
        test_hmm = logistic.cdf(test_hmm)

else:
    print("load profiles...")
    if pssm == True:
        train_pssm = np.load('../data/data_qzlshy/train_pssm.npy')
        test_pssm = np.load('../data/data_qzlshy/test_pssm.npy')

    if hmm == True:
        train_hmm = np.load('../data/data_qzlshy/train_hmm.npy')
        test_hmm = np.load('../data/data_qzlshy/test_hmm.npy')


if pssm and hmm:
    train_profiles = np.concatenate((train_pssm, train_hmm), axis=2)
    test_profiles = np.concatenate((test_pssm, test_hmm), axis=2)
elif pssm:
    train_profiles = train_pssm
    test_profiles = test_pssm
else:
    train_profiles = train_hmm
    test_profiles = test_hmm

X_aug_train=train_profiles
X_aug_test=test_profiles


#transform sequence to n-grams, default n=1
train_input_grams = seq2ngrams(train_input_seqs)
test_input_grams = seq2ngrams(test_input_seqs)

# Use tokenizer to encode and decode the sequences
tokenizer_encoder = Tokenizer()
tokenizer_encoder.fit_on_texts(train_input_grams)
tokenizer_decoder = Tokenizer(char_level = True) #char_level=True means that every character is treated as a token
tokenizer_decoder.fit_on_texts(train_target_seqs)

#train
train_input_data = tokenizer_encoder.texts_to_sequences(train_input_grams)
train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)

#test
test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
test_target_data = tokenizer_decoder.texts_to_sequences(test_target_seqs)

# pad sequences to maxlen_seq
X_train = sequence.pad_sequences(train_input_data, maxlen = maxlen_seq, padding = 'post')
X_test = sequence.pad_sequences(test_input_data, maxlen = maxlen_seq, padding = 'post')
train_target_data = sequence.pad_sequences(train_target_data, maxlen = maxlen_seq, padding = 'post')
test_target_data = sequence.pad_sequences(test_target_data, maxlen = maxlen_seq, padding = 'post')

# Computing the number of words and number of tags to be passed as parameters to the keras model
n_words = len(tokenizer_encoder.word_index) + 1
n_tags = len(tokenizer_decoder.word_index) + 1

print("number words or endoder word index: ", n_words)
print("number tags or decoder word index: ", n_tags)

# labels to one-hot
y_test = to_categorical(test_target_data)
y_train = to_categorical(train_target_data)

time_data = time.time() - start_time


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

def build_model():
    model = None
    input = Input(shape=(None,))
    profiles_input = Input(shape=(None, X_aug_train.shape[2]))

    # Defining an embedding layer mapping from the words (n_words) to a vector of len 250
    x1 = Embedding(input_dim=n_words, output_dim=250, input_length=None)(input)
    x1 = concatenate([x1, profiles_input])

    x1 = Dense(1200, activation="relu")(x1)
    x1 = Dropout(0.5)(x1)
    # Defining a bidirectional GRU using the embedded representation of the inputs
    x1 = Bidirectional(CuDNNGRU(units=500, return_sequences=True))(x1)
    x1 = Bidirectional(CuDNNGRU(units=100, return_sequences=True))(x1)

    x2 = Embedding(input_dim=n_words, output_dim=125, input_length=None)(input)
    x2 = concatenate([x2, profiles_input])
    # Defining a bidirectional GRU using the embedded representation of the inputs
    x2 = Bidirectional(CuDNNGRU(units=500, return_sequences=True))(x2)
    x2 = Bidirectional(CuDNNGRU(units=100, return_sequences=True))(x2)
    COMBO_MOVE = concatenate([x1, x2])

    w = Dense(500, activation="relu")(COMBO_MOVE)  # try 500
    w = Dropout(0.4)(w)
    w = super_conv_block(w)
    w = conv_block(w)

    # Defining an embedding layer mapping from the words (n_words) to a vector of len 250
    x3 = Embedding(input_dim=n_words, output_dim=250, input_length=None)(input)
    print(profiles_input.shape)
    x3 = concatenate([x3, profiles_input])

    x3 = Dense(600, activation="relu")(x3)
    x3 = Dropout(0.5)(x3)
    # Defining a bidirectional GRU using the embedded representation of the inputs
    x3 = Bidirectional(CuDNNGRU(units=300, return_sequences=True))(x3)
    x3 = Bidirectional(CuDNNGRU(units=150, return_sequences=True))(x3)

    x4 = Embedding(input_dim=n_words, output_dim=125, input_length=None)(input)
    x4 = concatenate([x4, profiles_input])
    # Defining a bidirectional GRU using the embedded representation of the inputs
    x4 = Bidirectional(CuDNNGRU(units=300, return_sequences=True))(x4)
    x4 = Bidirectional(CuDNNGRU(units=150, return_sequences=True))(x4)

    COMBO_MOVE2 = concatenate([x3, x4])
    w2 = Dense(300, activation="relu")(COMBO_MOVE2)  # try 500
    w2 = Dropout(0.4)(w2)
    w2 = super_conv_block(w2)
    w2 = conv_block(w2)

    COMBO_MOVE3 = concatenate([w, w2])
    w3 = Dense(150, activation="relu")(COMBO_MOVE3)  # try 500
    w3 = Dropout(0.4)(w3)
    w3 = tcn.TCN(return_sequences=True)(w3)

    y = TimeDistributed(Dense(n_tags, activation="softmax"))(w3)

    # Defining the model as a whole and printing the summary
    model = Model([input, profiles_input], y)
    # model.summary()

    # Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
    adamOptimizer = Adam(lr=0.001, beta_1=0.8, beta_2=0.8, epsilon=None, decay=0.0001, amsgrad=False)
    model.compile(optimizer=adamOptimizer, loss="categorical_crossentropy", metrics=["accuracy", accuracy])
    return model

def train_model(X_train_aug, y_train, X_val_aug, y_val, X_test_aug, y_test, epochs = 5):

    model = build_model()

    load_file = "./model/mod_3-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+".h5"

    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                                  patience=2, verbose=1, min_lr=0.0001)
    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, mode='max')
    checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose = 1, save_best_only=True, mode='max')
    #tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    # Training the model on the training data and validating using the validation set
    history = model.fit(X_train_aug, y_train, validation_data=(X_val_aug, y_val),
            epochs=epochs, batch_size=16, callbacks=[checkpointer, earlyStopping, reduce_lr], verbose=1, shuffle=True)

    model.load_weights(load_file)
    print("####evaluate:")
    score = model.evaluate(X_test_aug, y_test, verbose=2, batch_size=1)
    print(score)
    print ('test loss:', score[0])
    print ('test accuracy:', score[2])
    return model, score[2]



def crossValidation(X_train, X_aug_train, y_train, X_test, X_aug_test, y_test, n_folds=10):
    # Instantiate the cross validator
    kfold_splits = n_folds
    kf = KFold(n_splits=kfold_splits, shuffle=True)

    cv_scores = []
    model_history = []

    # Loop through the indices the split() method returns
    for index, (train_indices, val_indices) in enumerate(kf.split(X_train, y_train)):
        print("Training on fold " + str(index + 1) + "/"+kfold_splits+"...")
        # Generate batches from indices
        X_train_fold, X_val_fold = X_train[train_indices], X_train[val_indices]
        X_aug_train_fold, X_aug_val_fold = X_aug_train[train_indices], X_aug_train[val_indices]
        y_train_fold, y_val_fold = y_train[train_indices], y_train[val_indices]

        print("Training new iteration on " + str(X_train_fold.shape[0]) + " training samples, " + str(
            X_val_fold.shape[0]) + " validation samples...")

        X_train_aug_fold = [X_train_fold, X_aug_train_fold]
        X_val_aug_fold = [X_val_fold, X_aug_val_fold]

        X_test_aug = [X_test, X_aug_test]

        model, test_acc = train_model(X_train_aug_fold, y_train_fold,
                                  X_val_aug_fold, y_val_fold,
                                  X_test_aug, y_test)

        print('>%.3f' % test_acc)
        cv_scores.append(test_acc)
        model_history.append(model)
    return cv_scores, model_history



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
    model, test_acc = train_model(X_train_aug, y_train, X_val_aug, y_val, X_test_aug, y_test, epochs=30)


time_end = time.time() - start_time
m, s = divmod(time_end, 60)
print("The program needed {:.0f}s to load the data and {:.0f}min {:.0f}s in total.".format(time_data, m, s))


