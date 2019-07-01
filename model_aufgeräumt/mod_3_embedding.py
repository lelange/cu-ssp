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
import emoji
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
    parser.add_argument('--pssm', help='use pssm profiles', action='store_true')
    parser.add_argument('--hmm', help='use hmm profiles', action='store_true')
    parser.add_argument('--normalize', help='nomalize profiles', action='store_true')
    parser.add_argument('--standardize',  help='standardize profiles', action='store_true')
    parser.add_argument('--cv', help='use crossvalidation' , action= 'store_true')
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
train_input_seqs = np.load('../data/train_input_embedding.npy')
test_input_seqs = np.load('../data/test_input_embedding.npy')
#labels: secondary structure
train_target_seqs = np.load('../data/train_q8.npy')
test_target_seqs = np.load('../data/test_q8.npy')

#profiles
if normalize:
    # load normalized profiles
    print("load normalized profiles... ")
    if pssm == True:
        train_pssm = np.load('../data/train_pssm.npy')
        test_pssm = np.load('../data/test_pssm.npy')
        #train_pssm = normal(train_pssm)
        #test_pssm= normal(test_pssm)

    if hmm == True:
        train_hmm = np.load('../data/train_hmm.npy')
        test_hmm = np.load('../data/test_hmm.npy')
        train_hmm = normal(train_hmm)
        test_hmm= normal(test_hmm)

elif standardize:
    print("load standardized profiles... ")
    if pssm == True:
        train_pssm = np.load('../data/train_pssm.npy')
        test_pssm = np.load('../data/test_pssm.npy')
        train_pssm = standard(train_pssm)
        test_pssm = standard(test_pssm)

    if hmm == True:
        train_hmm = np.load('../data/train_hmm.npy')
        test_hmm = np.load('../data/test_hmm.npy')
        train_hmm = standard(train_hmm)
        test_hmm = standard(test_hmm)

else:
    print("load profiles...")
    if pssm == True:
        train_pssm = np.load('../data/train_pssm.npy')
        test_pssm = np.load('../data/test_pssm.npy')

    if hmm == True:
        train_hmm = np.load('../data/train_hmm.npy')
        test_hmm = np.load('../data/test_hmm.npy')


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

#?# maxlen_seq = len(test_input_seqs[0])
X_train = train_input_seqs
X_test = test_input_seqs

tokenizer_decoder = Tokenizer(char_level = True) #char_level=True means that every character is treated as a token
tokenizer_decoder.fit_on_texts(train_target_seqs)
train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)
test_target_data = tokenizer_decoder.texts_to_sequences(test_target_seqs)

train_target_data = sequence.pad_sequences(train_target_data, maxlen = maxlen_seq, padding = 'post')
test_target_data = sequence.pad_sequences(test_target_data, maxlen = maxlen_seq, padding = 'post')

n_tags = len(tokenizer_decoder.word_index) + 1

# labels to one-hot
y_test = to_categorical(test_target_data)
y_train = to_categorical(train_target_data)

print("X train shape: ", X_train.shape)
print("y train shape: ", y_train.shape)
print("X aug train shape: ", X_aug_train.shape)

print(X_train[0])
print(type(X_train[0]))
time_data = time.time() - start_time

def build_model():
    model = None
    input = Input(shape=(X_train.shape[1],))
    profiles_input = Input(shape=(X_aug_train.shape[1], X_aug_train.shape[2]))
    #reshaped = Reshape((None,))(profiles_input)
    #reshaped = Flatten()(profiles_input)
    # Defining an embedding layer mapping from the words (n_words) to a vector of len 128
    x1 = Embedding(input_dim=24, output_dim=175, input_length=None)(input)
    x1 = Reshape((16, 700, 256))(x1)
    #x1 = Dense(250, activation="relu")(reshaped)

    print(x1.shape)
    print(input.shape)
    x1 = concatenate([x1, profiles_input])

    x2 = Embedding(input_dim=24, output_dim=125, input_length=None)(input)
    print("x2 shape: ", x2.shape)
    #x2 = Dense(125, activation= "relu")(reshaped)
    x2 = concatenate([x2, profiles_input])
    print("x2 shape: ", x2.shape)

    x1 = Dense(1200, activation="relu")(x1)
    x1 = Dropout(0.5)(x1)

    # Defining a bidirectional LSTM using the embedded representation of the inputs
    x2 = Bidirectional(CuDNNGRU(units=500, return_sequences=True))(x2)
    x2 = Bidirectional(CuDNNGRU(units=100, return_sequences=True))(x2)
    COMBO_MOVE = concatenate([x1, x2])
    w = Dense(500, activation="relu")(COMBO_MOVE)  # try 500
    w = Dropout(0.4)(w)
    w = tcn.TCN(return_sequences=True)(w)

    y = TimeDistributed(Dense(n_tags, activation="softmax"))(w)

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

    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, mode='max')
    checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose = 1, save_best_only=True, mode='max')
    #tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    # Training the model on the training data and validating using the validation set
    history = model.fit(X_train_aug, y_train, validation_data=(X_val_aug, y_val),
            epochs=epochs, batch_size=16, callbacks=[checkpointer, earlyStopping], verbose=1, shuffle=True)

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
    model, test_acc = train_model(X_train_aug, y_train, X_val_aug, y_val, X_test_aug, y_test, epochs=10)


time_end = time.time() - start_time
m, s = divmod(time_end, 60)
print("The program needed {:.0f}s to load the data and {:.0f}min {:.0f}s in total.".format(time_data, m, s))

def message_me(model_name, m, s):
    username = 'charlie.gpu'
    password = '19cee1Et742'
    recipient = '100002834091853'  #Anna: 100002834091853, Chris: 100001479799294
    client = fbchat.Client(username, password)
    msg = Message(text='{} ist erfolgreich durchgelaufen! \U0001F973 '
                       '\n\n(Gesamtlaufzeit {:.0f}min {:.0f}s)'.format(model_name, m, s))

    sent = client.send(msg, thread_id=recipient, thread_type=ThreadType.USER)
    client.logout()

message_me(sys.argv[0], m, s)