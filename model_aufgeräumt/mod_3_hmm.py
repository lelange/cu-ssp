import sys

sys.path.append('keras-tcn')
from tcn import tcn
import h5py
from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import dill as pickle
import pandas as pd
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, CuDNNGRU
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Activation, RepeatVector, Permute
import tensorflow as tf
from keras.layers.merge import concatenate
# from google.colab import files
from keras.layers import Dropout
from keras import regularizers
from keras.layers import merge
from keras.optimizers import Adam
from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.layers.recurrent import Recurrent
from keras.engine import InputSpec
from keras.engine.topology import Layer
import os
from keras.callbacks import EarlyStopping ,ModelCheckpoint

from utils import *

print("##device name:")
print(tf.test.gpu_device_name())
print("##gpu available:")
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))



maxlen_seq = 700
minlen_seq= 100
normalize = True
standardize = False
pssm = True
hmm = True

if not pssm and not hmm:
    raise Exception('you should use one of the profiles!')

#inputs: primary structure
train_input_seqs = np.load('../data/train_input.npy')
test_input_seqs = np.load('../data/test_input.npy')
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
        train_pssm = normal(train_pssm)
        test_pssm= normal(test_pssm)

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


####


#transform sequence to n-grams, default n=3
train_input_grams = seq2ngrams(train_input_seqs)
test_input_grams = seq2ngrams(test_input_seqs)

# Use tokenizer to encode and decode the sequences
tokenizer_encoder = Tokenizer(char_level = True)
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

# labels to one-hot
y_test = to_categorical(test_target_data)
y_train = to_categorical(train_target_data)

# Computing the number of words and number of tags to be passed as parameters to the keras model
n_words = len(tokenizer_encoder.word_index) + 1
n_tags = len(tokenizer_decoder.word_index) + 1

print("number words or endoder word index: ", n_words)
print("number tags or decoder word index: ", n_tags)

#### validation data

n_samples = train_input_seqs.shape[0]
np.random.seed(0)
validation_idx = np.random.choice(np.arange(n_samples), size=300, replace=False)
training_idx = np.array(list(set(np.arange(n_samples))-set(validation_idx)))

X_val = X_train[validation_idx]
X_train = X_train[training_idx]

y_val = y_train[validation_idx]
y_train = y_train[training_idx]

X_aug_val = X_aug_train[validation_idx]
X_aug_train = X_aug_train[training_idx]

print("X train shape: ", X_train.shape)
print("y train shape: ", y_train.shape)
print("X aug train shape: ", X_aug_train.shape)
#### end validation

def build_model():
    input = Input(shape=(None,))
    profiles_input = Input(shape=(None, X_aug_train.shape[2]))

    # Defining an embedding layer mapping from the words (n_words) to a vector of len 128
    x1 = Embedding(input_dim=n_words, output_dim=250, input_length=None)(input)
    x1 = concatenate([x1, profiles_input], axis=2)

    x2 = Embedding(input_dim=n_words, output_dim=125, input_length=None)(input)
    x2 = concatenate([x2, profiles_input], axis=2)

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

model = build_model()

load_file = "./model/mod_3-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+".h5"

earlyStopping = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, mode='max')
checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose = 1, save_best_only=True, mode='max')
# Training the model on the training data and validating using the validation set
history=model.fit([X_train, X_aug_train], y_train, validation_data=([X_val, X_aug_val], y_val),
        epochs=10, batch_size=16, callbacks=[checkpointer], verbose=1, shuffle=True)

model.load_weights(load_file)
print("####evaluate:")
score = model.evaluate([X_test,X_aug_test], y_test, verbose=2, batch_size=1)
print(score)
print ('test loss:', score[0])
print ('test accuracy:', score[2])

