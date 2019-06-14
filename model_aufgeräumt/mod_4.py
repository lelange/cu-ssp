import numpy as np
from numpy import array
import pandas as pd
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import *
from keras.layers import *
from keras.regularizers import l2
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.metrics import categorical_accuracy
from keras import backend as K
from sklearn.model_selection import KFold
import tensorflow as tf
from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime
import os, pickle
from keras.callbacks import EarlyStopping

from utils import *

'''
various helper functions
'''

# Fixed-size Ordinally Forgetting Encoding
def encode_FOFE(onehot, alpha, maxlen):
    enc = np.zeros((maxlen, 2 * 22))
    enc[0, :22] = onehot[0]
    enc[maxlen-1, 22:] = onehot[maxlen-1]
    for i in range(1, maxlen):
        enc[i, :22] = enc[i-1, :22] * alpha + onehot[i]
        enc[maxlen-i-1, 22:] = enc[maxlen-i, 22:] * alpha + onehot[maxlen-i-1]
    return enc


'''
Getting Data
'''
#cb513filename= '../../../../py_charm_code/data/cb513.npy'
#cb6133filteredfilename = '../../../../py_charm_code/data/cb6133filtered.npy'
cb513filename = '../data/cb513.npy'
cb6133filteredfilename = '../data/cb6133filtered.npy'

'''
Setting up training, validation, test data
'''

maxlen_seq = 700 # maximum sequence length

#load train and test
train_df, X_aug_train = load_augmented_data(cb6133filteredfilename  ,maxlen_seq)
train_input_seqs, train_target_seqs = train_df[['input', 'expected']][(train_df.len <= maxlen_seq)].values.T
test_df, X_aug_test = load_augmented_data(cb513filename,maxlen_seq)
test_input_seqs, test_target_seqs = test_df[['input','expected']][(test_df.len <= maxlen_seq)].values.T

# Using the tokenizer to encode and decode the sequences for use in training
#tokenizer
train_input_grams = seq2ngrams(train_input_seqs)
tokenizer_encoder = Tokenizer()
tokenizer_encoder.fit_on_texts(train_input_grams)
tokenizer_decoder = Tokenizer(char_level = True)
tokenizer_decoder.fit_on_texts(train_target_seqs)

#train
train_input_data = tokenizer_encoder.texts_to_sequences(train_input_grams)
X_train = sequence.pad_sequences(train_input_data, maxlen = maxlen_seq, padding = 'post')
train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)
train_target_data = sequence.pad_sequences(train_target_data, maxlen = maxlen_seq, padding = 'post')
y_train = to_categorical(train_target_data)

#test
test_input_grams = seq2ngrams(test_input_seqs)
test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
X_test = sequence.pad_sequences(test_input_data, maxlen = maxlen_seq, padding = 'post')
test_target_data = tokenizer_decoder.texts_to_sequences(test_target_seqs)
test_target_data = sequence.pad_sequences(test_target_data, maxlen = maxlen_seq, padding = 'post')
y_test = to_categorical(test_target_data)

# Computing the number of words and number of tags to be passed as parameters to the keras model
n_words = len(tokenizer_encoder.word_index) + 1
n_tags = len(tokenizer_decoder.word_index) + 1

#### tensorboard
script_name = "mod_4"
model_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + script_name
log_dir = '../logs/{}'.format(model_name)
os.mkdir(log_dir)

'''
Model

'''

input = Input(shape = (maxlen_seq,))
embed_out = Embedding(input_dim = n_words, output_dim = 128, input_length = maxlen_seq)(input)
profile_input = Input(shape = (maxlen_seq,22))
x = concatenate([embed_out, profile_input]) # 5600, 700, 150

# one dense layer to remove sparsity
x = Dense(128, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
x = Reshape([maxlen_seq, 128, 1])(x)

# Defining 3 convolutional layers with different kernel sizes
# kernel size = 3
conv1 = ZeroPadding2D((3//2, 0), data_format='channels_last')(x)
conv1 = Conv2D(filters=64,
               kernel_size=(3, 128),
               input_shape=(1, maxlen_seq, 128),
               data_format='channels_last',
               strides=(1, 1),
               dilation_rate=(1, 1),
               activation='relu',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros')(conv1)
conv1 = BatchNormalization(axis=-1)(conv1)

# kernel size = 7
conv2 = ZeroPadding2D((7//2, 0), data_format='channels_last')(x)
conv2 = Conv2D(filters=64,
               kernel_size=(7, 128),
               input_shape=(1, maxlen_seq, 128),
               data_format='channels_last',
               strides=(1, 1),
               padding='valid',
               dilation_rate=(1, 1),
               activation='relu',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros')(conv2)
conv2 = BatchNormalization(axis=-1)(conv2)

# kernel size = 11
conv3 = ZeroPadding2D((11//2, 0), data_format='channels_last')(x)
conv3 = Conv2D(filters=64,
               kernel_size=(11, 128),
               input_shape=(1, maxlen_seq, 128),
               data_format='channels_last',
               strides=(1, 1),
               padding='valid',
               dilation_rate=(1, 1),
               activation='relu',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros')(conv3)
conv3 = BatchNormalization(axis=-1)(conv3)
conv = concatenate([conv1, conv2, conv3])
conv = Reshape([maxlen_seq, 3*64])(conv)

# Defining 3 bidirectional GRU layers; taking the concatenation of outputs
gru1 = Bidirectional(GRU(32,
                         return_sequences='True',
                         activation='tanh',
                         recurrent_activation='hard_sigmoid',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         recurrent_initializer='orthogonal',
                         bias_initializer='zeros',
                         dropout=0.0,
                         recurrent_dropout=0.1,
                         implementation=1))(conv)

gru2 = Bidirectional(GRU(32,
                         return_sequences='True',
                         activation='tanh',
                         recurrent_activation='hard_sigmoid',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         recurrent_initializer='orthogonal',
                         bias_initializer='zeros',
                         dropout=0.0,
                         recurrent_dropout=0.1,
                         implementation=1))(gru1)

gru3 = Bidirectional(GRU(32,
                         return_sequences='True',
                         activation='tanh',
                         recurrent_activation='hard_sigmoid',
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         recurrent_initializer='orthogonal',
                         bias_initializer='zeros',
                         dropout=0.0,
                         recurrent_dropout=0.1,
                         implementation=1))(gru2)

comb = concatenate([gru1, gru2, gru3, conv])

# Defining two fully-connected layers with dropout
x = TimeDistributed(Dense(256,
                          activation='relu',
                          use_bias=True,
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros'))(comb)
x = Dropout(0.1)(x)

x = TimeDistributed(Dense(128,
                          activation='relu',
                          use_bias=True,
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros'))(x)
x = Dropout(0.1)(x)

# Defining the output layer
y = TimeDistributed(Dense(n_tags,
                          activation='softmax',
                          use_bias=False,
                          kernel_initializer='glorot_uniform'))(x)

# Defining the model as a whole and printing the summary
model = Model([input, profile_input], y)
model.summary()

'''
Fitting and Predicting

'''
model.compile(optimizer = "nadam", loss = "categorical_crossentropy", metrics = ["accuracy", accuracy])
tensorboard = TensorBoard(log_dir=log_dir)

#checkpoint = ModelCheckpoint(os.path.join(log_dir, "best_val_acc.h5"),
#monitor='val_accuracy',
#verbose=1,
#save_best_only=True,
#mode='max')

model.fit([X_train, X_aug_train], y_train, batch_size = 64, epochs = 12, verbose = 1, callbacks=[tensorboard])

##uncomment later!!!!

#np.save('cb513_test_prob_4.npy', y_pre)

########evaluate accuracy#######
print(model.metrics_names)
acc = model.evaluate([X_test,X_aug_test], y_test)
print("evaluate via model.evaluate:")
print (acc)

y_pre = model.predict([X_test,X_aug_test])
evaluate_acc(y_pre)

