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

from utils import *

cb513filename = '../data/cb513.npy'
cb6133filteredfilename = '../data/cb6133filtered.npy'

maxlen_seq = 700 # protein residues padded to 700

#load train and test
train_df, X_aug_train = load_augmented_data(cb6133filteredfilename  ,maxlen_seq)
train_input_seqs, train_target_seqs = train_df[['input', 'expected']][(train_df.len <= maxlen_seq)].values.T
test_df, X_aug_test = load_augmented_data(cb513filename,maxlen_seq)
test_input_seqs, test_target_seqs = test_df[['input','expected']][(test_df.len <= maxlen_seq)].values.T

train_input_grams = seq2ngrams(train_input_seqs)
tokenizer_encoder = Tokenizer()

tokenizer_encoder.fit_on_texts(train_input_grams)
tokenizer_decoder = Tokenizer(char_level = True)
tokenizer_decoder.fit_on_texts(train_target_seqs)

train_input_data = tokenizer_encoder.texts_to_sequences(train_input_grams)
X_train = sequence.pad_sequences(train_input_data, maxlen = maxlen_seq, padding = 'post')
train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)
train_target_data = sequence.pad_sequences(train_target_data, maxlen = maxlen_seq, padding = 'post')
y_train = to_categorical(train_target_data)


test_input_grams = seq2ngrams(test_input_seqs)
test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
X_test = sequence.pad_sequences(test_input_data, maxlen = maxlen_seq, padding = 'post')
test_target_data = tokenizer_decoder.texts_to_sequences(test_target_seqs)
test_target_data = sequence.pad_sequences(test_target_data, maxlen = maxlen_seq, padding = 'post')
y_test = to_categorical(test_target_data)

###validation data##

n_samples = len(train_df)
print(n_samples)
print(train_df.shape)
np.random.seed(0)
validation_idx = np.random.choice(np.arange(n_samples), size=300, replace=False)
training_idx = np.array(list(set(np.arange(n_samples))-set(validation_idx)))

X_val = X_train[validation_idx]
print('shape X_train: ', X_train.shape)
X_train = X_train[training_idx]
y_val = y_train[validation_idx]
y_train = y_train[training_idx]

X_aug_train = X_aug_train[training_idx]
X_aug_val = X_aug_train[validation_idx]
print("shape X_aug_train: ", X_aug_train.shape)

###end validation###

n_words = len(tokenizer_encoder.word_index) + 1
n_tags = len(tokenizer_decoder.word_index) + 1


input = Input(shape = (maxlen_seq,))
input2 = Input(shape=(maxlen_seq,22))

x = Embedding(input_dim = n_words, output_dim = 128, input_length = maxlen_seq)(input)
x = concatenate([x,input2],axis=2)

z = Conv1D(64, 11, strides=1, padding='same')(x)
w = Conv1D(64, 7, strides=1, padding='same')(x)
x = concatenate([x,z],axis=2)
x = concatenate([x,w],axis=2)
z = Conv1D(64, 5, strides=1, padding='same')(x)
w = Conv1D(64, 3, strides=1, padding='same')(x)
x = concatenate([x,z],axis=2)
x = concatenate([x,w],axis=2)
x = Bidirectional(CuDNNLSTM(units = 128, return_sequences = True))(x)

y = TimeDistributed(Dense(n_tags, activation = "softmax"))(x)

model = Model([input,input2], y)
model.compile(optimizer = 'RMSprop', loss = "categorical_crossentropy", metrics = ["accuracy", accuracy, weighted_accuracy])
model.summary()

### monitor = 'val_weighted_accuracy'
earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

load_file = "./model/2c(3(21)42-200-0.003-400-600)-de-2LSTM-CB513.h5"

checkpointer = ModelCheckpoint(filepath=load_file, verbose=1, save_best_only=True)

history=model.fit([X_train,X_aug_train], y_train, validation_data=([X_val, X_aug_val], y_val),
        epochs=30, batch_size=128, callbacks=[checkpointer, earlyStopping], verbose=2, shuffle=True)


model.load_weights(load_file)

print("#########evaluate:##############")
score = model.evaluate([X_test,X_aug_test], y_test, verbose=2, batch_size=1)
print(score)
print ('test loss:', score[0])
print ('test accuracy:', score[1])

###
#model.fit([X_train,X_aug_train], y_train, batch_size = 128, epochs = 30, validation_data = ([X_test,X_aug_test], y_test), verbose = 1)

#acc = model.evaluate([X_test,X_aug_test], y_test)
#print("evaluate via model.evaluate:")
#print (acc)
y_pre = model.predict([X_test,X_aug_test])

evaluate_acc(y_pre)