############################################
#
# LSTMs with Luang attention
#
############################################

##### Load .npy data file and generate sequence csv and profile csv files  #####
import numpy as np
import pandas as pd
import numpy as np
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional
from keras.layers import Activation, BatchNormalization, dot, concatenate
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping ,ModelCheckpoint

from utils import *

max_len =700
maxlen_seq = 700

cb513filename = '../data/cb513.npy'
cb6133filteredfilename = '../data/cb6133filtered.npy'

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



############################### Model starts here ##############################

input = Input(shape = (maxlen_seq,))
embed_out = Embedding(input_dim = n_words, output_dim = 128, input_length = maxlen_seq)(input)
profile_input = Input(shape = (maxlen_seq, 22))
x = concatenate([embed_out, profile_input]) # 5600, 700, 150

x1_out = Bidirectional(LSTM(units = 75, return_sequences=True, recurrent_dropout=0.2), merge_mode='concat')(x)
x1_out_last = x1_out[:,-1,:]

x2_out = LSTM(units = 150, return_sequences = True, recurrent_dropout=0.2)(x1_out, initial_state=[x1_out_last, x1_out_last])
x2_out_last = x2_out[:,-1,:]

attention = dot([x2_out, x1_out], axes=[2, 2])
attention = Activation('softmax')(attention)
context = dot([attention, x1_out], axes=[2, 1])
x2_out_combined_context = concatenate([context, x2_out])

x3_out = LSTM(units = 150, return_sequences = True, recurrent_dropout=0.2)(x2_out, initial_state=[x2_out_last, x2_out_last])
x3_out_last = x3_out[:,-1,:]

attention_2 = dot([x3_out, x2_out], axes=[2, 2])
attention_2 = Activation('softmax')(attention_2)
context_2 = dot([attention_2, x2_out], axes=[2, 1])
x3_out_combined_context = concatenate([context_2, x3_out])

attention_2_1 = dot([x3_out, x1_out], axes=[2, 2])
attention_2_1 = Activation('softmax')(attention_2_1)
context_2_1 = dot([attention_2_1, x1_out], axes=[2, 1])
x3_1_out_combined_context = concatenate([context_2_1, x3_out])

x4_out = LSTM(units = 150, return_sequences = True, recurrent_dropout=0.2)(x3_out, initial_state=[x3_out_last, x3_out_last])
x4_out_last = x4_out[:,-1,:]

attention_3 = dot([x4_out, x3_out], axes=[2, 2])
attention_3 = Activation('softmax')(attention_3)
context_3 = dot([attention_3, x3_out], axes=[2, 1])
x4_out_combined_context = concatenate([context_3, x4_out])

attention_3_1 = dot([x4_out, x2_out], axes=[2, 2])
attention_3_1 = Activation('softmax')(attention_3_1)
context_3_1 = dot([attention_3_1, x2_out], axes=[2, 1])
x4_1_out_combined_context = concatenate([context_3_1, x4_out])

attention_3_2 = dot([x4_out, x1_out], axes=[2, 2])
attention_3_2 = Activation('softmax')(attention_3_2)
context_3_2 = dot([attention_3_2, x1_out], axes=[2, 1])
x4_2_out_combined_context = concatenate([context_3_2, x4_out])

x5_out = LSTM(units = 150, return_sequences = True, recurrent_dropout=0.2)(x4_out, initial_state=[x4_out_last, x4_out_last])
x5_out_last = x5_out[:,-1,:]

attention_4 = dot([x5_out, x4_out], axes=[2, 2])
attention_4 = Activation('softmax')(attention_4)
context_4 = dot([attention_4, x4_out], axes=[2, 1])
x5_out_combined_context = concatenate([context_4, x5_out])

attention_4_1 = dot([x5_out, x3_out], axes=[2, 2])
attention_4_1 = Activation('softmax')(attention_4_1)
context_4_1 = dot([attention_4_1, x3_out], axes=[2, 1])
x5_1_out_combined_context = concatenate([context_4_1, x5_out])

attention_4_2 = dot([x5_out, x2_out], axes=[2, 2])
attention_4_2 = Activation('softmax')(attention_4_2)
context_4_2 = dot([attention_4_2, x2_out], axes=[2, 1])
x5_2_out_combined_context = concatenate([context_4_2, x5_out])

attention_4_3 = dot([x5_out, x1_out], axes=[2, 2])
attention_4_3 = Activation('softmax')(attention_4_3)
context_4_3 = dot([attention_4_3, x1_out], axes=[2, 1])
x5_3_out_combined_context = concatenate([context_4_3, x5_out])

out = keras.layers.Add()([x2_out_combined_context, \
             x3_out_combined_context, x3_1_out_combined_context,\
             x4_out_combined_context, x4_1_out_combined_context, x4_2_out_combined_context, \
             x5_out_combined_context, x5_1_out_combined_context, x5_2_out_combined_context, x5_3_out_combined_context])

fc1_out = TimeDistributed(Dense(150, activation="relu"))(out) # equation (5) of the paper
output = TimeDistributed(Dense(n_tags, activation="softmax"))(fc1_out) # equation (6) of the paper

model = Model([input, profile_input], output)
model.summary()

################################################################################

# Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
rmsprop = keras.optimizers.RMSprop(lr=0.003, rho=0.9, epsilon=None, decay=0.0) # add decay=0.5 after 15 epochs
model.compile(optimizer = rmsprop, loss = "categorical_crossentropy", metrics = ["accuracy", accuracy])



# Training the model on the training data and validating using the validation set
model.fit([X_train, X_aug_train], y_train, batch_size = 64, epochs = 20, verbose = 1)

########evaluate accuracy#######
print(model.metrics_names)
acc = model.evaluate([X_test,X_aug_test], y_test)
print("evaluate via model.evaluate:")
print (acc)

y_pre = model.predict([X_test,X_aug_test])
evaluate_acc(y_pre)



