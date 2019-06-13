import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
import tensorflow as tf
from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime
import os, pickle

from utils import *

maxlen_seq = 768

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

# Dropout to prevent overfitting.
droprate = 0.3

def conv_block(x, n_channels, droprate):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(n_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Dropout(droprate)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(n_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    return x

def up_block(x, n_channels):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling1D(size = 2)(x)
    x = Conv1D(n_channels, 2, padding = 'same', kernel_initializer = 'he_normal')(x)
    return x

input = Input(shape = (maxlen_seq, ))
augment_input = Input(shape = (maxlen_seq, 22))

# Defining an embedding layer mapping from the words (n_words) to a vector of len 128
embed_input = Embedding(input_dim = n_words, output_dim = 128, input_length = None)(input)

merged_input = concatenate([embed_input, augment_input])

merged_input = Conv1D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(merged_input)

conv1 = conv_block(merged_input, 128, droprate)
pool1 = MaxPooling1D(pool_size=2)(conv1)

conv2 = conv_block(pool1, 192, droprate)
pool2 = MaxPooling1D(pool_size=2)(conv2)

conv3 = conv_block(pool2, 384, droprate)
pool3 = MaxPooling1D(pool_size=2)(conv3)

conv4 = conv_block(pool3, 768, droprate)
pool4 = MaxPooling1D(pool_size=2)(conv4)

conv5 = conv_block(pool4, 1536, droprate)

up4 = up_block(conv5, 768)
up4 = concatenate([conv4,up4], axis = 2)
up4 = conv_block(up4, 768, droprate)

up3 = up_block(up4, 384)
up3 = concatenate([conv3,up3], axis = 2)
up3 = conv_block(up3, 384, droprate)

up2 = up_block(up3, 192)
up2 = concatenate([conv2,up2], axis = 2)
up2 = conv_block(up2, 192, droprate)

up1 = up_block(up2, 128)
up1 = concatenate([conv1,up1], axis = 2)
up1 = conv_block(up1, 128, droprate)

up1 = BatchNormalization()(up1)
up1 = ReLU()(up1)

# the following it equivalent to Conv1D with kernel size 1
# A dense layer to output from the LSTM's64 units to the appropriate number of tags to be fed into the decoder
y = TimeDistributed(Dense(n_tags, activation = "softmax"))(up1)


# Defining the model as a whole and printing the summary
model = Model([input, augment_input], y)
model.summary()

optim = RMSprop(lr=0.002)

def scheduler(i, lr):
    if i in [60]:
        return lr * 0.5
    return lr

reduce_lr = LearningRateScheduler(schedule=scheduler, verbose=1)
# reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
#                             patience=8, min_lr=0.0005, verbose=1)

# Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
model.compile(optimizer = optim, loss = "categorical_crossentropy", metrics = ["accuracy", accuracy])


# Training the model on the training data and validating using the validation set
model.fit([X_train, X_train_augment], y_train, batch_size = 128, verbose = 1,
            epochs = 80)

#prediction for cb513

acc = model.evaluate([X_test,X_aug_test], y_test)
print("evaluate via model.evaluate:")
print (acc)

y_pred = model.predict([X_test, test_augment_data])
evaluate_acc(y_pre)

print(model.metrics_names)



