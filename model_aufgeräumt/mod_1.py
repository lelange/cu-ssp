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
from keras.models import Model, Input
from keras.layers import Embedding, Dense, TimeDistributed, Concatenate, BatchNormalization
from keras.layers import Bidirectional, Activation, Dropout, CuDNNGRU, Conv1D, GRU

from sklearn.model_selection import train_test_split, KFold
from keras.metrics import categorical_accuracy
from keras import backend as K
from keras.regularizers import l1, l2
import tensorflow as tf
import keras

from utils import *

### Data Retrieval
cb513filename = '../data/cb513.npy'
cb6133filteredfilename = '../data/cb6133filtered.npy'

maxlen_seq = r = 700  # protein residues padded to 700
f = 57  # number of features for each residue

#load train
train_df, X_aug_train = load_augmented_data(cb6133filteredfilename  ,maxlen_seq)
train_input_seqs, train_target_seqs = train_df[['input', 'expected']][(train_df.len <= maxlen_seq)].values.T
#load test
test_df, X_aug_test = load_augmented_data(cb513filename,maxlen_seq)
test_input_seqs, test_target_seqs = test_df[['input','expected']][(test_df.len <= maxlen_seq)].values.T

#tokenizer
# Converting the inputs to trigrams
train_input_grams = seq2ngrams(train_input_seqs)
# Initializing and defining the tokenizer encoders and decoders based on the train set
tokenizer_encoder = Tokenizer()
tokenizer_encoder.fit_on_texts(train_input_grams)
tokenizer_decoder = Tokenizer(char_level = True)
tokenizer_decoder.fit_on_texts(train_target_seqs)


# Using the tokenizer to encode and decode the sequences for use in training

#train inputs
train_input_data = tokenizer_encoder.texts_to_sequences(train_input_grams)
X_train = sequence.pad_sequences(train_input_data, maxlen = maxlen_seq, padding = 'post')
#train targets
train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)
train_target_data = sequence.pad_sequences(train_target_data, maxlen = maxlen_seq, padding = 'post')
y_train = to_categorical(train_target_data)
#test inputs
test_input_grams = seq2ngrams(test_input_seqs)
test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
X_test = sequence.pad_sequences(test_input_data, maxlen = maxlen_seq, padding = 'post')
#test targets
test_target_data = tokenizer_decoder.texts_to_sequences(test_target_seqs)
test_target_data = sequence.pad_sequences(test_target_data, maxlen = maxlen_seq, padding = 'post')
y_test = to_categorical(test_target_data)

# Computing the number of words and number of tags
n_words = len(tokenizer_encoder.word_index) + 1
n_tags = len(tokenizer_decoder.word_index) + 1


#used name for profiles
train_profiles_np = X_aug_train
test_profiles_np = X_aug_test
"""
def decode_results(y_, reverse_decoder_index):
    # print("prediction: " + str(onehot_to_seq(y_, reverse_decoder_index).upper()))
    return str(onehot_to_seq(y_, reverse_decoder_index).upper())
"""

def run_test(_model, data1, data2, npy_name, csv_name = None):
    # Get predictions using our model
    y_test_pred = _model.predict([data1, data2])

    if csv_name is not None:
        """
        reverse_decoder_index = {value: key for key, value in tokenizer_decoder.word_index.items()}
        reverse_encoder_index = {value: key for key, value in tokenizer_encoder.word_index.items()}

        decoded_y_pred = []
        for i in range(len(test_input_data)):
            res = decode_results(y_test_pred[i], reverse_decoder_index)
            decoded_y_pred.append(res)

        # Set Columns
        out_df = pd.DataFrame()
        out_df["id"] = test_df.id.values
        out_df["expected"] = decoded_y_pred

        # Save results
        with open(csv_name, "w") as f:
            out_df.to_csv(f, index=False)
        """
    #save prediction
    np.save(npy_name, y_test_pred)


""" Run below for a single run """


def train(X_train, y_train, X_val=None, y_val=None):
    """
    Main Training function with the following properties:
        Optimizer - Nadam
        Loss function - Categorical Crossentropy
        Batch Size - 128 (any more will exceed Collab GPU RAM)
        Epochs - 50
    """
    model = CNN_BIGRU()
    model.compile(
        optimizer="Nadam",
        loss="categorical_crossentropy",
        metrics=["accuracy", accuracy])

    if X_val is not None and y_val is not None:
        history = model.fit(X_train, y_train,
                            batch_size=128, epochs=75,
                            validation_data=(X_val, y_val))
    else:
        history = model.fit(X_train, y_train,
                            batch_size=128, epochs=75)

    return history, model


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

    input = Input(shape=(maxlen_seq,))
    embed_out = Embedding(input_dim=n_words, output_dim=128, input_length=maxlen_seq)(input)
    profile_input = Input(shape=(maxlen_seq, 22))
    x = Concatenate([embed_out, profile_input])  # 5600, 700, 150
    
    x = super_conv_block(x)
    x = conv_block(x)
    x = super_conv_block(x)
    x = conv_block(x)
    x = super_conv_block(x)
    x = conv_block(x)

    x = Bidirectional(CuDNNGRU(units=256, return_sequences=True, recurrent_regularizer=l2(0.2)))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(256, activation="relu"))(x)
    x = TimeDistributed(Dropout(0.5))(x)

    y = TimeDistributed(Dense(n_tags, activation="softmax"))(x)

    model = Model([inp, inp_profiles], y)

    return model


X_train = [train_input_data, train_profiles_np]
y_train = train_target_data

history, model = train(X_train, y_train)

# Save the model as a JSON format
"""
model.save_weights("cb513_weights_1.h5")
with open("model_1.json", "w") as json_file:
    json_file.write(model.to_json())

# Save training history for parsing
with open("history_1.pkl", "wb") as hist_file:
    pickle.dump(history.history, hist_file)
    
"""

# Predict on test dataset and save the output
run_test(model,
         test_input_data[:],
         test_profiles_np[:],
         "cb513_test_prob_1.npy")
""" End single run """
