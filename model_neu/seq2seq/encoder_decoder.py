from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Lambda
from keras.layers import Bidirectional, Activation, Dropout, CuDNNGRU, Conv1D, GRU, CuDNNLSTM
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import Activation, BatchNormalization, dot, concatenate
from datetime import datetime

def get_acc(gt, pred, mask=None):
    '''
         if len(gt)!=len(pred):
        print("Lengths are not equal. Len true = "+str(len(gt))+" len pred = "+str(len(pred)))
    '''
    correct = 0
    for i in range(len(gt)):
        if mask is not None:
            if mask[i] == 1:
                if gt[i] == pred[i]:
                    correct += 1
            length=np.sum(mask)

        else:
            try:
                if gt[i] == pred[i]:
                    correct += 1
            except:
                print(i)
            length = len(gt)
    return (1.0 * correct), length

def get_acc2(gt, pred, mask = None):
    '''
         if len(gt)!=len(pred):
        print("Lengths are not equal. Len true = "+str(len(gt))+" len pred = "+str(len(pred)))
    '''
    correct = 0
    for i in range(len(gt)):
        if mask is not None:
            if mask[i] == 1:
                if gt[i] == pred[i]:
                    correct += 1
            length=np.sum(mask)

        else:
            if gt[i] == pred[i]:
                correct += 1

            length = len(gt)

    return (1.0 * correct)/length

def accuracy(y_true, y_predicted):
    y = tf.argmax(y_true, axis =- 1)
    y_ = tf.argmax(y_predicted, axis =- 1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())


batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
MODEL_NAME = 's2q_lstm'
# Path to the data txt file on disk.
data_root = '/nosave/lange/cu-ssp/data/'
weights_file = MODEL_NAME+"-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+".h5"
load_file = "./model/"+weights_file

def seq2ngrams(seqs, n):
    """
    'AGAMQSASM' => [['AGA', 'MQS', 'ASM'], ['GAM','QSA'], ['AMQ', 'SAS']]
    """
    result = []
    for seq in seqs:
        a, b, c = zip(*[iter(seq)] * n), zip(*[iter(seq[1:])] * n), zip(*[iter(seq[2:])] * n)
        str_ngrams = []
        for ngrams in zip(a, b, c):
            for ngram in ngrams:
                str_ngrams.append("".join(ngram))
        result.append(str_ngrams)
    return result

def get_princeton_data(filename, max_len=700):
    start= 5
    end = 20
    n = 2

    ### filename = cb6133 for train, cb513 for test"
    path = data_root+'data_princeton/'

    primary_list=list('ACEDGFIHKMLNQPSRTWVYX') + ['NoSeq']
    q8_list = list('LBEGIHST') + ['NoSeq']

    data = np.load(path+filename+".npy")
    data_reshape = data.reshape(data.shape[0], max_len, -1)
    residue_onehot = data_reshape[:, :, 0:22]
    residue_q8_onehot = data_reshape[:, :, 22:31]
    profile = data_reshape[:, :, 35:57]

    # pad profiles to same length
    zero_arr = np.zeros((profile.shape[0], max_len - profile.shape[1], profile.shape[2]))
    profile_padded = np.concatenate([profile, zero_arr], axis=1)

    residue_array = np.array(primary_list)[residue_onehot.argmax(2)]
    q8_array = np.array(q8_list)[residue_q8_onehot.argmax(2)]
    residue_str_list = []
    q8_str_list = []
    for vec in residue_array:
        x = ''.join(vec[vec != 'NoSeq'])
        x = seq2ngrams(x[start:start+end], n)
        x = ['\t']+x
        residue_str_list.append(x)
    for vec in q8_array:
        x = ''.join(vec[vec != 'NoSeq'])
        x = seq2ngrams(x[start:start+end], n)
        x = ['\t'] + x + ['\n']
        q8_str_list.append(x)

    return residue_str_list, q8_str_list #, residue_onehot, residue_q8_onehot, profile_padded

# Vectorize the data.
'''
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

with open(data_path, 'r') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)
'''

def get_vocab(seqs):
    characters = []
    for seq in seqs:
        for char in seq:
            characters.append(char)
    return np.unique(characters)

input_texts, target_texts = get_princeton_data('cb6133filtered')

input_characters = get_vocab(input_texts)
target_characters = get_vocab(target_texts)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))

x1_out = Bidirectional(CuDNNLSTM(units=75, return_sequences=True), merge_mode='concat')(encoder_inputs)
x2_out = CuDNNLSTM(units=150, return_sequences=True)(x1_out)
attention = dot([x2_out, x1_out], axes=[2, 2])
attention = Activation('softmax')(attention)
context = dot([attention, x1_out], axes=[2, 1])
x2_out_combined_context = concatenate([context, x2_out])

encoder_outputs, state_h, state_c = CuDNNLSTM(latent_dim, return_state=True)(x2_out_combined_context)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = (CuDNNLSTM(latent_dim, return_sequences=True, return_state=True))
decoder_lstm2 = (CuDNNLSTM(latent_dim, return_sequences=True, return_state=True))

x, h, c = decoder_lstm2(decoder_inputs, initial_state=encoder_states)
decoder_outputs, _, _ = decoder_lstm(x)

decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# callbacks

earlyStopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max')
checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose=1, save_best_only=True,
                                   mode='max')
# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics = [accuracy])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          #callbacks=[checkpointer, earlyStopping],
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

# Save model
#model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

x, h, c = decoder_lstm2(decoder_inputs, initial_state=decoder_states_inputs)
decoder_outputs, state_h, state_c = decoder_lstm(
    x)

decoder_states = [state_h, state_c]

decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(30):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence: ', input_texts[seq_index])
    print('Decoded sentence: \t', decoded_sentence)
    print('Real sentence: ', target_texts[seq_index])

    #corr8, len8 = get_acc(target_texts[seq_index], decoded_sentence)
    #q8_accs = get_acc2(target_texts[seq_index], decoded_sentence)
    #print(1.0*corr8/len8)
    #print(q8_accs)