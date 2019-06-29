import numpy as np
from numpy import array
import pandas as pd
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras import backend as K
import tensorflow as tf
from datetime import datetime
import os, pickle

residue_list = list('ACEDGFIHKMLNQPSRTWVYX') + ['NoSeq']
q8_list      = list('LBEGIHST') + ['NoSeq']

# The custom accuracy metric used for this task
def accuracy(y_true, y_predicted):
    y = tf.argmax(y_true, axis =- 1)
    y_ = tf.argmax(y_predicted, axis =- 1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())

# Convert probabilities to secondary structure
def to_seq(y):
    seqs=[]
    for i in range(len(y)):
        seq_i=''
        for j in range(len(y[i])):
            seq_i += q8_list[np.argmax(y[i][j])]
        seqs.append(seq_i)
    return seqs

# Decode: map to a sequence from a one-hot
# encoding, takes a one-hot encoded y matrix
# with an lookup table "index"
# Maps the sequence to a one-hot encoding
def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        i = np.argmax(o)
        if i != 0:
            s += index[i]
        else:
            break
    return s

counter = 0
# prints the results
def print_results(x, y_, revsere_decoder_index, counter,test_df, write_df=False, print_pred=False):
    if write_df:
        Ans['id'][counter] = test_df['id'][counter]
        Ans['expected'][counter] = str(onehot_to_seq(y_, revsere_decoder_index).upper())
    if print_pred:
        print("prediction: " + str(onehot_to_seq(y_, revsere_decoder_index).upper()))

# Computes and returns the n-grams of a particular sequence, defaults to trigrams
def seq2ngrams(seqs, n = 3):
    return np.array([[seq[i : i + n] for i in range(len(seq))] for seq in seqs])


def load_augmented_data(npy_path, max_len):
    data = np.load(npy_path)

    data_reshape = data.reshape(data.shape[0], 700, -1)
    residue_onehot = data_reshape[:,:,0:22]
    residue_q8_onehot = data_reshape[:,:,22:31]
    profile = data_reshape[:,:,35:57]
    #pad profiles to same length
    zero_arr = np.zeros((profile.shape[0], max_len - profile.shape[1], profile.shape[2]))
    profile_padded = np.concatenate([profile, zero_arr], axis=1)

    residue_array = np.array(residue_list)[residue_onehot.argmax(2)]
    q8_array = np.array(q8_list)[residue_q8_onehot.argmax(2)]
    residue_str_list = []
    q8_str_list = []
    for vec in residue_array:
        x = ''.join(vec[vec != 'NoSeq'])
        residue_str_list.append(x)
    for vec in q8_array:
        x = ''.join(vec[vec != 'NoSeq'])
        q8_str_list.append(x)

    id_list = np.arange(1, len(residue_array) + 1)
    len_list = np.array([len(x) for x in residue_str_list])
    train_df = pd.DataFrame({'id': id_list, 'len': len_list, 'input': residue_str_list, 'expected': q8_str_list})
    return train_df, profile_padded


def get_acc(gt, pred):
    correct = 0
    for i in range(len(gt)):
        if gt[i] == pred[i]:
            correct += 1
    return (1.0 * correct) / len(gt)


def evaluate_acc(y_predicted):
    print('Analyse accuracy')

    order_list = [8, 5, 2, 0, 7, 6, 3, 1, 4]
    labels = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T', 'NoSeq']

    m1p = np.zeros_like(y_predicted)
    for count, i in enumerate(order_list):
        m1p[:, :, i] = y_predicted[:, :y_predicted.shape[1], count]

    summed_probs = m1p

    length_list = [len(line.strip().split(',')[2]) for line in open('cb513test_solution.csv').readlines()]
    print('max protein seq length is', np.max(length_list))

    ensemble_predictions = []
    for protein_idx, i in enumerate(length_list):
        new_pred = ''
        for j in range(i):
            new_pred += labels[np.argmax(summed_probs[protein_idx, j, :])]
        ensemble_predictions.append(new_pred)

    # calculating accuracy: compare to cb513test_solution
    gt_all = [line.strip().split(',')[3] for line in open('cb513test_solution.csv').readlines()]
    acc_list = []
    equal_counter = 0
    total = 0

    for gt, pred in zip(gt_all, ensemble_predictions):
        if len(gt) == len(pred):
            acc = get_acc(gt, pred)
            acc_list.append(acc)
            equal_counter += 1
        else:
            acc = get_acc(gt, pred)
            acc_list.append(acc)
        total += 1
    print('the accuracy is', np.mean(acc_list))
    print(str(equal_counter) + ' from ' + str(total) + ' proteins are of equal length')
    return acc_list

def weighted_accuracy(y_true, y_pred):
    return K.sum(K.equal(K.argmax(y_true, axis=-1),
                  K.argmax(y_pred, axis=-1)) * K.sum(y_true, axis=-1)) / K.sum(y_true)