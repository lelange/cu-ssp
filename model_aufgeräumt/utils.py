import numpy as np
from numpy import array
import pandas as pd
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from keras import backend as K
import tensorflow as tf
import argparse
import telegram
from datetime import datetime
import os, pickle

residue_list = list('ACEDGFIHKMLNQPSRTWVYX') + ['NoSeq']
q8_list      = list('LBEGIHST') + ['NoSeq']
data_root = '../data/netsurfp/'

def parse_arguments(default_epochs):
    """
    :return: command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_input', help='do not use input data', action='store_true')
    parser.add_argument('-optimize', help='perform hyperoptimization', action='store_true')
    parser.add_argument('-pssm', help='use pssm profiles', action='store_true')
    parser.add_argument('-hmm', help='use hmm profiles', action='store_true')
    parser.add_argument('-normalize', help='nomalize profiles', action='store_true')
    parser.add_argument('-standardize',  help='standardize profiles', action='store_true')
    parser.add_argument('-cv', help='use crossvalidation' , action= 'store_true')
    parser.add_argument('-embedding', help='embed input vectors via elmo embedder', action='store_true')
    parser.add_argument('-epochs',type=int ,required=False, help='number of training epochs', default=default_epochs)
    parser.add_argument('-plot', help='plot accuracy', action='store_true')
    return parser.parse_args()

def normal(data):
    min = np.min(data)
    max = np.max(data)
    data_ = (data-min)/(max-min)
    return data_

def standard(data):
    mean = np.mean(data)
    std = np.std(data)
    data_ = (data - mean) / std
    return data_

# for netsurf (hmm) data
def get_data(filename, hmm, normalize, standardize, embedding = False, no_input=False):

    print('Load ' + filename + ' data...')
    if embedding:
        input_seq = np.load(data_root + filename + '_netsurfp_input_embedding_residue.npy')
    else:
        if no_input:
            input_seq = np.load(data_root + filename + '_hmm.npy')
            if normalize:
                input_seq = normal(input_seq)
            if standardize:
                input_seq = standard(input_seq)

        else:
            input_seq =  np.load(data_root+filename+'_input.npy')
    q8 = np.load(data_root + filename + '_q8.npy')
    if hmm:
        profiles = np.load(data_root+filename+'_hmm.npy')
        if normalize:
            print('Normalize...')
            profiles = normal(profiles)
        if standardize:
            print('Standardize...')
            profiles = standard(profiles)
        input_aug = [input_seq, profiles]
    else:
        input_aug = input_seq
    return  input_aug, q8

# for pssm+hmm data
def prepare_profiles(pssm, hmm, normalize, standardize):
    # profiles
    if pssm == True:
        print("load pssm profiles... ")
        train_pssm = np.load('../data/train_pssm.npy')
        test_pssm = np.load('../data/test_pssm.npy')

        if normalize:
            print('normalize pssm profiles...')
            train_pssm = normal(train_pssm)
            test_pssm= normal(test_pssm)

        if standardize:
            train_pssm = standard(train_pssm)
            test_pssm = standard(test_pssm)

    if hmm == True:
        print("load hmm profiles... ")
        train_hmm = np.load('../data/train_hmm.npy')
        test_hmm = np.load('../data/test_hmm.npy')

        if normalize:
            train_hmm = normal(train_hmm)
            test_hmm = normal(test_hmm)

        if standardize:
            train_hmm = standard(train_hmm)
            test_hmm = standard(test_hmm)

    if pssm and hmm:
        train_profiles = np.concatenate((train_pssm, train_hmm), axis=2)
        test_profiles = np.concatenate((test_pssm, test_hmm), axis=2)
    elif pssm:
        train_profiles = train_pssm
        test_profiles = test_pssm
    else:
        train_profiles = train_hmm
        test_profiles = test_hmm

    return train_profiles, test_profiles

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
def seq2ngrams(seqs, n = 1):
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

def telegram_me(m, s, model_name, test_acc = None, hmm=False, standardize=False, normalize = False, no_input = False, embedding=False):
    Token = "806663548:AAEJIMIBEQ9eKdyF8_JYnxUhUsDQZls1w7w"
    chat_ID = "69661085"
    bot = telegram.Bot(token=Token)
    msg = '{} ist erfolgreich durchgelaufen! \U0001F60B \n\n' \
          '(Gesamtlaufzeit {:.0f}min {:.0f}s)'.format(model_name, m, s)
    if hmm:
        verb = ''
        if standardize:
            verb += 'standardisierte '
        if normalize:
            verb += 'und normalisierte '
        msg+='\nEs wurden '+verb+'HMM Profile verwendet.'
    if no_input:
        msg+='\n Es wurden nur HMM Profile als Input verwendet.'
    if embedding:
        msg+='\n Die Input-Daten wurden ins 1024-dim. eingebettet.'
    if test_acc is not None:
        msg += '\nTest accuracy: {:.3%}'.format(test_acc)
    bot.send_message(chat_id=chat_ID, text=msg)

def message_me(model_name, m, s):
    username = 'charlie.gpu'
    password = '19cee1Et742'
    recipient = '100002834091853'  #Anna: 100002834091853, Chris: 100001479799294
    client = fbchat.Client(username, password)
    msg = Message(text='{} ist erfolgreich durchgelaufen! \U0001F973 '
                       '\n(Gesamtlaufzeit {:.0f}min {:.0f}s)'.format(model_name, m, s))

    sent = client.send(msg, thread_id=recipient, thread_type=ThreadType.USER)
    client.logout()

def crossValidation(load_file, X_train_aug, y_train, n_folds=10):
    X_train, X_aug_train = X_train_aug
    # Instantiate the cross validator
    kfold_splits = n_folds
    kf = KFold(n_splits=kfold_splits, shuffle=True)

    cv_scores = []
    model_history = []

    # Loop through the indices the split() method returns
    for index, (train_indices, val_indices) in enumerate(kf.split(X_train, y_train)):
        print('\n\n----------------------')
        print('----------------------')
        print("Training on fold " + str(index + 1) + "/" + str(kfold_splits) +"...")
        print('----------------------')

        # Generate batches from indices
        X_train_fold, X_val_fold = X_train[train_indices], X_train[val_indices]
        X_aug_train_fold, X_aug_val_fold = X_aug_train[train_indices], X_aug_train[val_indices]
        y_train_fold, y_val_fold = y_train[train_indices], y_train[val_indices]

        print("Training new iteration on " + str(X_train_fold.shape[0]) + " training samples, " + str(
            X_val_fold.shape[0]) + " validation samples...")

        model= train_model([X_train_fold, X_aug_train_fold], y_train_fold,
                                  [X_val_fold, X_aug_val_fold], y_val_fold)

        test_acc = evaluate_model(model, load_file, test_ind = [0])

        print('>%.3f' % test_acc)
        cv_scores.append(test_acc)
        model_history.append(model)

    return cv_scores, model_history
