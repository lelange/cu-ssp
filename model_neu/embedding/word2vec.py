from gensim.models import Word2Vec

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
from keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau

import sys
import os
import time
import dill as pickle
from collections import defaultdict
from datetime import datetime

data_root = '/nosave/lange/cu-ssp/data/'

EMB_DIM = 50
WINDOW_SIZE = 5
NB_NEG = 5
NB_ITER = 10

def seq2ngrams(seqs, n = 3):
    return np.array( [[seq[i:i+n] for i in range(int(len(seq)-2))] for seq in seqs])


def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        if np.max(o) != 0:
            i = np.argmax(o)
            s += index[i]
        else:
            break
    return s


#def AA2index():

#def index2AA():


primary_list_princeton = list('ACEDGFIHKMLNQPSRTWVYX') + ['NoSeq']
q8_list_princeton = list('LBEGIHST') + ['NoSeq']

def get_princeton_data(filename, max_len=700):
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
        residue_str_list.append(x)
    for vec in q8_array:
        x = ''.join(vec[vec != 'NoSeq'])
        q8_str_list.append(x)

    out = pd.DataFrame({'input_AA': residue_str_list, 'input_onehot':residue_onehot,
                        'output_AA': q8_str_list, 'output_onehot':residue_q8_onehot,
                        'pssm':profile_padded})
    return out

def get_netsurf_data(filename, max_len=None):
    # filename =
    seq_list = list('ACDEFGHIKLMNPQRSTVWY')
    path = data_root+'netsurfp/'

    input_onehot = np.load(path+filename + '_input.npy')
    q8_onehot = np.load(path+filename + '_q8.npy')
    profiles = np.load(path+filename + '_hmm.npy')

    prim_seq = []
    for i, oh in enumerate(input_onehot):
        seq = onehot_to_seq(oh, seq_list)
        prim_seq.append(seq)

    out = pd.DataFrame({'input_AA': prim_seq, 'input_onehot': input_onehot,
                        'output_onehot': q8_onehot,
                        'hmm': profiles})

    return out

def get_qzlshy_data(filename, maxlen=700):
    ### filename = train or test
    path = data_root+'data_qzlshy/'

    pssm = np.load(path+filename+'_pssm.npy')
    hmm = np.load(path+filename+'_hmm.npy')
    input = np.load(path+filename+'_input.npy')
    q8 = np.load(path+filename+'_q8.npy')

    out = pd.DataFrame({'input_AA': input,
                        'output_AA': q8,
                        'pssm': pssm,
                        'hmm':hmm})
    return out

def load_data(dataname, mode):
    if dataname=='princeton':
        if mode=='train':
            filename = 'cb6133'
        if mode == 'test':
            filename= 'cb513'
        return get_princeton_data(filename)
    if dataname == 'netsurf':
        if mode == 'train':
            filename = 'train_full'
        if mode == 'test':
            filename = 'cb513_full'
        return get_netsurf_data(filename)
    if dataname == 'qzlshy':
        return get_qzlshy_data(mode)




def embed_data(dataname, mode):
    ''' should work for train and test '''
    # load input data

    data = load_data(dataname, mode)

    #onehot2AA
    seqs = data['input_AA']
    #create n-grams from AA sequence
    ngram_seq = seq2ngrams(seqs, n=3)


    #tokenize n-gram sequences (indices according to frequency)

    #


    w2v = Word2Vec(ngram_seq, size=EMB_DIM, window=WINDOW_SIZE, negative=NB_NEG, iter= NB_ITER)
    word_vectors = w2v.wv
    embedding_matrix = word_vectors.vectors







