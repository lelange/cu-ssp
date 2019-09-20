#create embedded input sequence with elmo embedder
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from allennlp.commands.elmo import ElmoEmbedder
from keras.preprocessing import text, sequence
from pathlib import Path
import torch
import sys
import os
import argparse
import time
import numpy as np
import pandas as pd

import os
import argparse
import time
import numpy as np
import dill as pickle
import pandas as pd
import tensorflow as tf
import sys


model_dir = Path('/nosave/lange/seqVec')
weights = model_dir / 'weights.hdf5'
options = model_dir / 'options.json'
seqvec  = ElmoEmbedder(options,weights,cuda_device=0) # cuda_device=-1 for CPU

residue_list = list('ACEDGFIHKMLNQPSRTWVYX') + ['NoSeq']
q8_list      = list('LBEGIHST') + ['NoSeq']


cb513filename = '/nosave/lange/cu-ssp/data/data_princeton/cb513.npy'
cb6133filteredfilename = '/nosave/lange/cu-ssp/data/data_princeton/cb6133filtered.npy'
maxlen_seq = 700

# Computes and returns the n-grams of a particular sequence, defaults to trigrams
def seq2ngrams(seqs, n = 1):
    return np.array([[seq[i : i + n] for i in range(len(seq))] for seq in seqs])

# load train and test and cut length to maxlen_seq
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

train_df, X_aug_train = load_augmented_data(cb6133filteredfilename, maxlen_seq)
train_input_seqs, train_target_seqs = train_df[['input', 'expected']][(train_df.len <= maxlen_seq)].values.T

test_df, X_aug_test = load_augmented_data(cb513filename, maxlen_seq)
test_input_seqs, test_target_seqs = test_df[['input', 'expected']][(test_df.len <= maxlen_seq)].values.T

# Using the tokenizer to encode and decode the sequences for use in training
# use preprocessing tools for text from keras to encode input sequence as word rank numbers and target sequence as one hot.
# To ensure easy to use training and testing, all sequences are padded with zeros to the maximum sequence length
# transform sequences to trigrams
train_input_data = seq2ngrams(train_input_seqs)
# transform sequences
# fit alphabet on train basis

#test_target_data = sequence.pad_sequences(test_target_data, maxlen=maxlen_seq, padding='post')

#
def calculate_and_save_embedding(input_seq):
    # Get embedding for amino acid sequence

    # use list or numpy array to store data
    input_embedding = np.zeros((len(input_seq), 700, 1024))
    #input_embedding = []

    for i, seq in enumerate(input_seq):
        t1 = time.time()
        print('\n \n----------------------')
        print('----------------------')
        print('Sequence ', (i + 1), '/', len(input_seq))
        print('----------------------')


        #list() only if not list yet
        print(type(input_seq))
        print(seq)

        embedding = seqvec.embed_sentence(seq)  # List-of-Lists with shape [3,L,1024]

        # Get 1024-dimensional embedding for per-residue predictions:
        residue_embd = torch.tensor(embedding).sum(dim=0) # Tensor with shape [L,1024]
        # Get 1024-dimensional embedding for per-protein predictions:
        #protein_embd = torch.tensor(embedding).sum(dim=0).mean(dim=0)  # Vector with shape [1024]
        residue_embd_pad = residue_embd
        residue_embd_np = residue_embd_pad.cpu().detach().numpy()
        print(residue_embd_np.shape)

        #input_embedding.append(residue_embd_np)
        input_embedding[i, :, :]= residue_embd_np

        t = time.time() - t1
        print("For {} residues {:.0f}s needed.".format(len(input_seq), t))

    end_time = time.time() - start_time
    m, s = divmod(end_time, 60)
    print("The embedding calculation needed {:.0f}min {:.0f}s in total.".format(m, s))
    return input_embedding

start_time = time.time()

train_input_embedding= calculate_and_save_embedding(train_input_data)
#np.save(data_root+'train_netsurfp_input_embedding_residue.npy', train_input_embedding)

time_end = time.time() - start_time
m, s = divmod(time_end, 60)
print(m, s)
'''
start_time = time.time()
mask = get_seq_mask(data_cb513)
test_input_embedding, test_times = calculate_and_save_embedding(test_input, mask_seq=mask[:,:maxlen_seq])
np.save(data_root+'cb513_netsurfp_times_residue.npy', test_times)
np.save(data_root+'cb513_netsurfp_input_embedding_residue.npy', test_input_embedding)

time_end = time.time() - start_time
m, s = divmod(time_end, 60)
telegram_me(m, s, sys.argv[0])
'''


