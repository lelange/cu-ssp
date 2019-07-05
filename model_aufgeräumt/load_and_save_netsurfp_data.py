import numpy as np
import pandas as pd

maxlen_seq = 600
minlen_seq= 100

# [0:20] Amino Acids (sparse encoding)
# Unknown residues are stored as an all-zero vector
# [20:50] hmm profile
# [50] Seq mask (1 = seq, 0 = empty)
# [51] Disordered mask (0 = disordered, 1 = ordered)
# [52] Evaluation mask (For CB513 dataset, 1 = eval, 0 = ignore)
# [53] ASA (isolated)
# [54] ASA (complexed)
# [55] RSA (isolated)
# [56] RSA (complexed)
# [57:65] Q8 GHIBESTC (Q8 -> Q3: HHHEECCC)
# [65:67] Phi+Psi
# [67] ASA_max

data_root = '../data/netsurfp/'

data_train = np.load(data_root+'Train_HHblits.npz')
data_cb513 = np.load(data_root+'CB513_HHblits.npz')
data_ts115 = np.load(data_root+'TS115_HHblits.npz')
data_casp12 = np.load(data_root+'CASP12_HHblits.npz')

def get_mask(data):
    return data[:,:,50]

def get_input(data, seq_range):
    return data[:,:maxlen_seq,:20][seq_range]

def get_hmm(data, seq_range):
    return data[:,:maxlen_seq,20:50][seq_range]

def get_q8(data, seq_range):
    return data[:,:maxlen_seq,57:65][seq_range]

def get_and_save_data(data, filename):
    database = data['data']
    mask = get_mask(database)
    seq_range = [minlen_seq<=mask[i].sum()<=maxlen_seq for i in range(len(mask))]
    input_seq = get_input(database, seq_range)
    q8 = get_q8(database, seq_range)
    hmm = get_hmm(database, seq_range)
    print('Input shape: ', input_seq.shape)
    print('q8 shape: ', q8.shape)
    print('hmm shape: ', hmm.shape)
    np.save(data_root+filename+'_input.npy', input_seq)
    np.save(data_root+filename+'_q8.npy', q8)
    np.save(data_root+filename+'_hmm.npy', hmm)
    print(filename+' is saved.')


get_and_save_data(data_train, 'train')
get_and_save_data(data_cb513, 'cb513')
get_and_save_data(data_ts115, 'ts115')
get_and_save_data(data_casp12, 'casp12')


