import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import pickle

maxlen_seq = 768 ###! change back!
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

data_root = '/nosave/lange/cu-ssp/data/netsurfp/'

data_train = np.load(data_root+'Train_HHblits.npz')
data_cb513 = np.load(data_root+'CB513_HHblits.npz')
data_ts115 = np.load(data_root+'TS115_HHblits.npz')
data_casp12 = np.load(data_root+'CASP12_HHblits.npz')

def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        i = np.argmax(o)
        s += index[i]
    return s

def seq2onehot(seqs, n):
    out = np.zeros((len(seqs), maxlen_seq, n))
    for i, seq in enumerate(seqs):
        for j in range(len(seq)):
            out[i, j, seq[j]] = 1
    return out

#q3_list = list('HHHEECCC')

def get_mask(data):
    return data[:,:,50]

def get_input(data, seq_range):
    return data[:,:maxlen_seq,:20][seq_range]

def get_hmm(data, seq_range):
    return data[:,:maxlen_seq,20:50][seq_range]

def get_q8(data, seq_range):
    return data[:,:maxlen_seq,57:65][seq_range]

def create_var_length_list(arr, lengths):
    return np.array([line[:int(len)] for line, len in zip(arr, lengths)])

#usage: create_var_length_list(arr, np.sum(new_mask, axis = 1))

#add extra dimension for no sequence in one hot encoded labels

def get_and_save_data(data, filename):
    database = data['data']
    mask = get_mask(database)
    if maxlen_seq is None:
        seq_range = range(len(mask))
    else:
        seq_range= [minlen_seq<=mask[i].sum()<=maxlen_seq for i in range(len(mask))]
    new_mask = mask[seq_range,:maxlen_seq]
    input_seq = get_input(database, seq_range)
    q8 = get_q8(database, seq_range)
    hmm = get_hmm(database, seq_range)
    print('Input shape: ', input_seq.shape)
    print('mask shape: '+str(mask.shape))
    print('mask seq_range shape: '+str(new_mask.shape))
    print('q8 shape: '+ str(q8.shape))
    q9 = np.pad(q8, ((0,0),(0,0),(1,0)), 'constant')
    print('q9 shape:' + str(q9.shape))
    print(np.sum(q9[:,:,0], axis=1))
    print(np.sum(q9[:, :, 8], axis=1))
    print(np.sum(new_mask, axis=1))
    print(np.sum(1-new_mask, axis=1))
    q9[:,:,0]= 1-new_mask
    print('hmm shape: ', hmm.shape)
    var_len_input_seq = create_var_length_list(input_seq, np.sum(new_mask, axis = 1))
    var_len_hmm = create_var_length_list(hmm, np.sum(new_mask, axis = 1))
    var_len_q9 = create_var_length_list(q9, np.sum(new_mask, axis = 1))

    #np.save(data_root + filename + '_var_len_input.npy', var_len_input_seq)
    #np.save(data_root + filename + '_var_len_hmm.npy', var_len_hmm)
    #np.save(data_root + filename + '_var_len_q9.npy', var_len_q9)
    #pickle.dump(var_len_input_seq, open(data_root + filename + '_var_len_input.txt', "wb"))
    #pickle.dump(var_len_hmm, open(data_root + filename + '_var_len_hmm.txt', "wb"))
    #pickle.dump(var_len_q9, open(data_root + filename + '_var_len_q9.txt', "wb"))

    np.save(data_root+filename+'_input.npy', input_seq)
    np.save(data_root+filename+'_q9.npy', q9)
    np.save(data_root+filename+'_hmm.npy', hmm)
    np.save(data_root + filename + '_mask.npy', new_mask)
    print(filename+' is saved.')

if maxlen_seq is None:
    ending = "full"
else:
    ending = str(maxlen_seq)

get_and_save_data(data_train, 'train_'+ending)
get_and_save_data(data_cb513, 'cb513_'+ending)
get_and_save_data(data_ts115, 'ts115_'+ending)
get_and_save_data(data_casp12, 'casp12_'+ending)


# prepare q8 and q3 data
'''
def get_and_save_data(data, filename):

    database = data['data']
    train_q8 = database[:, :, 57:65]
    mask = database[:, :, 50]
    seq_range = [minlen_seq <= mask[i].sum() <= maxlen_seq for i in range(len(mask))]
    mask_seq = mask[seq_range]

    q3_seq = []
    for i, q3 in enumerate(train_q8[seq_range]):
        seq = onehot_to_seq(q3[mask_seq[i] > 0], q3_list)
        q3_seq.append(seq)

    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(q3_seq[0])
    q3 = tokenizer.texts_to_sequences(q3_seq)
    q3 = np.array([np.array(q) - 1 for q in q3])
    q3_arr = seq2onehot(q3, 3)
    print(q3_arr.shape)
    print(q3_arr[0])
    np.save(data_root + filename + '_q3.npy', q3_arr)
    print(filename + ' is saved.')

get_and_save_data(data_train, 'train_700')
'''
