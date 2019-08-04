from gensim.models import Word2Vec

import numpy as np
import multiprocessing

data_root = '/nosave/lange/cu-ssp/data/'

EMB_DIM = 50
WINDOW_SIZE = 20
NB_NEG = 5
NB_ITER = 10

def seq2ngrams(seqs, n = 3):
    if n==1:
        return seqs
    else:
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


    return residue_str_list, residue_onehot, q8_str_list, residue_q8_onehot, profile_padded


def get_netsurf_data(filename, max_len=None):
    # filename =
    seq_list = list('ACDEFGHIKLMNPQRSTVWY')
    path = data_root + 'netsurfp/'

    input_onehot = np.load(path + filename + '_input.npy')
    q8_onehot = np.load(path + filename + '_q9.npy')
    profiles = np.load(path + filename + '_hmm.npy')
    prim_seq = np.load(path + filename + '_q9_AA_str.npy')

    return prim_seq, input_onehot, q8_onehot, profiles

def get_qzlshy_data(filename, maxlen=700):
    ### filename = train or test
    path = data_root+'data_qzlshy/'

    pssm = np.load(path+filename+'_pssm.npy')
    hmm = np.load(path+filename+'_hmm.npy')
    input_AA = np.load(path+filename+'_input.npy')
    q8_AA = np.load(path+filename+'_q8.npy')

    return input_AA, q8_AA, pssm, hmm

def load_data(dataname, mode):
    if dataname=='princeton':
        if mode=='train':
            filename = 'cb6133'
        if mode == 'test':
            filename= 'cb513'
        return get_princeton_data(filename)
    if dataname == 'netsurfp':
        if mode == 'train':
            filename = 'train_full'
        if mode == 'test':
            filename = 'cb513_full'
        return get_netsurf_data(filename)
    if dataname == 'qzlshy':
        return get_qzlshy_data(mode)


def embed_data(dataname='netsurfp', mode='train', data=None):
    ''' should work for train and test '''
    # load input data

    if data is None:
        data = load_data(dataname, mode)
    print('Load data..')
    #onehot2AA
    seqs = data[0]
    #create n-grams from AA sequence
    print('Create n-grams...')
    ngram_seq = seq2ngrams(seqs, n=1)

    #tokenize n-gram sequences (indices according to frequency)

    #
    print('Perform Word2Vec embedding...')

    w2v = Word2Vec(ngram_seq, size=EMB_DIM, window=WINDOW_SIZE,
                   negative=NB_NEG, iter= NB_ITER,
                   workers = multiprocessing.cpu_count())
    word_vectors = w2v.wv
    embedding_matrix = word_vectors.vectors
    return embedding_matrix


datanames = ['princeton', 'netsurfp', 'qzlshy']

w2v_matrix = embed_data()

np.save(data_root+'netsurfp/embedding/train_input_700_word2vec.npy', w2v_matrix)

print('Data has been saved to '+data_root+'netsurfp/embedding/train_input_700_word2vec.npy')




