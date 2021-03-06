from gensim.models import Word2Vec

import numpy as np
import pickle
import multiprocessing

import time
data_root = '/nosave/lange/cu-ssp/data/'

'''
EMB_DIM = 50
WINDOW_SIZE = 20
NB_NEG = 5
NB_ITER = 10

'''
EMB_DIM = 230
WINDOW_SIZE = 20
NB_NEG = 3
NB_ITER = 24
N_GRAM = 1


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
print('Program has started...')

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
    '''
    prim_seq = []
    for i, oh in enumerate(input_onehot):
        seq = onehot_to_seq(oh, seq_list)
        prim_seq.append(seq)

    np.save(path+filename + '_q9_AA_str.npy', prim_seq)
    print('saved AA '+filename+' to disk.')
    '''
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

def seq2ngrams2(seqs, n = 3):
    if n==1:
        return seqs
    else:
        result = []
        n_begin = int((n-1)/2)
        n_end = (n-1) - n_begin
        for seq in seqs:
            seq = ('C'*n_begin)+seq+('C'*n_end)

            result.append([seq[i:i + n] for i in range(int(len(seq)))])

        return np.array(result)

def seq2ngrams(seqs, n):
    """
    'AGAMQSASM' => [['AGA', 'MQS', 'ASM'], ['GAM','QSA'], ['AMQ', 'SAS']]
    """
    result = []
    for seq in seqs:
        a, b, c = zip(*[iter(seq)] * n), zip(*[iter(seq[1:])] * n), zip(*[iter(seq[2:])] * n)
        str_ngrams = []
        for ngrams in [a, b, c]:
            for ngram in ngrams:
                str_ngrams.append("".join(ngram))
        result.append(str_ngrams)
    return result


def get_embedding(dataname='netsurfp', mode='train', data=None, n_gram = N_GRAM):
    start_time=time.time()
    # load input data
    if data is None:
        data = load_data(dataname, mode)
    print('Load data..')
    #onehot2AA
    seqs = data[0]
    #create n-grams from AA sequence
    print('Create n-grams for n = {}...'.format(n_gram))
    ngram_seq = seq2ngrams(seqs, n=n_gram)

    print('Perform Word2Vec embedding...')
    w2v = Word2Vec(ngram_seq, size=EMB_DIM, window=WINDOW_SIZE,
                   negative=NB_NEG, iter= NB_ITER, min_count=1, sg=1,
                   workers = multiprocessing.cpu_count())

    embed_time = time.time()
    m,s = divmod(embed_time-start_time, 60)
    print("Needed {:.0f}min {:.0f}s for W2V embedding.".format(m, s))

    word_vectors = w2v.wv
    print('We have '+str(len(word_vectors.vocab))+ ' n-grams.')
    return w2v

def embed_data(seqs, model, n_gram=N_GRAM):

    embed_seq = np.zeros((len(seqs), 700, EMB_DIM))
    ngram_seq = seq2ngrams(seqs, n=n_gram)
    #new_model = model.train(ngram_seq)

    for i, grams in enumerate(ngram_seq):
        for j, g in enumerate(grams[:700]):
            embed_seq[i, j, :] = model.wv[g]


    print(embed_seq.shape)
    return embed_seq

datanames = ['princeton', 'netsurfp', 'qzlshy']
nb_try = 4




try:

    w2v_dict = Word2Vec.load('word2vec'+str(nb_try)+'.model')
    print('Found model! Load...')

except:
    print('No w2v model found. Get new embedding...')
    start_time = time.time()
    w2v_dict = get_embedding(mode='train')
    time_end = time.time() - start_time
    m, s = divmod(time_end, 60)
    print("The program needed {:.0f}min {:.0f}s to generate the embedding.".format(m, s))

    w2v_dict.save('word2vec' + str(nb_try) + '.model')
    print('Saved model to '+'word2vec' + str(nb_try) + '.model')



start_time = time.time()


w2v_input = embed_data(get_netsurf_data('train_full')[0], w2v_dict)
np.save(data_root+'netsurfp/embedding/train_full_700_input_word2vec_'+str(nb_try)+'.npy', w2v_input)
print('Data has been saved to '+data_root+'netsurfp/embedding/train_full_700_input_word2vec_'+str(nb_try)+'.npy')

time_end = time.time()-start_time
m, s = divmod(time_end, 60)

print("The program needed {:.0f}min {:.0f}s to embed training data.".format(m, s))

w2v_input = embed_data(get_netsurf_data('cb513_full')[0], w2v_dict)
np.save(data_root+'netsurfp/embedding/cb513_full_700_input_word2vec_'+str(nb_try)+'.npy', w2v_input)
print('Data has been saved to '+data_root+'netsurfp/embedding/cb513_full_700_input_word2vec_'+str(nb_try)+'.npy')

w2v_input = embed_data(get_netsurf_data('casp12_full')[0], w2v_dict)
np.save(data_root+'netsurfp/embedding/casp12_full_700_input_word2vec_'+str(nb_try)+'.npy', w2v_input)
print('Data has been saved to '+data_root+'netsurfp/embedding/casp12_full_700_input_word2vec_'+str(nb_try)+'.npy')

w2v_input = embed_data(get_netsurf_data('ts115_full')[0], w2v_dict)
np.save(data_root+'netsurfp/embedding/ts115_full_700_input_word2vec_'+str(nb_try)+'.npy', w2v_input)
print('Data has been saved to '+data_root+'netsurfp/embedding/ts115_full_700_input_word2vec_'+str(nb_try)+'.npy')




