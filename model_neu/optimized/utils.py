from bson import json_util
import json
import os
import numpy as np
import tensorflow as tf
import keras.backend as K
import time

import multiprocessing
from gensim.models import Word2Vec

RESULTS_DIR = "results/"
MAXLEN_SEQ = 700

"""Json utils to print, save and load training results."""

def print_json(result):
    """Pretty-print a jsonable structure (e.g.: result)."""
    print(json.dumps(
        result,
        default=json_util.default, sort_keys=True,
        indent=4, separators=(',', ': ')
    ))


def save_json_result(model_name, result):
    """Save json to a directory and a filename."""
    result_name = '{}.txt.json'.format(model_name)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    with open(os.path.join(RESULTS_DIR, result_name), 'w') as f:
        json.dump(
            result, f,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        )


def load_json_result(best_result_name):
    """Load json from a path (directory + filename)."""
    result_path = os.path.join(RESULTS_DIR, best_result_name)
    with open(result_path, 'r') as f:
        return json.JSONDecoder().decode(
            f.read()
            # default=json_util.default,
            # separators=(',', ': ')
        )


def load_best_hyperspace(name = 'json'):
    results = [
        f for f in list(sorted(os.listdir(RESULTS_DIR))) if name in f
    ]
    if len(results) == 0:
        return None

    best_result_name = results[-1]
    return load_json_result(best_result_name)["space"]

# The custom accuracy metric used for this task
def accuracy(y_true, y_predicted):
    y = tf.argmax(y_true, axis =- 1)
    y_ = tf.argmax(y_predicted, axis =- 1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())

def train_val_split(X_train_aug, y_train, hmm=True, perc = None):

    n_samples = len(y_train)
    np.random.seed(0)
    if perc is None:
        perc = 0.1
    size = int(n_samples*perc)
    validation_idx = np.random.choice(np.arange(n_samples), size=size, replace=False)
    training_idx = np.array(list(set(np.arange(n_samples)) - set(validation_idx)))

    y_val = y_train[validation_idx]
    y_train = y_train[training_idx]

    if hmm:
        X_val_aug = np.concatenate((X_train_aug[0], X_train_aug[1]), axis=2)[validation_idx]
        X_train_aug = np.concatenate((X_train_aug[0], X_train_aug[1]), axis = 2)[training_idx]
    else:
        X_val_aug = X_train_aug[validation_idx]
        X_train_aug = X_train_aug[training_idx]

    return X_train_aug, y_train, X_val_aug, y_val


def get_test_data(filename):
    data_root = '/nosave/lange/cu-ssp/data/netsurfp/'

    X_test = np.load(data_root + filename + '_input.npy')
    profiles = np.load(data_root + filename + '_hmm.npy')
    mean = np.mean(profiles)
    std = np.std(profiles)
    X_aug_test = (profiles - mean) / std
    X_test_aug = np.concatenate((X_test, X_aug_test), axis=2)
    y_test = np.load(data_root + filename + '_q9.npy')

    return  X_test_aug, y_test


def train_val_split_(X_train, y_train, X_aug):
    n_samples = len(y_train)
    np.random.seed(0)
    perc = 0.1
    size = int(n_samples * perc)
    validation_idx = np.random.choice(np.arange(n_samples), size=size, replace=False)
    training_idx = np.array(list(set(np.arange(n_samples)) - set(validation_idx)))

    y_val = y_train[validation_idx]
    y_train = y_train[training_idx]

    X_val = X_train[validation_idx]
    X_aug_val = X_aug[validation_idx]
    X_train = X_train[training_idx]
    X_aug = X_aug[training_idx]

    X_aug = standard(X_aug)
    X_aug_val = standard(X_aug_val)

    return X_train, y_train, X_aug, X_val, y_val, X_aug_val


def standard(data):
    mean = np.mean(data)
    std = np.std(data)
    data_ = (data - mean) / std
    return data_


def seq2ngrams2(seqs, n=3):
    if n == 1:
        return seqs
    else:
        return np.array([[seq[i:i + n] for i in range(int(len(seq) - 2))] for seq in seqs])

def seq2ngrams(seqs, n):
    """
    'AGAMQSASM' => [['AGA', 'MQS', 'ASM'], ['GAM','QSA'], ['AMQ', 'SAS']]
    """
    if n == 1:
        return seqs

    if n == 2:
        result = []
        for seq in seqs:
            a, b = zip(*[iter(seq)] * n), zip(*[iter(seq[1:])] * n)
            str_ngrams = []
            for ngrams in [a, b]:
                for ngram in ngrams:
                    str_ngrams.append("".join(ngram))
            result.append(str_ngrams)
        return result

    if n == 3:
        result = []
        for seq in seqs:
            a, b, c = zip(*[iter(seq)] * n), zip(*[iter(seq[1:])] * n), zip(*[iter(seq[2:])] * n)
            str_ngrams = []
            for ngrams in [a, b, c]:
                for ngram in ngrams:
                    str_ngrams.append("".join(ngram))
            result.append(str_ngrams)
        return result

    if n == 4:
        result = []
        for seq in seqs:
            a, b, c = zip(*[iter(seq)] * n), zip(*[iter(seq[1:])] * n), zip(*[iter(seq[2:])] * n)
            d = zip(*[iter(seq[3:])] * n)
            str_ngrams = []
            for ngrams in [a, b, c, d]:
                for ngram in ngrams:
                    str_ngrams.append("".join(ngram))
            result.append(str_ngrams)
        return result

    else:
        result = []
        for seq in seqs:
            a, b, c = zip(*[iter(seq)] * n), zip(*[iter(seq[1:])] * n), zip(*[iter(seq[2:])] * n)
            d, e = zip(*[iter(seq[3:])] * n), zip(*[iter(seq[4:])] * n)
            str_ngrams = []
            for ngrams in [a, b, c, d, e]:
                for ngram in ngrams:
                    str_ngrams.append("".join(ngram))
            result.append(str_ngrams)
        return result

def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        if np.max(o) != 0:
            i = np.argmax(o)
            s += index[i]
        else:
            break
    return s


def get_netsurf_data(filename):
    data_root = '/nosave/lange/cu-ssp/data/'
    seq_list = list('ACDEFGHIKLMNPQRSTVWY')
    path = data_root + 'netsurfp/'

    # input_onehot = np.load(path + filename + '_input.npy')
    q8_onehot = np.load(path + filename + '_q9.npy')
    profiles = np.load(path + filename + '_hmm.npy')
    prim_seq = np.load(path + filename + '_q9_AA_str.npy')

    return prim_seq, q8_onehot[:, :MAXLEN_SEQ, :], profiles[:, :MAXLEN_SEQ, :]


def get_embedding(emb_dim, window_size, nb_neg, nb_iter, n_gram, mod,
                  filename=None, seqs=None, tokens=1):
    start_time = time.time()
    if seqs is None:
        data = get_netsurf_data(filename)
        print('Load data..')
        # onehot2AA
        seqs = data[0]

    # create n-grams from AA sequence
    print('Create n-grams for n = {}...'.format(n_gram))
    if tokens==1:
        ngram_seq = seq2ngrams(seqs, n=n_gram)
    else:
        ngram_seq = split_ngrams(seqs, n=n_gram)

    print('Word2Vec embedding...')

    w2v = Word2Vec(ngram_seq,
                   size=emb_dim,
                   window=window_size,
                   negative=nb_neg,
                   iter=nb_iter,
                   sg = mod,
                   min_count=1,
                   workers=multiprocessing.cpu_count())

    word_vectors = w2v.wv
    print('We have ' + str(len(word_vectors.vocab)) + ' n-grams.')

    m, s = divmod(time.time() - start_time, 60)
    print("Needed {:.0f}min {:.0f}s for W2V embedding.".format(m, s))

    return w2v


def embed_data(seqs, w2v, emb_dim, n_gram, tokens=1):

    print('Create n-grams for embedding...')

    start_time= time.time()

    if tokens==1:
        ngram_seq = seq2ngrams(seqs, n=n_gram)
    else:
        ngram_seq = split_ngrams(seqs, n=n_gram)

    n_time = time.time()
    end_time = n_time - start_time
    m, s = divmod(end_time, 60)
    print("Needed {:.0f}min {:.0f}s to create n-grams.".format(m, s))

    print('Emded data...')
    embed_seq = np.zeros((len(seqs), MAXLEN_SEQ, emb_dim))
    f = 0
    c = 0
    for i, grams in enumerate(ngram_seq):
        for j, g in enumerate(grams[:MAXLEN_SEQ]):
            try:
                embed_seq[i, j, :] = w2v.wv[g]
                c+=1
            except:
                print('Model not trained for '+g)
                f+=1
    print('{} n-grams are embedded but for {} the model has not been trained.'.format(c,f))

    m,s = divmod(time.time() - n_time, 60)
    print("Needed {:.0f}min {:.0f}s to embed data.".format(m, s))

    return embed_seq

