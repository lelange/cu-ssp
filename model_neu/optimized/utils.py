from bson import json_util
import json
import os
import numpy as np
import tensorflow as tf
import keras.backend as K

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


def seq2ngrams(seqs, n=3):
    if n == 1:
        return seqs
    else:
        return np.array([[seq[i:i + n] for i in range(int(len(seq) - 2))] for seq in seqs])


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


def get_embedding(emb_dim, window_size, nb_neg, nb_iter, n_gram,
                  filename=None, seqs=None):
    if seqs is None:
        data = get_netsurf_data(filename)
        print('Load data..')
        # onehot2AA
        seqs = data[0]

    # create n-grams from AA sequence
    print('Create n-grams...')
    ngram_seq = seq2ngrams(seqs, n=n_gram)
    print('Perform Word2Vec embedding...')

    w2v = Word2Vec(ngram_seq, size=emb_dim, window=window_size,
                   negative=nb_neg, iter=nb_iter,
                   workers=multiprocessing.cpu_count())
    word_vectors = w2v.wv
    embedding_matrix = word_vectors.vectors
    l = []
    for item in embedding_matrix:
        l.append(item[0])
    if len(np.unique(l)) != len(list('ACDEFGHIKLMNPQRSTVWY')):
        print('ERRRRRRRRRRRRROR!_________________________________________________')
    index2embedding = {}
    for item in list('ACDEFGHIKLMNPQRSTVWY'):
        index2embedding.update({item: embedding_matrix[l.index(word_vectors[item][0])]})
    return index2embedding


def embed_data(seqs, index2embedding, emb_dim, n_gram):
    embed_seq = np.zeros((len(seqs), MAXLEN_SEQ, emb_dim))
    ngram_seq = seq2ngrams(seqs, n=n_gram)

    for i, grams in enumerate(ngram_seq):
        for j, g in enumerate(grams[:MAXLEN_SEQ]):
            embed_seq[i, j, :] = index2embedding[g]

    return embed_seq


def evaluate_model(model, load_file, emb_dim, n_gram, index2embed):
    print(load_file)
    file_test = ['cb513_full', 'ts115_full', 'casp12_full']
    test_accs = []
    names = []
    for test in file_test:
        X_test, y_test, X_aug = get_netsurf_data(test)
        print(X_test.shape)
        X_embed = embed_data(X_test, index2embed, emb_dim, n_gram)
        print(emb_dim)
        print(X_embed.shape)
        X_test_aug = [X_embed, X_aug]
        model.load_weights(load_file)
        score = model.evaluate(X_test_aug, y_test, verbose=2, batch_size=1)
        print(model.metrics_names)
        print(score)
        test_accs.append(score[1])
        names.append(test)
    return dict(zip(names, test_accs))