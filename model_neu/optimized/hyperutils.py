from bson import json_util
import json
import os
import numpy as np
import tensorflow as tf
from keras.layers.core import K  #import keras.backend as K
import time
import multiprocessing

RESULTS_DIR = "results/"
MAXLEN_SEQ = 700

residue_list = list('ACEDGFIHKMLNQPSRTWVYX') + ['NoSeq']
q8_list      = list('LBEGIHST') + ['NoSeq']

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


# transformations for pssm:
def sigmoid_p(data):
    return logistic.cdf(data)

# transformations for hmm:
def normal_h(data):
    return 2**((-data/1000))

# for both:
def standard(data):
    mean = np.mean(data)
    std = np.std(data)
    data_ = (data - mean) / std
    return data_

# Computes and returns the n-grams of a particular sequence, defaults to trigrams
def seq2ngrams(seqs, n = 1):
    return np.array([[seq[i : i + n] for i in range(len(seq))] for seq in seqs])

## metrics for this task:

# The custom accuracy metric used for this task
def accuracy(y_true, y_predicted):
    y = tf.argmax(y_true, axis =- 1)
    y_ = tf.argmax(y_predicted, axis =- 1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())

def weighted_accuracy(y_true, y_pred):
    return K.sum(K.equal(K.argmax(y_true, axis=-1),
                  K.argmax(y_pred, axis=-1)) * K.sum(y_true, axis=-1)) / K.sum(y_true)

# losses:
def nll(y_true, y_pred):
    """ Negative log likelihood. """

    # keras.losses.binary_crossentropy give the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


def get_data(npy_path, normalize_profiles):

    # daten durcheinander würfeln?

    data = np.load(npy_path+'.npy')
    max_len = 700

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

    input_data = [input_one_hot, input_seqs, input_pssm, input_hmm]

    output_data = [q8_onehot, rsa_onehot]

    return input_data, output_data

# fit_on_texts Updates internal vocabulary based on a list of texts
# texts_to_sequences Transforms each text in texts to a sequence of integers, 0 is reserved for padding

#fertig, nur get_data noch machen
def evaluate_model(model, load_file, hype_space):

    start_time = time.time()
    file_test = ['cb513'] #add more later
    test_accs = []

    for test in file_test:
        X_test, y_test = get_data(test, hype_space['normalize_profiles'])

        model.load_weights(load_file)

        score = model.evaluate(X_test, y_test, verbose=2, batch_size=1)

        for metric, s in zip(model.metrics_names, score):
            print(test + ' test ', metric, ': ', s)

        test_accs.append(score[1])

    m, s = divmod(time.time() - start_time, 60)
    print("Needed {:.0f}min {:.0f}s to evaluate model.".format(m, s))

    return dict(zip(file_test, test_accs))


def load_6133_filted():
    '''
    TRAIN data Cullpdb+profile_6133_filtered
    Test data  CB513\CASP10\CASP11
    '''
    print("Loading train data (Cullpdb_filted)...")
    data = np.load()
    data = np.reshape(data, (-1, 700, 57))
    # print data.shape
    datahot = data[:, :, 0:21]  # sequence feature
    # print 'sequence feature',dataonehot[1,:3,:]
    datapssm = data[:, :, 35:56]  # profile feature
    # print 'profile feature',datapssm[1,:3,:]
    labels = data[:, :, 22:30]  # secondary struture label , 8-d
    # shuffle data
    # np.random.seed(2018)
    num_seqs, seqlen, feature_dim = np.shape(data)
    num_classes = labels.shape[2]
    seq_index = np.arange(0, num_seqs)  #
    np.random.shuffle(seq_index)

    # train data
    trainhot = datahot[seq_index[:5278]]  # 21
    trainlabel = labels[seq_index[:5278]]  # 8
    trainpssm = datapssm[seq_index[:5278]]  # 21

    # val data
    vallabel = labels[seq_index[5278:5534]]  # 8
    valpssm = datapssm[seq_index[5278:5534]]  # 21
    valhot = datahot[seq_index[5278:5534]]  # 21

    train_hot = np.ones((trainhot.shape[0], trainhot.shape[1]))
    for i in xrange(trainhot.shape[0]):
        for j in xrange(trainhot.shape[1]):
            if np.sum(trainhot[i, j, :]) != 0:
                train_hot[i, j] = np.argmax(trainhot[i, j, :])

    val_hot = np.ones((valhot.shape[0], valhot.shape[1]))
    for i in xrange(valhot.shape[0]):
        for j in xrange(valhot.shape[1]):
            if np.sum(valhot[i, j, :]) != 0:
                val_hot[i, j] = np.argmax(valhot[i, j, :])

    solvindex = range(33, 35)

    trainsolvlabel = data[:5600, :, solvindex]
    trainsolvvalue = trainsolvlabel[:, :, 0] * 2 + trainsolvlabel[:, :, 1]
    trainsolvlabel = np.zeros((trainsolvvalue.shape[0], trainsolvvalue.shape[1], 4))
    for i in xrange(trainsolvvalue.shape[0]):
        for j in xrange(trainsolvvalue.shape[1]):
            if np.sum(trainlabel[i, j, :]) != 0:
                trainsolvlabel[i, j, trainsolvvalue[i, j]] = 1

    return train_hot, trainpssm, trainlabel, val_hot, valpssm, vallabel

