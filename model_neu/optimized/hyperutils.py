from bson import json_util
import json
import os
import numpy as np
import tensorflow as tf
from keras.layers.core import K  #import keras.backend as K
import time
import pandas as pd
import multiprocessing
#
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

RESULTS_DIR = "results/"
MAXLEN_SEQ = 700

data_root = '/nosave/lange/cu-ssp/data/'
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

def kullback_leibler_divergence(y_true, y_pred):
    '''Calculates the Kullback-Leibler (KL) divergence between prediction
    and target values.
    '''
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)

def matthews_correlation(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    '''Calculates the F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


# losses:
def nll(y_true, y_pred):
    """ Negative log likelihood. """

    # keras.losses.binary_crossentropy give the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

'''
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
    id_list = np.arange(1, len(residue_array) + 1)
    len_list = np.array([len(x) for x in residue_str_list])
    train_df = pd.DataFrame({'id': id_list, 'len': len_list, 'input': residue_str_list, 'expected': q8_str_list})
    
    input_one_hot = residue_onehot
    q8_onehot = residue_q8_onehot
    
    train_input_seqs, train_target_seqs= train_df[['input', 'expected']][(train_df.len <= 700)].values.T
    input_seqs 
    input_pssm = profile_padded
    #SPÄTERE::
    #nput_hmm = None
    #rsa_onehot = None; output_data = [q8_onehot, rsa_onehot]
    #input_data = [input_one_hot, input_seqs, input_pssm, input_hmm]
    input_data = [input_one_hot, input_seqs, input_pssm]

    output_data = q8_onehot

    return input_data, output_data


'''
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

def get_data():
    cb513filename = data_root+'data_princeton/cb513.npy'
    cb6133filteredfilename = data_root+'data_princeton/cb6133filtered.npy'
    maxlen_seq = 700
    # load train and test and cut length to maxlen_seq

    train_df, X_aug_train = load_augmented_data(cb6133filteredfilename, maxlen_seq)
    train_input_seqs, train_target_seqs = train_df[['input', 'expected']][(train_df.len <= maxlen_seq)].values.T

    test_df, X_aug_test = load_augmented_data(cb513filename, maxlen_seq)
    test_input_seqs, test_target_seqs = test_df[['input', 'expected']][(test_df.len <= maxlen_seq)].values.T

    # Using the tokenizer to encode and decode the sequences for use in training
    # use preprocessing tools for text from keras to encode input sequence as word rank numbers and target sequence as one hot.
    # To ensure easy to use training and testing, all sequences are padded with zeros to the maximum sequence length
    # transform sequences to trigrams
    train_input_grams = seq2ngrams(train_input_seqs)
    # transform sequences
    # fit alphabet on train basis
    tokenizer_encoder = Tokenizer()
    tokenizer_encoder.fit_on_texts(train_input_grams)

    tokenizer_decoder = Tokenizer(char_level=True)
    tokenizer_decoder.fit_on_texts(train_target_seqs)

    # train
    train_input_data = tokenizer_encoder.texts_to_sequences(train_input_grams)
    X_train = sequence.pad_sequences(train_input_data, maxlen=maxlen_seq, padding='post')
    # transform targets to one-hot
    train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)
    train_target_data = sequence.pad_sequences(train_target_data, maxlen=maxlen_seq, padding='post')

    y_train = to_categorical(train_target_data)
    input_one_hot = to_categorical(X_train)

    # test
    test_input_grams = seq2ngrams(test_input_seqs)
    test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
    X_test = sequence.pad_sequences(test_input_data, maxlen=maxlen_seq, padding='post')
    test_target_data = tokenizer_decoder.texts_to_sequences(test_target_seqs)
    test_target_data = sequence.pad_sequences(test_target_data, maxlen=maxlen_seq, padding='post')
    y_test = to_categorical(test_target_data)
    input_one_hot_test = to_categorical(X_test)

    #### validation data
    '''
     n_samples = len(train_df)
    np.random.seed(0)
    validation_idx = np.random.choice(np.arange(n_samples), size=300, replace=False)
    training_idx = np.array(list(set(np.arange(n_samples)) - set(validation_idx)))

    X_val = X_train[validation_idx]
    X_train = X_train[training_idx]
    y_val = y_train[validation_idx]
    y_train = y_train[training_idx]
    X_aug_val = X_aug_train[validation_idx]
    X_aug_train = X_aug_train[training_idx]
    '''
    #hmm profiles
    input_hmm = np.load(data_root+'data_princeton/hmm_train.npy', allow_pickle=True)[:,:700,:]
    input_hmm_test = np.load(data_root+'data_princeton/hmm_cb513.npy', allow_pickle=True)[:,:700,:]

    #elmo embedding
    input_elmo_train = np.load(data_root+'data_princeton/train_input_embedding.npy')
    input_elmo_test = np.load(data_root+'data_princeton/cb513_input_embedding.npy')
    print(input_elmo_train.shape)

    input_data_train = [input_one_hot, X_train, input_elmo_train, standard(X_aug_train), input_hmm]
    output_data_train = y_train
    print(len(y_train))
    print(input_hmm.shape)
    print(len(y_test))
    print(input_hmm_test.shape)
    input_data_test = [input_one_hot_test, X_test, input_elmo_test, standard(X_aug_test), input_hmm_test]
    output_data_test = y_test

    return input_data_train, output_data_train, input_data_test, output_data_test

# fit_on_texts Updates internal vocabulary based on a list of texts
# texts_to_sequences Transforms each text in texts to a sequence of integers, 0 is reserved for padding

#fertig, nur get_data noch machen
def evaluate_model(model, load_file, hype_space, X_test=None, y_test=None):

    start_time = time.time()
    file_test = ['cb513'] #add more later
    test_accs = []

    for test in file_test:
        if X_test is None:
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

