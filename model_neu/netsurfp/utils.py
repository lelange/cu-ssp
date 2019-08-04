import numpy as np
import numpy.ma as ma
from numpy import array
import pandas as pd
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from keras import backend as K
import tensorflow as tf
import argparse
import telegram
import sys
import os, pickle
import random
import matplotlib.pyplot as plt
from datetime import datetime

residue_list = list('ACEDGFIHKMLNQPSRTWVYX') + ['NoSeq']
q8_list      = list('LBEGIHST') + ['NoSeq']
data_root = '/nosave/lange/cu-ssp/data/netsurfp/'
PRED_DIR = "preds/"
q8_list = list('-GHIBESTC')
q3_list = list('-HHHEECCC')

def parse_arguments(default_epochs):
    """
    :return: command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_mode', help='test mode: epochs=1 and n_folds = 1', action='store_true')
    parser.add_argument('-no_input', help='do not use input data', action='store_true')
    parser.add_argument('-optimize', help='perform hyperoptimization', action='store_true')
    parser.add_argument('-pssm', help='use pssm profiles', action='store_true')
    parser.add_argument('-hmm', help='use hmm profiles', action='store_true')
    parser.add_argument('-normalize', help='nomalize profiles', action='store_true')
    parser.add_argument('-standardize',  help='standardize profiles', action='store_true')
    parser.add_argument('-cv', help='use crossvalidation' , action= 'store_true')
    parser.add_argument('-embedding', help='embed input vectors via elmo embedder', action='store_true')
    parser.add_argument('-epochs',type=int ,required=False, help='number of training epochs', default=default_epochs)
    parser.add_argument('-tv_perc',type=float, required=False, help='ratio train validation split')
    parser.add_argument('-plot', help='plot accuracy', action='store_true')
    parser.add_argument('-predict', help='predict model', action='store_true')
    return parser.parse_args()

def normal(data):
    min = np.min(data)
    max = np.max(data)
    data_ = (data-min)/(max-min)
    return data_

def standard(data):
    mean = np.mean(data)
    std = np.std(data)
    data_ = (data - mean) / std
    return data_

# for netsurf (hmm) data
def get_data2(filename, hmm=True, normalize=False, standardize=True, embedding = False, no_input=False):

    print('Load ' + filename + ' data...')
    if embedding:
        input_seq = np.load(data_root + filename + '_netsurfp_input_embedding_residue.npy')
    else:
        if no_input:
            #input_seq = pickle.load(open(data_root + filename + '_hmm.txt', "rb")) #
            input_seq= np.load(data_root + filename + '_hmm.npy', allow_pickle=True)
            if normalize:
                input_seq = normal(input_seq)
            if standardize:
                input_seq = standard(input_seq)

        else:
            #input_seq =  pickle.load(open(data_root + filename + '_input.txt', "rb"))#
            input_seq = np.load(data_root+filename+'_input.npy', allow_pickle=True)
    #q8 = pickle.load(open(data_root + filename + '_q9.txt', "rb"))#
    q8 = np.load(data_root + filename + '_q9.npy', allow_pickle=True)
    #q3 = np.load(data_root + filename + '_q3.npy')
    if hmm:
        #profiles = pickle.load(open(data_root + filename + '_hmm.txt', "rb"))#
        profiles = np.load(data_root+filename+'_hmm.npy', allow_pickle=True)
        if normalize:
            print('Normalize...')
            profiles = normal(profiles)
        if standardize:
            print('Standardize...')
            profiles = standard(profiles)
        input_aug = [input_seq, profiles]
    else:
        input_aug = input_seq
    return input_aug, q8

def get_data(filename, hmm=True, normalize=False, standardize=True, embedding = False, no_input=False, nb_components = 500):

    print('Load ' + filename + ' data...')
    outputs=[]

    if not no_input:
        input_seq = np.load(data_root+'embedding/'+filename+'_700_input_word2vec.npy')
        print(input_seq.shape)
        #input_seq = np.load(data_root + filename + '_input_features.npy')
        #input_seq=standard(input_seq)
        #input_seq_features = np.load(data_root + filename + '_input_features.npy')
        #input_seq = np.load(data_root + filename + '_input.npy')
        #input_seq = np.concatenate((input_seq, input_seq_features), axis=2)

    else:
        input_seq = np.load(data_root + filename + '_hmm.npy')
        if normalize:
            input_seq = normal(input_seq)
        if standardize:
            input_seq = standard(input_seq)

    if hmm:
        profiles = np.load(data_root+filename+'_hmm.npy')[:,:700,:]
        if normalize:
            print('Normalize...')
            profiles = normal(profiles)
        if standardize:
            print('Standardize...')
            profiles = standard(profiles)
        input_aug = [input_seq, profiles]
    else:
        input_aug = input_seq
    outputs.append(input_aug)
    if embedding:
        embed_seq = np.load(data_root + filename + '_word2vec_3D_input.npy')
        input_aug = embed_seq
    q8 = np.load(data_root + filename + '_q9.npy')[:,:700,:]
    outputs.append(q8)

    return input_aug, q8

# for pssm+hmm data
def prepare_profiles(pssm, hmm, normalize, standardize):
    # profiles
    if pssm == True:
        print("load pssm profiles... ")
        train_pssm = np.load('../data/train_pssm.npy')
        test_pssm = np.load('../data/test_pssm.npy')

        if normalize:
            print('normalize pssm profiles...')
            train_pssm = normal(train_pssm)
            test_pssm= normal(test_pssm)

        if standardize:
            train_pssm = standard(train_pssm)
            test_pssm = standard(test_pssm)

    if hmm == True:
        print("load hmm profiles... ")
        train_hmm = np.load('../data/train_hmm.npy')
        test_hmm = np.load('../data/test_hmm.npy')

        if normalize:
            train_hmm = normal(train_hmm)
            test_hmm = normal(test_hmm)

        if standardize:
            train_hmm = standard(train_hmm)
            test_hmm = standard(test_hmm)

    if pssm and hmm:
        train_profiles = np.concatenate((train_pssm, train_hmm), axis=2)
        test_profiles = np.concatenate((test_pssm, test_hmm), axis=2)
    elif pssm:
        train_profiles = train_pssm
        test_profiles = test_pssm
    else:
        train_profiles = train_hmm
        test_profiles = test_hmm

    return train_profiles, test_profiles

# The custom accuracy metric used for this task
def accuracy(y_true, y_predicted):
    y = tf.argmax(y_true, axis =- 1)
    y_ = tf.argmax(y_predicted, axis =- 1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())


def cohens_kappa(y_true, y_pred):

    y_true_classes = tf.argmax(y_true, -1)
    y_pred_classes = tf.argmax(y_pred, -1)
    tf.reshape(y_true_classes, [-1])
    tf.reshape(y_pred_classes, [-1])
    tf.transpose(y_true_classes)
    tf.transpose(y_pred_classes)
    return tf.contrib.metrics.cohen_kappa(y_true_classes, y_pred_classes, 9)[1]

def tf_pearson(y_true, y_pred):
    return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]

def tf_f1score(y_true, y_pred):
    return tf.contrib.metrics.f1_score(y_true, y_pred)[1]

def tf_accuracy(y_true, y_pred):
    y_true_classes = tf.argmax(y_true, -1)
    y_pred_classes = tf.argmax(y_pred, -1)
    # preds and targets are switched in tf.contrib.metrics.accuracy
    return tf.contrib.metrics.accuracy(y_pred_classes, y_true_classes)

# Convert probabilities to secondary structure
def to_seq(y):
    seqs=[]
    for i in range(len(y)):
        seq_i=''
        for j in range(len(y[i])):
            seq_i += q8_list[np.argmax(y[i][j])]
        seqs.append(seq_i)
    return seqs

# Decode: map to a sequence from a one-hot
# encoding, takes a one-hot encoded y matrix
# with an lookup table "index"
# Maps the sequence to a one-hot encoding


def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        i = np.argmax(o)
        s += index[i]
    return s

def onehot_to_seq2(oh_seq, index, length=None):
    s = ''
    if length is None:
        for o in oh_seq:
            i = np.argmax(o)
            if i != 0:
                s += index[i]
            else:
                break
    else:
        for idx, o in enumerate(oh_seq):
            i = np.argmax(o[1:])
            if idx < length:
                s += index[i+1]
            else:
                break
    return s

counter = 0
# prints the results
def print_results(x, y_, revsere_decoder_index, counter,test_df, write_df=False, print_pred=False):
    if write_df:
        Ans['id'][counter] = test_df['id'][counter]
        Ans['expected'][counter] = str(onehot_to_seq(y_, revsere_decoder_index).upper())
    if print_pred:
        print("prediction: " + str(onehot_to_seq(y_, revsere_decoder_index).upper()))

# Computes and returns the n-grams of a particular sequence, defaults to trigrams
def seq2ngrams(seqs, n = 1):
    return np.array([[seq[i : i + n] for i in range(len(seq))] for seq in seqs])

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

def evaluate_acc(y_predicted):
    print('Analyse accuracy')

    order_list = [8, 5, 2, 0, 7, 6, 3, 1, 4]
    labels = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T', 'NoSeq']

    m1p = np.zeros_like(y_predicted)
    for count, i in enumerate(order_list):
        m1p[:, :, i] = y_predicted[:, :y_predicted.shape[1], count]

    summed_probs = m1p

    length_list = [len(line.strip().split(',')[2]) for line in open('cb513test_solution.csv').readlines()]
    print('max protein seq length is', np.max(length_list))

    ensemble_predictions = []
    for protein_idx, i in enumerate(length_list):
        new_pred = ''
        for j in range(i):
            new_pred += labels[np.argmax(summed_probs[protein_idx, j, :])]
        ensemble_predictions.append(new_pred)

    # calculating accuracy: compare to cb513test_solution
    gt_all = [line.strip().split(',')[3] for line in open('cb513test_solution.csv').readlines()]
    acc_list = []
    equal_counter = 0
    total = 0

    for gt, pred in zip(gt_all, ensemble_predictions):
        if len(gt) == len(pred):
            acc = get_acc(gt, pred)
            acc_list.append(acc)
            equal_counter += 1
        else:
            acc = get_acc(gt, pred)
            acc_list.append(acc)
        total += 1
    print('the accuracy is', np.mean(acc_list))
    print(str(equal_counter) + ' from ' + str(total) + ' proteins are of equal length')
    return acc_list

def weighted_accuracy(y_true, y_pred):
    return K.sum(K.equal(K.argmax(y_true, axis=-1),
                  K.argmax(y_pred, axis=-1)) * K.sum(y_true, axis=-1)) / K.sum(y_true)

def train_val_split(hmm, X_train_aug, y_train, perc = None):
    n_samples = len(y_train)
    np.random.seed(0)
    if perc is None:
        perc = 0.1
    size = int(n_samples * perc)

    validation_idx = np.random.choice(np.arange(n_samples), size=size, replace=False)
    training_idx = np.array(list(set(np.arange(n_samples)) - set(validation_idx)))

    y_val = y_train[validation_idx]
    y_train = y_train[training_idx]

    if hmm:
        '''
        X_val_aug = np.concatenate((X_train_aug[0], X_train_aug[1]), axis=2)[validation_idx]
        X_train_aug = np.concatenate((X_train_aug[0], X_train_aug[1]), axis=2)[training_idx]
        '''
        X_val_aug = [X_train_aug[0][validation_idx], X_train_aug[1][validation_idx]]
        X_train_aug = [X_train_aug[0][training_idx], X_train_aug[1][training_idx]]
    else:
        X_val_aug = X_train_aug[validation_idx]
        X_train_aug = X_train_aug[training_idx]

    return X_train_aug, y_train, X_val_aug, y_val


def telegram_me(m, s, model_name, test_acc = None, hmm=False, standardize=False, normalize = False, no_input = False, embedding=False):
    Token = "806663548:AAEJIMIBEQ9eKdyF8_JYnxUhUsDQZls1w7w"
    chat_ID = "69661085"
    bot = telegram.Bot(token=Token)
    msg = '{} ist erfolgreich durchgelaufen! :) \n\n' \
          '(Gesamtlaufzeit {:.0f}min {:.0f}s)'.format(model_name, m, s)
    if hmm:
        verb = ''
        if standardize:
            verb += 'standardisierte '
        if normalize:
            verb += 'und normalisierte '
        msg+='\nEs wurden '+verb+'HMM Profile verwendet.'
    if no_input:
        msg+='\n Es wurden nur HMM Profile als Input verwendet.'
    if embedding:
        msg+='\n Die Input-Daten wurden ins 1024-dim. eingebettet.'
    if test_acc is not None:
        for name, value in test_acc.items():
            msg += '\n'+name+' test accuracy: {:.3%}'.format(value)
    bot.send_message(chat_id=chat_ID, text=msg)

def message_me(model_name, m, s):
    username = 'charlie.gpu'
    password = '19cee1Et742'
    recipient = '100002834091853'  #Anna: 100002834091853, Chris: 100001479799294
    client = fbchat.Client(username, password)
    msg = Message(text='{} ist erfolgreich durchgelaufen! \U0001F973 '
                       '\n(Gesamtlaufzeit {:.0f}min {:.0f}s)'.format(model_name, m, s))

    sent = client.send(msg, thread_id=recipient, thread_type=ThreadType.USER)
    client.logout()

def save_cv(weights_file, cv_scores, file_scores, file_scores_mean, n_folds):
    # print results and save them to logfiles

    #calculate mean and std of cross validation results
    test_acc = {}
    for k, v in cv_scores.items():
        test_acc[k + '_mean'] = np.mean(v)
        test_acc[k + '_std'] = np.std(v)
        print('Estimated accuracy %.3f (%.3f)' % (np.mean(v) * 100, np.std(v) * 100))

    # save mean and std of cross validation results
    if not os.path.exists(file_scores_mean):
        f = open(file_scores_mean, "a+")
        f.write('### Log file for tests on ' + sys.argv[0] + ' with standardized hmm profiles. \n\n')
        f.close()

    f = open(file_scores_mean, "a+")
    i = 0
    for k, v in test_acc.items():
        if i % 2 == 0:
            f.write(str(k) + " estimated accuracy: " + "%.3f " % v)
        else:
            f.write("(%.3f)\n" % v)
        i += 1
    f.write('\n')
    f.write('Calculation on ' + str(n_folds) + ' folds.\n')
    f.write("Weights are saved to: " + weights_file + "\n")
    f.write('-----------------------\n\n')
    f.close()

    # save all history of scores used to calculate cross validation score
    if not os.path.exists(file_scores):
        f = open(file_scores, "a+")
        f.write('### Log file for tests on ' + sys.argv[0] + ' with standardized hmm profiles. \n\n')
        f.close()

    f = open(file_scores, "a+")
    for k, v in cv_scores.items():
        f.write(str(k) + ": " + str(v))
        f.write("\n")
    f.write("\n")
    f.write('Calculation on ' + str(n_folds) + ' folds.\n')
    f.write("Weights are saved to: " + weights_file + "\n")
    f.write('-----------------------\n\n')

    f.close()
    return test_acc


def get_acc(gt, pred, mask=None):
    '''
         if len(gt)!=len(pred):
        print("Lengths are not equal. Len true = "+str(len(gt))+" len pred = "+str(len(pred)))
    '''
    correct = 0
    for i in range(len(gt)):
        if mask is not None:
            if mask[i] == 1:
                if gt[i] == pred[i]:
                    correct += 1
            length=np.sum(mask)

        else:
            try:
                if gt[i] == pred[i]:
                    correct += 1
            except:
                print(i)
            length = len(gt)
    return (1.0 * correct), length

def get_acc2(gt, pred, mask = None):
    '''
         if len(gt)!=len(pred):
        print("Lengths are not equal. Len true = "+str(len(gt))+" len pred = "+str(len(pred)))
    '''
    correct = 0
    for i in range(len(gt)):
        if mask is not None:
            if mask[i] == 1:
                if gt[i] == pred[i]:
                    correct += 1
            length=np.sum(mask)

        else:
            if gt[i] == pred[i]:
                correct += 1

            length = len(gt)

    return (1.0 * correct)/length

#name2label = {j:i  for i,j in enumerate(labels[:-1])}

def get_confusion_matrix(true_q, pred_q, model_name, labels_full=q8_list[1:], save=True):
    labels=np.unique(labels_full)
    labels = list(labels)
    classes = len(labels)
    conf_matrix = np.zeros((classes, classes))
    for gt, pred in zip(true_q, pred_q):
        if len(gt) < len(pred):
            pred = pred[:len(gt)]
        for g, p in zip(gt, pred):
            if p == '-' or g == '-':
                print(gt)
                print(pred)
            else:
                conf_matrix[labels.index(p), labels.index(g)] += 1
    conf = pd.DataFrame(conf_matrix, columns=labels, index=labels)
    print(conf)

    recall_list = np.zeros((classes,))
    precision_list = np.zeros((classes,))
    for i in range(classes):
        if np.sum(conf_matrix[:, i]) != 0:
            recall_list[i] = conf_matrix[i, i] / float(np.sum(conf_matrix[:, i]))
        else:
            recall_list[i] = 0

        if np.sum(conf_matrix[i, :]) != 0:
            precision_list[i] = conf_matrix[i, i] / float(np.sum(conf_matrix[i, :]))
        else:
            precision_list[i] = 0

    fscore = np.zeros((classes,))

    for i in range(classes):
        if precision_list[i] + recall_list[i] != 0:
            fscore[i] = 2.0 * precision_list[i] * recall_list[i] / (precision_list[i] + recall_list[i])
        else:
            fscore[i] = 0

    for i in range(classes):
        print('precision of class', labels[i], precision_list[i])
    print('average precision', np.mean(precision_list))

    for i in range(classes):
        print('recall of class', labels[i], recall_list[i])
    print('average recall', np.mean(recall_list))

    for i in range(classes):
        print('fscore of class', labels[i], fscore[i])
    print('average f-score', np.mean(fscore))

    if save:
        with open('logs/metrics_cb513_' + model_name + '.csv', 'a+') as f:
            conf.to_csv(f)
            f.write('\n')
            f.write('secondary structure, precision, recall, f-score' + '\n')
            for i in range(classes):
                f.write(str(labels[i]) + ',' + str(precision_list[i]) + ',' + str(recall_list[i]) + ',' + str(
                    fscore[i]) + '\n')

            f.write('mean,' + str(np.mean(precision_list)) + ',' + str(np.mean(recall_list)) + ',' + str(
                np.mean(fscore)) + '\n')
            f.write('\n')
    return conf

def build_and_predict(model, best_weights, save_pred_file, model_name, file_test=['cb513_700'], save_eval=True):
    if model is None:
        model = build_model()
    if model_name == "mod_2":
        file_test = ['cb513_768']  ### uncomment later!!
    else:
        file_test = ['cb513_full'] ### uncomment later!!

    input_mask = np.load(data_root+'cb513_700_seq_range.npy')
    print(len(input_mask))
    print(input_mask)

    # save all accuracys from Q8 and Q3 preditions
    f = open(PRED_DIR + "prediction_accuracy.txt", "a+")
    for test in file_test:

        i = True
        X_test_aug, y_test = get_data(test, hmm=True, normalize=False, standardize=True)
        X_test_aug = X_test_aug[input_mask]
        y_test = y_test[input_mask]
        print(X_test_aug[0].shape)
        print(X_test_aug[1].shape)
        print(y_test.shape)
        print(best_weights)
        model.load_weights(best_weights)

        print("\nPredict " + test + "...")

        y_test_pred = model.predict(X_test_aug)
        score = model.evaluate(X_test_aug, y_test)
        print("Accuracy from model evaluate: " + str(score[2]))
        np.save(PRED_DIR +'Q8/' + test + save_pred_file, y_test_pred)
        print("Saved predictions to " + PRED_DIR + 'Q8/' + test + save_pred_file + ".")

        '''
        sess = tf.Session()
        with sess.as_default():
            acc = accuracy2(y_test, y_test_pred)
            print("Accuracy2: ")
            print(acc.eval()[:30])
            print(np.sum(acc.eval()))
            print(len(acc.eval()))
            print(np.sum(acc.eval())/len(acc.eval()))
            print("Test argmax (len 5, max at 3): "+str(tf.argmax(input=[2,0,1,0,0]).eval()))
            print("Test argmax (len 2): " + str(tf.argmax(input=[0]).eval()))

        '''
        q3_pred = 0
        q8_pred = 0
        q3_len = 0
        q8_len = 0

        q8_accs=[]
        q3_accs=[]

        q3_pred_mask = 0
        q8_pred_mask = 0
        q3_len_mask = 0
        q8_len_mask = 0

        q8_accs_mask = []
        q3_accs_mask = []

        pred_q3 = []
        pred_q8 = []
        true_q3 = []
        true_q8 = []

        g = open(PRED_DIR +'Q8/' +"q9_pred_mod_1.txt", "w+")
        h = open(PRED_DIR +'Q3/'+ "q4_pred_mod_1.txt", "w+")


        if test == 'cb513_700':
            mask = np.load(data_root+"cb513_700_evaluation_mask.npy")
        else:
            mask = [None]*len(y_test_pred)

        #calculate q8, q3 representations from one hot encoding and calculate accuracy
        for m, true, pred in zip(mask, y_test, y_test_pred):
            seq3 = onehot_to_seq(pred, q3_list)
            seq8 = onehot_to_seq(pred, q8_list)
            seq_true_3 = onehot_to_seq2(true, q3_list)
            seq_true_8 = onehot_to_seq2(true, q8_list)
            pred_q3.append(seq3)
            pred_q8.append(seq8)
            true_q3.append(seq_true_3)
            true_q8.append(seq_true_8)

            '''
            h.write(seq3)
            g.write(seq8)
            h.write("\n")
            g.write("\n")
            '''

            corr3, len3 = get_acc(seq_true_3, seq3)
            corr8, len8 = get_acc(seq_true_8, seq8)
            q8_accs.append(get_acc2(seq_true_8, seq8))
            q3_accs.append(get_acc2(seq_true_3, seq3))
            q3_pred += corr3
            q8_pred += corr8
            q3_len += len3
            q8_len += len8

            corr3, len3 = get_acc(seq_true_3, seq3,m)
            corr8, len8 = get_acc(seq_true_8, seq8,m)
            q8_accs_mask.append(get_acc2(seq_true_8, seq8,m))
            q3_accs_mask.append(get_acc2(seq_true_3, seq3,m))
            q3_pred_mask += corr3
            q8_pred_mask += corr8
            q3_len_mask += len3
            q8_len_mask += len8

        g.close()
        h.close()

        print('Saved Q8 sequences to '+PRED_DIR +'Q8/' +"q9_pred_mod_1.txt")
        print('Saved Q3 sequences to ' + PRED_DIR + 'Q3/' + "q4_pred_mod_1.txt")

        #print results
        print("Accuracy #sum(correct per proteins)/#sum(len_proteins):")
        print("Q3 " + test + " test accuracy: " + str(q3_pred / q3_len))
        print("Q8 " + test + " test accuracy: " + str(q8_pred / q8_len))
        print("\nAccuracy mean(#correct per protein/#len_protein):")
        print("Q3 " + test + " test accuracy: " + str(np.nanmean(q3_accs)))
        print("Q8 " + test + " test accuracy: " + str(np.nanmean(q8_accs)))

        print('_______________________')
        print('_______________________')

        print("best Q3 and true:"+str(np.max(q3_accs)))
        print(pred_q3[np.argmax(q3_accs)])
        print(true_q3[np.argmax(q3_accs)])

        print("best Q8 and true:"+str(np.max(q8_accs)))
        print(pred_q8[np.argmax(q8_accs)])
        print(true_q8[np.argmax(q8_accs)])

        print('_______________________')

        print("worst Q3 and true:" + str(np.min(q3_accs)))
        print(pred_q3[np.argmin(q3_accs)])
        print(true_q3[np.argmin(q3_accs)])
        mst = ""
        if not mask[0] is None:
            for i in mask[np.argmin(q3_accs)]:
                mst += str(int(i))

        print(mst[:len(pred_q3[np.argmin(q3_accs)])])

        print('_______________________')

        print("worst Q8 and true:" + str(np.min(q8_accs)))
        print(pred_q8[np.argmin(q8_accs)])
        print(true_q8[np.argmin(q8_accs)])
        print(mst[:len(pred_q8[np.argmin(q8_accs)])])

        print("all under 50% or 70% accuracy:")
        bads_q3 = np.argwhere(np.array(q3_accs)<0.7)
        bads_q8 = np.argwhere(np.array(q8_accs) < 0.5)
        print("Q3: "+str(len(bads_q3)))
        for i in bads_q3:
            i = int(i)
            print("Accuracy: "+str(q3_accs[i]))
            print(pred_q3[i])
            print(true_q3[i])

        print('_______________________')

        print("Q8: " + str(len(bads_q8)))
        for i in bads_q8:
            i = int(i)
            print("Accuracy: "+str(q8_accs[i]))
            print(pred_q8[i])
            print(true_q8[i])


        get_confusion_matrix(true_q8, pred_q8, model_name, q8_list[1:] ,save_eval)
        get_confusion_matrix(true_q3, pred_q3, model_name, q3_list[1:], save_eval)


        plt.hist(q3_accs, label='Q3', alpha = 0.5)
        plt.hist(q8_accs, label='Q8', alpha = 0.5)
        plt.legend()
        plt.show()
        plt.savefig('./plots/'+model_name+'_'+datetime.now().strftime("%m_%d-%H_%M") + '_Q3Q8_accuracy_hist.png')
        plt.clf()

        plt.plot(q3_accs, label='Q3', alpha=0.5)
        plt.plot(q8_accs, label='Q8', alpha=0.5)
        plt.legend()
        plt.show()
        plt.savefig('./plots/' + model_name+'_' + datetime.now().strftime("%m_%d-%H_%M") + '_Q3Q8_accuracy_plot.png')
        plt.clf()

        fig, ax = plt.subplots()
        plotdata = [q3_accs, q8_accs]
        ax.boxplot(plotdata)
        ax.set_xticklabels(['q3', 'q8'])
        plt.show()
        plt.savefig('./plots/' + model_name+'_' + datetime.now().strftime("%m_%d-%H_%M") + '_Q3Q8_accuracy_boxplot.png')



        if test == 'cb513_700':
            print('MASKED RESULTS:')
            print("Accuracy #sum(correct per proteins)/#sum(len_proteins):")
            print(q3_len_mask)
            print("Q3 " + test + " test accuracy: " + str(q3_pred_mask / q3_len_mask))
            print("Q8 " + test + " test accuracy: " + str(q8_pred_mask / q8_len_mask))
            print("\nAccuracy mean(#correct per protein/#len_protein):")
            print("Q3 " + test + " test accuracy: " + str(np.nanmean(q3_accs_mask)))
            print("Q8 " + test + " test accuracy: " + str(np.nanmean(q8_accs_mask)))



    '''
        #save results to file
        f.write("Results for " + model_name + " and weights " + best_weights+" on "+test+".")
        f.write("\n\n")
        f.write("Netsurf data were used with standardized hhblits profiles.\n")
        f.write("Accuracy #sum(correct per proteins)/#sum(len_proteins):\n")
        f.write("Q3 " + test + " test accuracy: " + str(q3_pred / q3_len))
        f.write("\n")
        f.write("Q8 " + test + " test accuracy: " + str(q8_pred / q8_len))
        f.write("\n\n")

        f.write("Accuracy mean(#correct per protein/#len_protein):\n")
        f.write("Q3 " + test + " test accuracy: " + str(np.mean(q3_accs)))
        f.write("\n")
        f.write("Q8 " + test + " test accuracy: " + str(np.mean(q8_accs)))
        f.write("\n\n")

        f.write("Accuracy from model evaluate: " + str(score[2]))
        f.write("\n\n")

        f.write("Predictions are saved to: " + PRED_DIR + test + save_pred_file)
        f.write("\n----------------------------\n\n")
        
        '''
    f.write("----------------------------\n\n\n")
    f.close()

def save_results_to_file(time_end, model_name, weights_file, test_acc, hmm=True, standardize=True, normalize=False, no_input=False, embedding=False):
    f = open("results_experiments.txt", "a+")
    f.write("Results for " + model_name + " and weights " + weights_file +".")
    f.write("\n\n")

    m, s = divmod(time_end, 60)
    msg = 'Runtime: {:.0f}min {:.0f}s'.format(m, s)
    if embedding:
        "Input values were split to 3-grams and each 3-gram has been embedded in 100 dim with word2 vec.\n " \
        "After that, the input dimension was reduced from (#samples, 70000) to (#samples, 500) with UMAP."
    if hmm:
        verb = ''
        if standardize:
            verb += 'Standardized '
        if normalize:
            verb += 'and normalized '
        msg += '\n' + verb + 'hhblits profiles and netsurf data were used.'
    if no_input:
        msg += '\n Only hhblits profiles were used as input.'
    if test_acc is not None:
        for name, value in test_acc.items():
            msg += '\n' + name + ' test accuracy: {:.3%}'.format(value)

    f.write(msg)
    f.write("\n")
    f.write("----------------------------\n")
    f.write("\n\n")
    f.close()

def evaluate_model(model, load_file, test_ind = None, hmm=True,
                   normalize=False, standardize=True, file_test=None):
    if file_test is None:
        file_test = ['cb513_700', 'ts115_700', 'casp12_700']
    if test_ind is None:
        test_ind = range(len(file_test))
    test_accs = []
    names = []
    for i in test_ind:
        X_test_aug, y_test = get_data(file_test[i], hmm, normalize, standardize)
        model.load_weights(load_file)
        print("####evaluate " + file_test[i] +":")
        score = model.evaluate(X_test_aug, y_test, verbose=2, batch_size=1)
        print(file_test[i] +' test loss:', score[0])
        print(file_test[i] +' test accuracy:', score[2])
        test_accs.append(score[2])
        names.append(file_test[i])

    replace = 'mod_'
    withstring = str(test_accs[0])[2:8]+'_mod_'
    newstr, found, endpart = load_file.partition(replace)

    if found:
        newstr += withstring + endpart
        print(newstr)
    else:
        print("%r string is not in %r" % (replace, load_file))

    os.rename(load_file, newstr)
    print('Weights are now saved to '+newstr)
    return dict(zip(names, test_accs))



