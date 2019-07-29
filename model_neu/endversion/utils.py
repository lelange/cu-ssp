import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from keras import backend as K
import tensorflow as tf
import argparse
import telegram
from datetime import datetime
import os, pickle
from collections import defaultdict

residue_list = list('ACEDGFIHKMLNQPSRTWVYX') + ['NoSeq']
q8_list      = list('LBEGIHST') + ['NoSeq']

data_root = '/nosave/lange/cu-ssp/data/'
qzlsy_root = '/nosave/lange/cu-ssp/data/data_qzlshy/'
netsurfp_root = '/nosave/lange/cu-ssp/data/netsurfp/'
princeton_root = '/nosave/lange/cu-ssp/data/data_princeton/'


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

def get_data(filename, primary=True, hmm=True, pssm=False, normalize=False, standardize=True, embedding = False):
    #input (primary and/or embedding)

    if princeton:

    if qzlsy:

    if netsurf:


    #profiles (pssm and/or hmm)

    return

def get_qzlshy_profiles(name, pssm, hmm, normalize, standardize):
    # profiles can be use for train and test profiles
    if not name in ['train', 'test']:
        print("Mode should be either train or test.")

    filename=data_root+name

    if pssm == True:
        print("load pssm profiles... ")
        train_pssm = np.load(filename+'_pssm.npy')

        if normalize:
            print('normalize pssm profiles...')
            train_pssm = normal(train_pssm)

        if standardize:
            print('standardize pssm profiles...')
            train_pssm = standard(train_pssm)

    if hmm == True:
        print("load hmm profiles... ")
        train_hmm = np.load(filename+'_hmm.npy')

        if normalize:
            train_hmm = normal(train_hmm)

        if standardize:
            train_hmm = standard(train_hmm)

    if pssm and hmm:
        train_profiles = np.concatenate((train_pssm, train_hmm), axis=2)
    elif pssm:
        train_profiles = train_pssm
    else:
        train_profiles = train_hmm


    return train_profiles


# The custom accuracy metric used for this task
def accuracy(y_true, y_predicted):
    y = tf.argmax(y_true, axis =- 1)
    y_ = tf.argmax(y_predicted, axis =- 1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())

def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        i = np.argmax(o)
        s += index[i]
    return s

def get_acc(gt, pred):
    '''
         if len(gt)!=len(pred):
        print("Lengths are not equal. Len true = "+str(len(gt))+" len pred = "+str(len(pred)))
    '''
    correct = 0
    for i in range(len(gt)):
        if gt[i] == pred[i]:
            correct += 1
    return (1.0 * correct), len(gt)

def get_acc2(gt, pred):
    '''
         if len(gt)!=len(pred):
        print("Lengths are not equal. Len true = "+str(len(gt))+" len pred = "+str(len(pred)))
    '''
    correct = 0
    for i in range(len(gt)):
        if gt[i] == pred[i]:
            correct += 1
    return (1.0 * correct)/len(gt)

def weighted_accuracy(y_true, y_pred):
    return K.sum(K.equal(K.argmax(y_true, axis=-1),
                  K.argmax(y_pred, axis=-1)) * K.sum(y_true, axis=-1)) / K.sum(y_true)

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


def build_and_predict(model, best_weights, save_pred_file, model_name, file_test=['ts115_700']):
    if model is None:
        model = build_model()

    # save all accuracys from Q8 and Q3 preditions
    f = open(PRED_DIR + "prediction_accuracy.txt", "a+")
    for test in file_test:

        i = True
        X_test_aug, y_test = get_data(test, hmm=True, normalize=False, standardize=True)
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

        g = open(PRED_DIR +'Q8/' +"q9_pred_mod_1.txt", "w+")
        h = open(PRED_DIR +'Q3/'+ "q4_pred_mod_1.txt", "w+")

        #calculate q8, q3 representations from one hot encoding and calculate accuracy
        for true, pred in zip(y_test, y_test_pred):
            seq3 = onehot_to_seq(pred, q3_list)
            seq8 = onehot_to_seq(pred, q8_list)
            seq_true_3 = onehot_to_seq2(true, q3_list)
            seq_true_8 = onehot_to_seq2(true, q8_list)

            if i:
                print('Q3 prediction, first pred then true: ')
                print(seq3[:60])
                print(seq_true_3[:60])

                print('Q8 prediction, first pred then true: ')
                print(seq8[:60])
                print(seq_true_8[:60])

                i = False

            h.write(seq3)
            g.write(seq8)
            h.write("\n")
            g.write("\n")

            corr3, len3 = get_acc(seq_true_3, seq3)
            corr8, len8 = get_acc(seq_true_8, seq8)
            q8_accs.append(get_acc2(seq_true_8, seq8))
            q3_accs.append(get_acc2(seq_true_3, seq3))
            q3_pred += corr3
            q8_pred += corr8
            q3_len += len3
            q8_len += len8
        g.close()
        h.close()
        print('Saved Q8 sequences to '+PRED_DIR +'Q8/' +"q9_pred_mod_1.txt")
        print('Saved Q3 sequences to ' + PRED_DIR + 'Q3/' + "q4_pred_mod_1.txt")

        #print results
        print("Accuracy #sum(correct per proteins)/#sum(len_proteins):")
        print("Q3 " + test + " test accuracy: " + str(q3_pred / q3_len))
        print("Q8 " + test + " test accuracy: " + str(q8_pred / q8_len))
        print("\nAccuracy mean(#correct per protein/#len_protein):")
        print("Q3 " + test + " test accuracy: " + str(np.mean(q3_accs)))
        print("Q8 " + test + " test accuracy: " + str(np.mean(q8_accs)))

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


def build_and_train (model, batch_size, epochs,
                     X_train_aug, y_train, X_val_aug, y_val,
                     load_file, callbacks = None, patience=None):

    if patience is None:
        patience = np.max(int(epochs/10), 2)

    if callbacks is None:
        callbacks = []

        earlyStopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max')
        checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose=1, save_best_only=True,
                                       mode='max')

        callbacks.append(earlyStopping)
        callbacks.append(checkpointer)

    model = model

    # Training the model on the training data and validating using the validation set
    history = model.fit(X_train_aug, y_train, validation_data=(X_val_aug, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                        verbose=1, shuffle=True)

    return model, history


def evaluate_model(model, load_file, file_test):
    test_accs = []
    names = []
    for test in file_test:
        X_test_aug, y_test = get_data(test, hmm, normalize, standardize)
        model.load_weights(load_file)
        print("\nevaluate " + test +":")
        score = model.evaluate(X_test_aug, y_test, verbose=2, batch_size=1)
        print(test +' test loss:', score[0])
        print(test +' test accuracy:', score[2])
        test_accs.append(score[2])
        names.append(test)
    return dict(zip(names, test_accs))

def crossValidation(model, batch_size, epochs, load_file,
                    X_train_aug, y_train,
                    n_folds, file_test):

    X_train, X_aug_train = X_train_aug
    # Instantiate the cross validator
    kfold_splits = n_folds
    kf = KFold(n_splits=kfold_splits, shuffle=True)

    cv_scores = defaultdict(list)
    model_history = []

    # Loop through the indices the split() method returns
    for index, (train_indices, val_indices) in enumerate(kf.split(X_train, y_train)):
        print('\n\n-----------------------')
        print("Training on fold " + str(index + 1) + "/" + str(kfold_splits) +"...")
        print('-----------------------\n')

        # Generate batches from indices
        X_train_fold, X_val_fold = X_train[train_indices], X_train[val_indices]
        X_aug_train_fold, X_aug_val_fold = X_aug_train[train_indices], X_aug_train[val_indices]
        y_train_fold, y_val_fold = y_train[train_indices], y_train[val_indices]

        print("Training new iteration on " + str(X_train_fold.shape[0]) + " training samples, " + str(
            X_val_fold.shape[0]) + " validation samples...")

        model, history = build_and_train(model, batch_size, epochs,[X_train_fold, X_aug_train_fold], y_train_fold,
                                  [X_val_fold, X_aug_val_fold], y_val_fold, load_file)

        print(history.history)

        test_acc = evaluate_model(model, load_file, file_test)

        cv_scores['val_accuracy'].append(max(history.history['val_accuracy']))

        for k, v in test_acc.items():
            cv_scores[k].append(v)

        model_history.append(model)

    return cv_scores, model_history