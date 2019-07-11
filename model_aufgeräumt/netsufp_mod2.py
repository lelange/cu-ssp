import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
import tensorflow as tf
from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import sys

import time
import dill as pickle
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval

from utils import *

start_time = time.time()

args = parse_arguments(default_epochs=80)

normalize = args.normalize
standardize = args.standardize
hmm = args.hmm
embedding = args.embedding
epochs = args.epochs
plot = args.plot
no_input = args.no_input
optimize = args.optimize
cross_validate = args.cv
tv_perc = args.tv_perc

batch_size = 128

n_tags = 8
n_words = 20
data_root = '../data/netsurfp/'

file_train = 'train_608'
file_test = ['cb513_608', 'ts115_608', 'casp12_608']

#load data
X_train_aug, y_train = get_data(file_train, hmm, normalize, standardize)

if hmm:
    print("X train shape: ", X_train_aug[0].shape)
    print("X aug train shape: ", X_train_aug[1].shape)
else:
    print("X train shape: ", X_train_aug.shape)
print("y train shape: ", y_train.shape)

time_data = time.time() - start_time

# Dropout to prevent overfitting.
droprate = 0.3

#### model

def conv_block(x, n_channels, droprate):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(n_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Dropout(droprate)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(n_channels, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    return x

def up_block(x, n_channels):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling1D(size = 2)(x)
    x = Conv1D(n_channels, 2, padding = 'same', kernel_initializer = 'he_normal')(x)
    return x

def evaluate_model(model, load_file, test_ind = None):
    if test_ind is None:
        test_ind = range(len(file_test))
    test_accs = []
    names = []
    for i in test_ind:
        X_test_aug, y_test = get_data(file_test[i], hmm, normalize, standardize)
        model.load_weights(load_file)
        print("####evaluate" + file_test[i] +":")
        score = model.evaluate(X_test_aug, y_test, verbose=2, batch_size=1)
        print(file_test[i] +' test loss:', score[0])
        print(file_test[i] +' test accuracy:', score[2])
        test_accs.append(score[2])
        names.append(file_test[i])
    return dict(zip(names, test_accs))


def build_model():
    model = None

    if hmm:
        input = Input(shape=(X_train_aug[0].shape[1], X_train_aug[0].shape[2],))
        profiles_input = Input(shape=(X_train_aug[1].shape[1], X_train_aug[1].shape[2],))
        merged_input = concatenate([input, profiles_input])
        inp = [input, profiles_input]
    else:
        input = Input(shape=(X_train_aug.shape[1], X_train_aug.shape[2],))
        merged_input = input
        inp = input

    merged_input = Conv1D(128, 3, padding='same', kernel_initializer='he_normal')(merged_input)

    conv1 = conv_block(merged_input, 128, droprate)
    pool1 = MaxPooling1D(pool_size=2)(conv1)


    conv2 = conv_block(pool1, 152, droprate)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = conv_block(pool2, 304, droprate)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = conv_block(pool3, 608, droprate)
    pool4 = MaxPooling1D(pool_size=2)(conv4)

    conv5 = conv_block(pool4, 1212, droprate)

    print('Conv1: ', conv1.shape)
    print('pool1: ', pool1.shape)
    print('Conv2: ', conv2.shape)
    print('pool2: ', pool2.shape)
    print('Conv3: ', conv3.shape)
    print('pool3: ', pool3.shape)
    print('Conv4: ', conv4.shape)
    print('pool4: ', pool4.shape)
    print('Conv5: ', conv5.shape)

    up4 = up_block(conv5, 608)
    up4 = concatenate([conv4, up4])
    up4 = conv_block(up4, 608, droprate)

    up3 = up_block(up4, 304)
    up3 = concatenate([conv3, up3])
    up3 = conv_block(up3, 304, droprate)

    up2 = up_block(up3, 152)
    up2 = concatenate([conv2, up2])
    up2 = conv_block(up2, 152, droprate)

    up1 = up_block(up2, 128)
    up1 = concatenate([conv1, up1])
    up1 = conv_block(up1, 128, droprate)

    up1 = BatchNormalization()(up1)
    up1 = ReLU()(up1)

    # the following it equivalent to Conv1D with kernel size 1
    # A dense layer to output from the LSTM's64 units to the appropriate number of tags to be fed into the decoder
    y = TimeDistributed(Dense(n_tags, activation="softmax"))(up1)

    # Defining the model as a whole and printing the summary
    model = Model(inp, y)
    # Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy

    optim = RMSprop(lr=0.002)

    model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=["accuracy", accuracy])
    model.summary()

    return model

load_file = "./model/mod_2-CB513-" + datetime.now().strftime("%Y_%m_%d-%H_%M") + ".h5"

def train_model(X_train_aug, y_train, X_val_aug, y_val, epochs = epochs):
    model = build_model()

    ####callbacks for fitting
    def scheduler(i, lr):
        if i in [60]:
            return lr * 0.5
        return lr
    reduce_lr = LearningRateScheduler(schedule=scheduler, verbose=1)
    # reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
    #                             patience=8, min_lr=0.0005, verbose=1)


    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=15, verbose=1, mode='max')
    checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose=1, save_best_only=True,
                                   mode='max')
    # Training the model on the training data and validating using the validation set
    history = model.fit(X_train_aug, y_train, validation_data=(X_val_aug, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, reduce_lr, earlyStopping],
                        verbose=1, shuffle=True)



    evaluate_model(model, load_file)

    model.load_weights(load_file)

    evaluate_model(model, load_file)

    # plot accuracy during training

    return model

####save predictions
#y_pre = model.predict([X_test, X_aug_test])

if cross_validate :
    cv_scores, model_history = crossValidation(load_file, X_train_aug, y_train)
    test_acc = np.mean(cv_scores)
    print('Estimated accuracy %.3f (%.3f)' % (test_acc, np.std(cv_scores)))
else:
    X_train_aug, y_train, X_val_aug, y_val = train_val_split(hmm, X_train_aug, y_train, tv_perc)

    if optimize:

        '''
        #---- create a Trials database to store experiment results
        trials = Trials()
        #---- use that Trials database for fmin
        best = fmin(build_model_ho, space, algo=tpe.suggest, trials=trials, max_evals=100, rstate=np.random.RandomState(99))
        #---- save trials
        pickle.dump(trials, open("./trials/mod_3-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+"-hyperopt.p", "wb"))
        #trials = pickle.load(open("./trials/mod_3-CB513-"+datetime.now().strftime("%Y_%m_%d-%H_%M")+"-hyperopt.p", "rb"))
        print('Space evaluation: ')
        space_eval(space, best)
        data = list(map(_flatten, extract_params(trials)))
        df = pd.DataFrame(list(data))
        df = df.fillna(0)  # missing values occur when the object is not populated
        corr = df.corr()
        print(corr)
        '''

    else:
        model = train_model(X_train_aug, y_train, X_val_aug, y_val, epochs=epochs)
        test_acc = evaluate_model(model, load_file)

time_end = time.time() - start_time
m, s = divmod(time_end, 60)
print("The program needed {:.0f}s to load the data and {:.0f}min {:.0f}s in total.".format(time_data, m, s))

telegram_me(m, s, sys.argv[0], test_acc, hmm, standardize, normalize, no_input)

