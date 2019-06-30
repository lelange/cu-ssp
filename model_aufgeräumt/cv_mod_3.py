import sys
import os
import argparse
import time
import numpy as np
import dill as pickle
import pandas as pd
import tensorflow as tf
sys.path.append('keras-tcn')
from tcn import tcn
import h5py

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from keras import backend as K
from keras import regularizers, constraints, initializers, activations

from keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, CuDNNGRU
from keras.layers import Dropout, Flatten, Activation, RepeatVector, Permute

from keras.layers import Dropout
from keras.layers import merge
from keras.layers.merge import concatenate
from keras.layers.recurrent import Recurrent
from keras.metrics import categorical_accuracy
from keras.models import Model, Input, Sequential
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import fbchat
from fbchat.models import *
import emoji
from utils import *

# Instantiate the cross validator
kfold_splits = 5
kf = KFold(n_splits=kfold_splits, shuffle=True)
cv_scores, model_history = list(), list()

# Loop through the indices the split() method returns
for index, (train_indices, val_indices) in enumerate(kf.split(X_train, y_train)):

    print ("Training on fold " + str(index + 1) + "/10...")
    # Generate batches from indices
    X_train_fold, X_val_fold = X_train[train_indices], X_train[val_indices]
    X_aug_train_fold, X_aug_val_fold = X_aug_train[train_indices], X_aug_train[val_indices]
    y_train_fold, y_val_fold = y_train[train_indices], y_train[val_indices]

    print ("Training new iteration on " + str(X_train_fold.shape[0]) + " training samples, " + str(X_val_fold.shape[0]) + " validation samples...")

    X_train_aug = [X_train_fold, X_aug_train_fold]
    X_val_aug = [X_val_fold, X_aug_val_fold]
    X_test_aug = [X_test, X_aug_test]

    model, test_acc = train_model(X_train_aug, y_train, X_val_aug, y_val, X_test_aug, y_test)
    print('>%.3f' % test_acc)
    cv_scores.append(test_acc)
    model_history.append(model)

print('Estimated accuracy %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))