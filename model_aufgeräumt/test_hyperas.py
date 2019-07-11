from hyperas import optim
from hyperas.distributions import quniform, uniform
from hyperopt import STATUS_OK, tpe, mongoexp

import numpy as np

from utils import *
from keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, CuDNNGRU
from keras.layers import Dropout, Flatten, Activation, RepeatVector, Permute

from keras.layers.merge import concatenate

from keras.metrics import categorical_accuracy
from keras.models import Model, Input, Sequential
from keras.optimizers import Adam

import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.datasets import mnist
import tempfile

def data():
    data_root = '~/../../nosave/lange/cu-ssp/data/netsurfp/'
    file_train = 'train'
    file_test = ['cb513', 'ts115', 'casp12']

    X_test = np.load(data_root + file_test[0] + '_input.npy')
    profiles = np.load(data_root + file_test[0] + '_hmm.npy')
    mean = np.mean(profiles)
    std = np.std(profiles)
    X_aug_test = (profiles - mean) / std
    X_test_aug = [X_test, X_aug_test]
    y_test = np.load(data_root + file_test[0] + '_q8.npy')

    X_train = np.load(data_root + file_train + '_input.npy')
    profiles = np.load(data_root + file_train + '_hmm.npy')
    mean = np.mean(profiles)
    std = np.std(profiles)
    X_aug_train = (profiles - mean) / std
    X_train_aug = [X_train, X_aug_train]
    y_train = np.load(data_root + file_train + '_q8.npy')

    X_train_aug, y_train, X_val_aug, y_val = train_val_split(True, X_train_aug, y_train)

    return X_train_aug, y_train, X_val_aug, y_val, X_test_aug, y_test


def create_model(X_train_aug, y_train, X_val_aug, y_val, X_test_aug, y_test):
    """
    Create your model...
    """
    DROPOUT_CHOICES = np.arange(0.0, 0.9, 0.1)
    UNIT_CHOICES = [100, 200, 500, 800, 1000, 1200]
    GRU_CHOICES = [100, 200, 300, 400, 500, 600]
    BATCH_CHOICES = [16, 32]
    LR_CHOICES = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01]
    params = {
        'dense1': hp.choice('dense1', UNIT_CHOICES),
        'dropout1': hp.choice('dropout1', DROPOUT_CHOICES),
        'gru1': hp.choice('gru1', GRU_CHOICES),
        # nesting the layers ensures they're only un-rolled sequentially
        'gru2': hp.choice('gru2', [False, {
            'gru2_units': hp.choice('gru2_units', GRU_CHOICES),
            # only make the 3rd layer availabile if the 2nd one is
            'gru3': hp.choice('gru3', [False, {
                'gru3_units': hp.choice('gru3_units', GRU_CHOICES)
            }]),
        }]),
        'dense2': hp.choice('dense2', UNIT_CHOICES),
        'dropout2': hp.choice('dropout2', DROPOUT_CHOICES),
        'lr': hp.choice('lr', LR_CHOICES),
        'decay': hp.choice('decay', LR_CHOICES),
        'batch_size': hp.choice('batch_size', BATCH_CHOICES)
    }

    input = Input(shape=(X_train_aug[0].shape[1], X_train_aug[0].shape[2],))
    profiles_input = Input(shape=(X_train_aug[1].shape[1], X_train_aug[1].shape[2],))
    x1 = Dense(params['dense1'], activation="relu")(x1)
    x1 = Dropout(params['dropout1'])(x1)
    # x1 = Bidirectional(CuDNNGRU(units=100, return_sequences=True))(x1)
    # Defining a bidirectional LSTM using the embedded representation of the inputs
    x2 = Bidirectional(CuDNNGRU(units=params['gru1'], return_sequences=True))(x2)
    # x2 = Dropout(0.5)(x2)
    if params['gru2']:
        x2 = Bidirectional(CuDNNGRU(units=params['gru2']['gru2_units'], return_sequences=True))(x2)
    if params['gru2'] and params['gru2']['gru3']:
        x2 = Bidirectional(CuDNNGRU(units=params['gru2']['gru3']['gru3_units'], return_sequences=True))(x2)
    # x2 = Dropout(0.5)(x2)
    COMBO_MOVE = concatenate([x1, x2])
    w = Dense(params['dense2'], activation="relu")(COMBO_MOVE)  # try 500
    w = Dropout(params['dropout2'])(w)
    w = tcn.TCN(return_sequences=True)(w)
    y = TimeDistributed(Dense(8, activation="softmax"))(w)
    model = Model([input, profiles_input], y)

    adamOptimizer = Adam(lr=params['lr'], beta_1=0.8, beta_2=0.8, epsilon=None, decay=params['decay'], amsgrad=False)
    model.compile(optimizer=adamOptimizer, loss="categorical_crossentropy", metrics=["accuracy", accuracy])

    #earlyStopping = EarlyStopping(monitor='val_accuracy', patience=3, verbose=verbose, mode='max')
    #checkpointer = ModelCheckpoint(filepath=load_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    history = model.fit(X_train_aug, y_train, validation_data=(X_val_aug, y_val),
                        epochs=20, batch_size=params['batch_size'],
                        verbose=1, shuffle=True)

    score = model.evaluate(X_test_aug, y_test)

    out = {
        'loss': -score[2],
        'score': score[0],
        'status': STATUS_OK,
        'model_params': params,
    }
    # optionally store a dump of your model here so you can get it from the database later
    temp_name = tempfile.gettempdir()+'/'+next(tempfile._get_candidate_names()) + '.h5'
    model.save(temp_name)
    with open(temp_name, 'rb') as infile:
        model_bytes = infile.read()
    out['model_serial'] = model_bytes
    return out


if __name__ == "__main__":
    trials = mongoexp.MongoTrials('mongo://localhost:27017/jobs/jobs', exp_key='exp1')
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=trials,
                                          keep_temp=True)  # this last bit is important
    print("Best performing model chosen hyper-parameters:")
    print(best_run)