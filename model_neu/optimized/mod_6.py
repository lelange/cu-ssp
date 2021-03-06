"""Convolutional neural network built with Keras."""

from utils import print_json, train_val_split, accuracy, get_test_data
import numpy as np
import keras
from keras.layers.core import K  # import keras.backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam, Nadam, RMSprop
import tensorflow as tf
from hyperopt import STATUS_OK, STATUS_FAIL
from datetime import datetime
import traceback
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TENSORBOARD_DIR = "TensorBoard/"
WEIGHTS_DIR = "weights/"

MAXLEN_SEQ = 700
NB_CLASSES_FINE = 9
NB_CLASSES_COARSE = 3
NB_FEATURES = 50
MODEL_NAME = "mod_6"


# You may want to reduce this considerably if you don't have a killer GPU:
EPOCHS = 70
STARTING_L2_REG = 0.0007

OPTIMIZER_STR_TO_CLASS = {
    'Adam': Adam,
    'Nadam': Nadam,
    'RMSprop': RMSprop
}

def data():
    data_root = '/nosave/lange/cu-ssp/data/netsurfp/'
    file_train = 'train_700'
    file_test = ['cb513_700', 'ts115_700', 'casp12_700']

    X_test = np.load(data_root + file_test[0] + '_input.npy')
    profiles = np.load(data_root + file_test[0] + '_hmm.npy')
    mean = np.mean(profiles)
    std = np.std(profiles)
    X_aug_test = (profiles - mean) / std
    X_test_aug = np.concatenate((X_test, X_aug_test), axis = 2)
    y_test = np.load(data_root + file_test[0] + '_q9.npy')

    X_train = np.load(data_root + file_train + '_input.npy')
    profiles = np.load(data_root + file_train + '_hmm.npy')
    mean = np.mean(profiles)
    std = np.std(profiles)
    X_aug_train = (profiles - mean) / std
    X_train_aug = [X_train, X_aug_train]
    y_train = np.load(data_root + file_train + '_q9.npy')

    X_train_aug, y_train, X_val_aug, y_val = train_val_split(X_train_aug, y_train)

    return X_train_aug, y_train, X_val_aug, y_val, X_test_aug, y_test

X_train_aug, y_train, X_val_aug, y_val, X_test_aug, y_test = data()


def evaluate_model(model, load_file, test_ind = None):
    file_test = ['cb513_700', 'ts115_700', 'casp12_700']
    if test_ind is None:
        test_ind = range(len(file_test))
    test_accs = []
    names = []
    for i in test_ind:
        X_test_aug, y_test = get_test_data(file_test[i])
        model.load_weights(load_file)
        score = model.evaluate(X_test_aug, y_test, verbose=2, batch_size=1)
        #print(file_test[i] +' test accuracy: ' + str(score[1]))
        test_accs.append(score[1])
        names.append(file_test[i])
    return dict(zip(names, test_accs))

def build_and_train(hype_space, save_best_weights=True, log_for_tensorboard=False):
    """Build the model and train it."""
    K.set_learning_phase(1)

    # if log_for_tensorboard:
    #     # We need a smaller batch size to not blow memory with tensorboard
    #     hype_space["lr_rate_mult"] = hype_space["lr_rate_mult"] / 10.0
    #     hype_space["batch_size"] = hype_space["batch_size"] / 10.0

    model = build_model(hype_space)

    # K.set_learning_phase(1)
    time_str = datetime.now().strftime("%Y_%m_%d-%H_%M")
    model_weight_name = MODEL_NAME + "-" + time_str

    callbacks = []

    # Weight saving callback:
    if save_best_weights:
        weights_save_path = os.path.join(
            WEIGHTS_DIR, '{}.hdf5'.format(model_weight_name))
        print("Model's weights will be saved to: {}".format(weights_save_path))
        if not os.path.exists(WEIGHTS_DIR):
            os.makedirs(WEIGHTS_DIR)

        callbacks.append(keras.callbacks.ModelCheckpoint(
            weights_save_path,
            monitor='val_accuracy',
            save_best_only=True, mode='max'))

        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10, verbose=1, mode='max'))

    # TensorBoard logging callback:
    log_path = None
    if log_for_tensorboard:
        log_path = os.path.join(TENSORBOARD_DIR, model_weight_name)
        print("Tensorboard log files will be saved to: {}".format(log_path))
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        # Right now Keras's TensorBoard callback and TensorBoard itself are not
        # properly documented so we do not save embeddings (e.g.: for T-SNE).

        # embeddings_metadata = {
        #     # Dense layers only:
        #     l.name: "../10000_test_classes_labels_on_1_row_in_plain_text.tsv"
        #     for l in model.layers if 'dense' in l.name.lower()
        # }

        tb_callback = keras.callbacks.TensorBoard(
            log_dir=log_path,
            histogram_freq=2,
            # write_images=True, # Enabling this line would require more than 5 GB at each `histogram_freq` epoch.
            write_graph=True
            # embeddings_freq=3,
            # embeddings_layer_names=list(embeddings_metadata.keys()),
            # embeddings_metadata=embeddings_metadata
        )
        tb_callback.set_model(model)
        callbacks.append(tb_callback)

    # Train net:
    history = model.fit(
        X_train_aug,
        y_train,
        batch_size=int(hype_space['batch_size']),
        epochs=EPOCHS,
        shuffle=True,
        verbose=2,
        callbacks=callbacks,
        validation_data=(X_val_aug, y_val)
    ).history

    # Test net:
    K.set_learning_phase(0)
    score = evaluate_model(model, weights_save_path)
    print("\n\n")
    max_acc = max(history['val_accuracy'])

    model_name = MODEL_NAME+"_{}_{}".format(str(max_acc), time_str)
    print("Model name: {}".format(model_name))

    # Note: to restore the model, you'll need to have a keras callback to
    # save the best weights and not the final weights. Only the result is
    # saved.
    print(history.keys())
    print(history)
    print('Score: ', score)
    result = {
        # We plug "-val_accuracy" as a
        # minimizing metric named 'loss' by Hyperopt.
        'loss': -max_acc,
        # Misc:
        'model_name': model_name,
        'space': hype_space,
        'status': STATUS_OK
    }

    print("RESULT:")
    print_json(result)

    f = open("/nosave/lange/cu-ssp/model_neu/optimized/logs/test_results_mod6.txt", "a+")
    res = ""
    for k, v in score.items():
        res += str(k)+": "+str(v)+"\t"
    f.write("\n"+str(model_weight_name)+"\t"+ res)
    f.close()

    return model, model_name, result, log_path


def build_model(hype_space):
    """Create model according to the hyperparameter space given."""
    print("Hyperspace:")
    print(hype_space)

    input_layer = keras.layers.Input(
        (MAXLEN_SEQ, NB_FEATURES))

    x = input_layer
    #print(x._keras_shape)
    z = Conv1D(int(hype_space['conv_filter_size']), 11, strides=1, padding='same')(x)
    #print(z._keras_shape)
    w = Conv1D(int(hype_space['conv_filter_size']), 7, strides=1, padding='same')(x)
    if hype_space['use_BN']:
        w = bn(w)
    #print(w._keras_shape)
    x = concatenate([x, z], axis=2)
    #print(x._keras_shape)
    x = concatenate([x, w], axis=2)
    #print(x._keras_shape)
    x = TimeDistributed(Dropout(hype_space['dropout']))(x)
    z = Conv1D(int(hype_space['conv_filter_size']), 5, strides=1, padding='same')(x)
    #print(z._keras_shape)
    w = Conv1D(int(hype_space['conv_filter_size']), 3, strides=1, padding='same')(x)
    if hype_space['use_BN']:
        w = bn(w)
    #print(w._keras_shape)
    x = concatenate([x, z], axis=2)
    #print(x._keras_shape)
    x = concatenate([x, w], axis=2)
    #print(x._keras_shape)
    x = TimeDistributed(Dropout(hype_space['dropout']))(x)
    x = Bidirectional(CuDNNLSTM(units=int(128*hype_space['LSTM_units_mult']), return_sequences=True))(x)
    x = TimeDistributed(Dropout(hype_space['dropout2']))(x)
    #print('units = ', int(128*hype_space['LSTM_units_mult']))
    #print(x._keras_shape)
    # Two heads as outputs:
    q8_output = TimeDistributed(Dense(
        units=NB_CLASSES_FINE,
        activation="softmax",
    ))(x)

    '''
      q3_output = TimeDistributed(Dense(
        units=NB_CLASSES_COARSE,
        activation="softmax",
        name='q3_output'
    ))(x)

    '''

    # Finalize model:
    model = keras.models.Model(
        inputs=[input_layer],
        outputs=[q8_output]
    )
    model.compile(
        optimizer=OPTIMIZER_STR_TO_CLASS[hype_space['optimizer']](
            lr=0.001 * hype_space['lr_rate_mult']
        ),
        loss='categorical_crossentropy',
        metrics=[accuracy]
    )
    return model

def bn(prev_layer):
    """Perform batch normalisation."""
    return keras.layers.normalization.BatchNormalization()(prev_layer)