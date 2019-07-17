"""Convolutional neural network built with Keras."""

from utils import print_json, train_val_split, accuracy
import numpy as np
import keras
from keras.layers.core import K  # import keras.backend as K
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, GRU, Conv1D, CuDNNLSTM, concatenate
from keras.optimizers import Adam, Nadam, RMSprop
import tensorflow as tf
from hyperopt import STATUS_OK, STATUS_FAIL
from datetime import datetime
import uuid
import traceback
import os

TENSORBOARD_DIR = "TensorBoard/"
WEIGHTS_DIR = "weights/"

NB_SAMPLES = 3
MAXLEN_SEQ = 700
# NB_CLASSES = 10
NB_CLASSES_FINE = 8
NB_CLASSES_COARSE = 3
NB_FEATURES = 50

def data():
    data_root = '/nosave/lange/cu-ssp/data/netsurfp/'
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

    X_train_aug, y_train, X_val_aug, y_val = train_val_split(X_train_aug, y_train)

    return X_train_aug, y_train, X_val_aug, y_val, X_test_aug, y_test

X_train_aug, y_train, X_val_aug, y_val, X_test_aug, y_test = data()


# You may want to reduce this considerably if you don't have a killer GPU:
EPOCHS = 10
STARTING_L2_REG = 0.0007

OPTIMIZER_STR_TO_CLASS = {
    'Adam': Adam,
    'Nadam': Nadam,
    'RMSprop': RMSprop
}


def build_and_train(hype_space, save_best_weights=False, log_for_tensorboard=False):
    """Build the model and train it."""
    K.set_learning_phase(1)

    # if log_for_tensorboard:
    #     # We need a smaller batch size to not blow memory with tensorboard
    #     hype_space["lr_rate_mult"] = hype_space["lr_rate_mult"] / 10.0
    #     hype_space["batch_size"] = hype_space["batch_size"] / 10.0

    model = build_model(hype_space)

    # K.set_learning_phase(1)
    time_str = datetime.now().strftime("%Y_%m_%d-%H_%M")
    model_weight_name = "mod_6-" + time_str + ".h5"

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
        verbose=1,
        callbacks=callbacks,
        validation_data=(X_val_aug, y_val)
    ).history

    # Test net:
    K.set_learning_phase(0)
    score = model.evaluate(X_test_aug, y_test, verbose=0)
    max_acc = max(history['val_accuracy'])

    model_name = "model_{}_{}".format(str(max_acc), time_str)
    print("Model name: {}".format(model_name))

    # Note: to restore the model, you'll need to have a keras callback to
    # save the best weights and not the final weights. Only the result is
    # saved.
    print(history.keys())
    print(history)
    print(score)
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

    return model, model_name, result, log_path


def build_model(hype_space):
    """Create model according to the hyperparameter space given."""
    print("Hyperspace:")
    print(hype_space)

    input_layer = keras.layers.Input(
        (MAXLEN_SEQ, NB_FEATURES))

    x = input_layer

    z = Conv1D(64, 11, strides=1, padding='same')(x)
    w = Conv1D(64, 7, strides=1, padding='same')(x)
    x = concatenate([x, z], axis=2)
    x = concatenate([x, w], axis=2)
    z = Conv1D(64, 5, strides=1, padding='same')(x)
    w = Conv1D(64, 3, strides=1, padding='same')(x)
    x = concatenate([x, z], axis=2)
    x = concatenate([x, w], axis=2)
    x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(x)

    # Two heads as outputs:
    q8_outputs = TimeDistributed(Dense(
        units=NB_CLASSES_FINE,
        activation="softmax",
        name='q8_outputs'
    ))(x)

    '''
      q3_outputs = TimeDistributed(Dense(
        units=NB_CLASSES_COARSE,
        activation="softmax",
        name='q3_outputs'
    ))(x)

    '''

    # Finalize model:
    model = keras.models.Model(
        inputs=[input_layer],
        outputs=[q8_outputs]
    )
    model.compile(
        optimizer=OPTIMIZER_STR_TO_CLASS[hype_space['optimizer']](
            lr=0.001 * hype_space['lr_rate_mult']
        ),
        loss='categorical_crossentropy',
        metrics=[accuracy]
    )
    return model
