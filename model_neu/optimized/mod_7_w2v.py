from keras.models import *
from keras.layers import *
from keras import backend as K
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping ,ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler

import dill as pickle
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval, STATUS_FAIL

from utils import *

import numpy as np
import keras
from keras.layers.core import K  # import keras.backend as K
from keras.optimizers import Adam, Nadam, RMSprop
from hyperopt import STATUS_OK, STATUS_FAIL
from datetime import datetime
import traceback
import sys
import os
import time
import tensorflow as tf
sys.path.append('keras-tcn')
from tcn import tcn


data_root = '/nosave/lange/cu-ssp/data/'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TENSORBOARD_DIR = "TensorBoard/"
WEIGHTS_DIR = "weights/"

MAXLEN_SEQ = 700
NB_CLASSES_Q8 = 9
NB_FEATURES = 30
MODEL_NAME = "mod_7_w2v"

epochs = 75
batch_size = 128

EPOCHS = 100
STARTING_L2_REG = 0.0007

OPTIMIZER_STR_TO_CLASS = {
    'Adam': Adam,
    'Nadam': Nadam,
    'RMSprop': RMSprop
}


def build_and_train(hype_space, save_best_weights=True):
    """Build the model and train it."""

    K.set_learning_phase(1)
    model = build_model(hype_space)

    time_str = datetime.now().strftime("%Y_%m_%d-%H_%M")
    model_weight_name = MODEL_NAME+"-" + time_str

    callbacks = []

    # Weight saving callback:
    if save_best_weights:
        weights_save_path = os.path.join(
            WEIGHTS_DIR, '{}.hdf5'.format(model_weight_name))
        print("Model's weights will be saved to: {}".format(weights_save_path))
        if not os.path.exists(WEIGHTS_DIR):
            os.makedirs(WEIGHTS_DIR)

        callbacks.append(ModelCheckpoint(
            filepath=weights_save_path,
            monitor='val_accuracy',
            verbose = 1,
            save_best_only=True, mode='max'))

    callbacks.append(EarlyStopping(
        monitor='val_accuracy',
        patience=10, verbose=1, mode='max'))

    callbacks.append(ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5,
        patience=10, verbose=1, mode='max', cooldown = 2))

    # TensorBoard logging callback (see model 6):
    log_path = None

    emb_dim = int(hype_space['embed_dim'])
    window_size = int(hype_space['window_size'])
    nb_neg = int(hype_space['negative'])
    nb_iter = int(hype_space['iter'])
    n_gram = int(hype_space['n_gram'])
    mod = int(hype_space['model'])
    #tokens = int(hype_space['tokens'])

    #standardize train and val profiles
    X_train, y_train, X_aug = get_netsurf_data('train_full')

    X_train, y_train, X_aug, X_val, y_val, X_aug_val = train_val_split_(X_train, y_train, X_aug)

    ## load data and get embedding form train data, embed train+val
    index2embed = get_embedding(emb_dim, window_size, nb_neg, nb_iter, n_gram, mod,
                                seqs=X_train)

    X_train_embed = embed_data(X_train, index2embed, emb_dim, n_gram)
    X_val_embed = embed_data(X_val, index2embed, emb_dim, n_gram)

    X_train_aug = [X_train_embed, X_aug]
    X_val_aug = [X_val_embed, X_aug_val]

    print('We have '+str(len(callbacks))+' callbacks.')

    # Train net:
    history = model.fit(
        X_train_aug,
        y_train,
        batch_size=int(hype_space['batch_size']),
        epochs=epochs,
        shuffle=True,
        verbose=2,
        callbacks=callbacks,
        validation_data=(X_val_aug, y_val)
    ).history


    # Test net:
    score = evaluate_model(model, weights_save_path,
                           emb_dim, n_gram, index2embed)
    K.set_learning_phase(0)
    print("\n\n")
    min_loss = min(history['val_loss'])
    max_acc = max(history['val_accuracy'])
    number_of_epochs_it_ran = len(history['loss'])

    model_name = MODEL_NAME+"_{}_{}".format(str(max_acc), time_str)
    print("Model name: {}".format(model_name))

    print('Score: ', score)
    result = {
        # We plug "-val_accuracy" as a minimizing metric named 'loss' by Hyperopt.
        'loss': -max_acc,
        'real_loss': min_loss,
        'cb513': score['cb513_full'],
        'casp12':score['casp12_full'],
        'ts115':score['ts115_full'],
        'nb_epochs': number_of_epochs_it_ran,
        # Misc:
        'model_name': model_name,
        'space': hype_space,
        'status': STATUS_OK
    }

    print("RESULT:")
    print_json(result)

    # save test results to logfile
    f = open("/nosave/lange/cu-ssp/model_neu/optimized/logs/test_results_mod7_w2v.txt", "a+")
    res = ""
    for k, v in score.items():
        res += str(k)+": "+str(v)+"\t"
    f.write("\n"+str(model_weight_name)+"\t"+ res)
    f.close()

    return model, model_name, result, log_path


""" Build model """

def build_model(hype_space):
    """Create model according to the hyperparameter space given."""
    print("Hyperspace:")
    print(hype_space)

    input = Input(shape=(MAXLEN_SEQ, int(hype_space['embed_dim']) ))
    profiles_input = Input(shape=(MAXLEN_SEQ, NB_FEATURES,))

    current_layer = input

    if hype_space['first_conv'] is not None:
        k = hype_space['first_conv']
        current_layer = keras.layers.convolutional.Conv1D(
            filters=16, kernel_size=k, strides=1,
            padding='same', activation=hype_space['activation'],
            kernel_regularizer=keras.regularizers.l2(
                STARTING_L2_REG * hype_space['l2_weight_reg_mult'])
        )(current_layer)

    # Core loop that stacks multiple conv+pool layers, with maybe some
    # residual connections and other fluffs:
    n_filters = int(40 * hype_space['conv_hiddn_units_mult'])
    for i in range(hype_space['nb_conv_pool_layers']):
        print(i)
        print(n_filters)
        print(current_layer._keras_shape)

        current_layer = convolution(current_layer, n_filters, hype_space)
        if hype_space['use_BN']:
            current_layer = bn(current_layer)
        print(current_layer._keras_shape)

        deep_enough_for_res = hype_space['conv_pool_res_start_idx']
        if i >= deep_enough_for_res and hype_space['residual'] is not None:
            current_layer = residual(current_layer, n_filters, hype_space)
            print(current_layer._keras_shape)

        current_layer = auto_choose_pooling(
            current_layer, n_filters, hype_space)
        print(current_layer._keras_shape)

        current_layer = dropout(current_layer, hype_space)

        n_filters *= 2
    # Fully Connected (FC) part:
    current_layer = keras.layers.core.Flatten()(current_layer)
    print(current_layer._keras_shape)

    current_layer = keras.layers.core.Dense(
        units=int(1000 * hype_space['fc_units_1_mult']),
        activation=hype_space['activation'],
        kernel_regularizer=keras.regularizers.l2(
            STARTING_L2_REG * hype_space['l2_weight_reg_mult'])
    )(current_layer)
    print(current_layer._keras_shape)

    current_layer = dropout(
        current_layer, hype_space, for_convolution_else_fc=False)

    if hype_space['one_more_fc'] is not None:
        current_layer = keras.layers.core.Dense(
            units=int(750 * hype_space['one_more_fc']),
            activation=hype_space['activation'],
            kernel_regularizer=keras.regularizers.l2(
                STARTING_L2_REG * hype_space['l2_weight_reg_mult'])
        )(current_layer)
        print(current_layer._keras_shape)

        current_layer = dropout(
            current_layer, hype_space, for_convolution_else_fc=False)

    y = TimeDistributed(Dense(NB_CLASSES_Q8, activation="softmax"))(current_layer)

    # Finalize model:
    inp = [input, profiles_input]
    model = Model(inp, y)

    model.compile(
        optimizer=OPTIMIZER_STR_TO_CLASS[hype_space['optimizer']](
            lr=0.001 * hype_space['lr_rate_mult']
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def random_image_mirror_left_right(input_layer):
    """
    Flip each image left-right like in a mirror, randomly, even at test-time.
    This acts as a data augmentation technique. See:
    https://stackoverflow.com/questions/39574999/tensorflow-tf-image-functions-on-an-image-batch
    """
    return keras.layers.core.Lambda(function=lambda batch_imgs: tf.map_fn(
        lambda img: tf.image.random_flip_left_right(img), batch_imgs
    )
    )(input_layer)


def bn(prev_layer):
    """Perform batch normalisation."""
    return keras.layers.normalization.BatchNormalization()(prev_layer)


def dropout(prev_layer, hype_space, for_convolution_else_fc=True):
    """Add dropout after a layer."""
    if for_convolution_else_fc:
        return keras.layers.core.Dropout(
            rate=hype_space['conv_dropout_drop_proba']
        )(prev_layer)
    else:
        return keras.layers.core.Dropout(
            rate=hype_space['fc_dropout_drop_proba']
        )(prev_layer)


def convolution(prev_layer, n_filters, hype_space, force_ksize=None):
    """Basic convolution layer, parametrized by the hype_space."""
    if force_ksize is not None:
        k = force_ksize
    else:
        k = int(round(hype_space['conv_kernel_size']))
    return keras.layers.convolutional.Conv1D(
        filters=n_filters, kernel_size=k, strides=1,
        padding='same', activation=hype_space['activation'],
        kernel_regularizer=keras.regularizers.l2(
            STARTING_L2_REG * hype_space['l2_weight_reg_mult'])
    )(prev_layer)


def residual(prev_layer, n_filters, hype_space):
    """Some sort of residual layer, parametrized by the hype_space."""
    current_layer = prev_layer
    for i in range(int(round(hype_space['residual']))):
        lin_current_layer = keras.layers.convolutional.Conv1D(
            filters=n_filters, kernel_size=1, strides=1,
            padding='same', activation='linear',
            kernel_regularizer=keras.regularizers.l2(
                STARTING_L2_REG * hype_space['l2_weight_reg_mult'])
        )(current_layer)

        layer_to_add = dropout(current_layer, hype_space)
        layer_to_add = convolution(
            layer_to_add, n_filters, hype_space,
            force_ksize=int(round(hype_space['res_conv_kernel_size'])))

        current_layer = keras.layers.add([
            lin_current_layer,
            layer_to_add
        ])
        if hype_space['use_BN']:
            current_layer = bn(current_layer)
    if not hype_space['use_BN']:
        current_layer = bn(current_layer)

    return bn(current_layer)


def auto_choose_pooling(prev_layer, n_filters, hype_space):
    """Deal with pooling in convolution steps."""
    if hype_space['pooling_type'] == 'all_conv':
        current_layer = convolution_pooling(
            prev_layer, n_filters, hype_space)

    elif hype_space['pooling_type'] == 'inception':
        current_layer = inception_reduction(prev_layer, n_filters, hype_space)

    elif hype_space['pooling_type'] == 'avg':
        current_layer = keras.layers.pooling.AveragePooling1D(
            pool_size=2
        )(prev_layer)

    else:  # 'max'
        current_layer = keras.layers.pooling.MaxPooling1D(
            pool_size=(2, 2)
        )(prev_layer)

    return current_layer

def convolution_pooling(prev_layer, n_filters, hype_space):
    """
    Pooling with a convolution of stride 2.
    See: https://arxiv.org/pdf/1412.6806.pdf
    """
    current_layer = keras.layers.convolutional.Conv1D(
        filters=n_filters, kernel_size=3, strides=2,
        padding='same', activation='linear',
        kernel_regularizer=keras.regularizers.l2(
            STARTING_L2_REG * hype_space['l2_weight_reg_mult'])
    )(prev_layer)

    if hype_space['use_BN']:
        current_layer = bn(current_layer)

    return current_layer


def inception_reduction(prev_layer, n_filters, hype_space):
    """
    Reduction block, vaguely inspired from inception.
    See: https://arxiv.org/pdf/1602.07261.pdf
    """
    n_filters_a = int(n_filters * 0.33 + 1)
    n_filters = int(n_filters * 0.4 + 1)

    conv1 = convolution(prev_layer, n_filters_a, hype_space, force_ksize=3)
    conv1 = convolution_pooling(prev_layer, n_filters, hype_space)

    conv2 = convolution(prev_layer, n_filters_a, hype_space, 1)
    conv2 = convolution(conv2, n_filters, hype_space, 3)
    conv2 = convolution_pooling(conv2, n_filters, hype_space)

    conv3 = convolution(prev_layer, n_filters, hype_space, force_ksize=1)
    conv3 = keras.layers.pooling.MaxPooling1D(
        pool_size=3, strides=2, padding='same'
    )(conv3)

    current_layer = keras.layers.concatenate([conv1, conv2, conv3], axis=-1)

    return current_layer
