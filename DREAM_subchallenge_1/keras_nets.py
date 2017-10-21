from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, Dropout, Activation, Conv1D, Conv2D, Flatten, BatchNormalization, AveragePooling1D, AtrousConvolution1D, Convolution1D
from keras.layers.merge import concatenate, multiply, add
from hyperas.distributions import choice, uniform, conditional
from hyperas import optim
from hyperopt import STATUS_OK, Trials, tpe
from feature_selection import *
from patient_selectors import *
from cross_validate import cross_validate
from keras.regularizers import l1
import warnings
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
import os
from sklearn import metrics
from keras import backend as K
from theano import tensor as T
import keras
"""
These metrics are not present in the latest version of Keras for some reason.
"""
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

def _binary_crossentropy_theano_weighted(output, target):
    return -(target*T.basic.log(output)*3 + (1-target) * T.basic.log(1-output))

def _binary_crossentropy_weighted(output, target):
    # avoid numerical instability with clipping
    output = T.clip(output, 10e-8, 1.0 - 10e-8)
    return _binary_crossentropy_theano_weighted(output, target)


def binary_crossentropy_weighted_keras(y_true, y_pred):
    return K.mean(_binary_crossentropy_weighted(y_pred, y_true), axis=-1)


class roc_callback(keras.callbacks.Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = metrics.roc_auc_score(self.y, y_pred)

        y_pred_val = self.model.predict(self.x_val)
        roc_val = metrics.roc_auc_score(self.y_val, y_pred_val)

        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc, 4)), str(round(roc_val, 4))), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


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

def fmeasure(y_true, y_pred):
    '''Calculates the f-measure, the harmonic mean of precision and recall.
    '''
    return fbeta_score(y_true, y_pred, beta=1)



def basicLSTM(dim):
    model = Sequential()
    model.add(LSTM(160, input_shape=dim))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    opt = keras.optimizers.nadam()

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def basicCNN(dim):
    model = Sequential()
    model.add(Conv1D(32, 5, kernel_initializer='he_uniform', input_shape=dim, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(32, 5, kernel_initializer='he_uniform', padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv1D(32, 5, kernel_initializer='he_uniform', padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv1D(64, 5, kernel_initializer='he_uniform', padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv1D(64, 5, kernel_initializer='he_uniform', padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv1D(64, 5, kernel_initializer='he_uniform', padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv1D(64, 5, kernel_initializer='he_uniform', padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(64, kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(16, kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    opt = keras.optimizers.nadam()
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def basicCNN2D(dim):
    model = Sequential()
    model.add(Conv2D(32, 5, kernel_initializer='he_uniform', input_shape=dim, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, 5, kernel_initializer='he_uniform', padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(32, 5, kernel_initializer='he_uniform', padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(64, 5, kernel_initializer='he_uniform', padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(64, 5, kernel_initializer='he_uniform', padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(64, 5, kernel_initializer='he_uniform', padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(64, 5, kernel_initializer='he_uniform', padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(64, kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    opt = keras.optimizers.nadam()
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def basicConvLSTM(dim):
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    #model.add(LSTM(64, return_sequences=True))
    #model.add(Dropout(0.5))
    #model.add(Activation('relu'))
    #model.add(BatchNormalization())

    model.add(Conv1D(64, 5, kernel_initializer='he_uniform', padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(Conv1D(64, 5, kernel_initializer='he_uniform', padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(Conv1D(64, 5, kernel_initializer='he_uniform', padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(Conv1D(64, 5, kernel_initializer='he_uniform', padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.6))
    model.add(BatchNormalization())

    model.add(Conv1D(64, 5, kernel_initializer='he_uniform', padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(Conv1D(64, 5, kernel_initializer='he_uniform', padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(128, kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(8, kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    opt = keras.optimizers.nadam()
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def basicConvLSTM_merged_rest_walk(rest_dim, walk_dim, walk_dim_raw):

    rest_input = Input(rest_dim, name="rest_input")
    rest_model = LSTM(32, return_sequences=True, input_shape=rest_dim)(rest_input)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.1)(rest_model)
    rest_model = BatchNormalization()(rest_model)


    rest_model = Conv1D(32, 5, kernel_initializer='he_uniform', padding='same')(rest_model)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.1)(rest_model)
    rest_model = BatchNormalization()(rest_model)

    rest_model = Conv1D(32, 5, kernel_initializer='he_uniform', padding='same')(rest_model)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.1)(rest_model)
    rest_model = BatchNormalization()(rest_model)

    rest_model = Conv1D(32, 5, kernel_initializer='he_uniform', padding='same')(rest_model)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.1)(rest_model)
    rest_model = BatchNormalization()(rest_model)

    rest_model = Conv1D(32, 5, kernel_initializer='he_uniform', padding='same')(rest_model)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.4)(rest_model)
    rest_model = BatchNormalization()(rest_model)
    rest_out = Flatten()(rest_model)

    walk_input = Input(walk_dim, name="walk_input")

    walk_model=LSTM(32, return_sequences=True, input_shape=walk_dim)(walk_input)
    walk_model=Activation('relu')(walk_model)
    walk_model=Dropout(0.1)(walk_model)
    walk_model=BatchNormalization()(walk_model)

    walk_model=LSTM(32, return_sequences=True)(walk_model)
    walk_model=Activation('relu')(walk_model)
    walk_model=Dropout(0.1)(walk_model)
    walk_model=BatchNormalization()(walk_model)

    walk_model=Conv1D(32, 5, kernel_initializer='he_uniform', padding='same')(walk_model)
    walk_model=Activation('relu')(walk_model)
    walk_model=Dropout(0.1)(walk_model)
    walk_model=BatchNormalization()(walk_model)

    walk_model=Conv1D(32, 5, kernel_initializer='he_uniform', padding='same')(walk_model)
    walk_model=Activation('relu')(walk_model)
    walk_model=Dropout(0.1)(walk_model)
    walk_model=BatchNormalization()(walk_model)

    walk_model=Conv1D(32, 5,kernel_initializer='he_uniform', padding='same')(walk_model)
    walk_model=Activation('relu')(walk_model)
    walk_model=Dropout(0.5)(walk_model)
    walk_model=BatchNormalization()(walk_model)

    walk_model=Conv1D(32, 5, kernel_initializer='he_uniform', padding='same')(walk_model)
    walk_model=Activation('relu')(walk_model)
    walk_model=Dropout(0.2)(walk_model)
    walk_model=BatchNormalization()(walk_model)
    walk_out=Flatten()(walk_model)


    walk_input_raw = Input(walk_dim_raw, name="walk_raw_input")
    walk_model_raw=LSTM(32, return_sequences=True, input_shape=walk_dim_raw)(walk_input_raw)
    walk_model_raw=Activation('relu')(walk_model_raw)
    walk_model_raw=Dropout(0.2)(walk_model_raw)
    walk_model_raw=BatchNormalization()(walk_model_raw)
    walk_model_raw=LSTM(32, return_sequences=True)(walk_model_raw)
    walk_model_raw=Activation('relu')(walk_model_raw)
    walk_model_raw=Dropout(0.2)(walk_model_raw)
    walk_model_raw=BatchNormalization()(walk_model_raw)

    walk_model_raw=Conv1D(32, 5,kernel_initializer='he_uniform', padding='same')(walk_model_raw)
    walk_model_raw=Activation('relu')(walk_model_raw)
    walk_model_raw=Dropout(0.1)(walk_model_raw)
    walk_model_raw=BatchNormalization()(walk_model_raw)

    walk_model_raw=Conv1D(32, 5,kernel_initializer='he_uniform', padding='same')(walk_model_raw)
    walk_model_raw=Activation('relu')(walk_model_raw)
    walk_model_raw=Dropout(0.1)(walk_model_raw)
    walk_model_raw=BatchNormalization()(walk_model_raw)

    walk_model_raw=Conv1D(32, 5, kernel_initializer='he_uniform', padding='same')(walk_model_raw)
    walk_model_raw=Activation('relu')(walk_model_raw)
    walk_model_raw=Dropout(0.5)(walk_model_raw)
    walk_model_raw=BatchNormalization()(walk_model_raw)

    # walk_model_raw=Conv1D(64, 5, kernel_initializer='he_uniform', padding='same')(walk_model_raw)
    # walk_model_raw=Activation('relu')(walk_model_raw)
    # walk_model_raw=Dropout(0.33)(walk_model_raw)
    # walk_out_raw=Flatten()(walk_model_raw)

    walk_model_raw=LSTM(32, return_sequences=False)(walk_model_raw)
    walk_model_raw=BatchNormalization()(walk_model_raw)
    walk_model_raw=Activation('relu')(walk_model_raw)
    walk_out_raw=Dropout(0.2)(walk_model_raw)

    model = concatenate([rest_out, walk_out, walk_out_raw])

    model=Dense(512, kernel_initializer='he_uniform')(model)
    model=Activation('relu')(model)
    model=Dropout(0.1)(model)
    model=BatchNormalization()(model)

    model=Dense(128, kernel_initializer='he_uniform')(model)
    model=Activation('relu')(model)
    model=Dropout(0.5)(model)
    model=BatchNormalization()(model)

    model=Dense(24, kernel_initializer='he_uniform')(model)
    model=Activation('relu')(model)
    model=BatchNormalization()(model)

    model_out = Dense(1, activation='sigmoid', name='main_output')(model)

    model = Model(inputs=[rest_input, walk_input, walk_input_raw], outputs=[model_out])
    opt = keras.optimizers.nadam()
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def wavenetBlock(n_atrous_filters, atrous_filter_size, atrous_rate):
    def f(input_):
        residual = input_
        tanh_out = Conv1D(n_atrous_filters, atrous_filter_size,
                                       dilation_rate=atrous_rate,
                                       padding='same',
                                       activation='tanh')(input_)
        # tanh_out = BatchNormalization()(tanh_out)
        sigmoid_out = Conv1D(n_atrous_filters, atrous_filter_size,
                             dilation_rate=atrous_rate,
                                          padding='same',
                                          activation='sigmoid')(input_)
        # sigmoid_out = BatchNormalization()(sigmoid_out)
        merged = multiply([tanh_out, sigmoid_out])
        skip_out = Conv1D(1, 1, activation='relu', padding='same')(merged)
        out = add([skip_out, residual])
        return out, skip_out
    return f

def basicConvLSTM_merged_rest_walk_wavenet2(rest_dim, walk_dim_raw):

    rest_input = Input(rest_dim, name="rest_input")
    rest_model = LSTM(32, return_sequences=True, input_shape=rest_dim)(rest_input)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.1)(rest_model)
    rest_model = BatchNormalization()(rest_model)


    rest_model = Conv1D(48, 5, kernel_initializer='he_uniform', padding='same')(rest_model)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.1)(rest_model)
    rest_model = BatchNormalization()(rest_model)

    rest_model = Conv1D(48, 5, kernel_initializer='he_uniform', padding='same')(rest_model)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.1)(rest_model)
    rest_model = BatchNormalization()(rest_model)

    rest_model = Conv1D(48, 5, kernel_initializer='he_uniform', padding='same')(rest_model)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.1)(rest_model)
    rest_model = BatchNormalization()(rest_model)

    rest_model = Conv1D(48, 5, kernel_initializer='he_uniform', padding='same')(rest_model)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.4)(rest_model)
    rest_model = BatchNormalization()(rest_model)
    rest_out = Flatten()(rest_model)

    walk_input_raw = Input(walk_dim_raw, name="walk_raw_input")
    A, B = wavenetBlock(48,3,2)(walk_input_raw)
    skip_connections = [B]
    for i in range(16):
        A, B = wavenetBlock(48, 3, 2**((i+2)%8))(A)
        skip_connections.append(B)
    walk_out_raw = add(skip_connections)
    walk_out_raw = Activation('relu')(walk_out_raw)
    # walk_out_raw = BatchNormalization()(walk_out_raw)
    walk_out_raw = Convolution1D(1, 1, activation='relu')(walk_out_raw)
    # walk_out_raw = BatchNormalization()(walk_out_raw)
    walk_out_raw = Convolution1D(1, 1)(walk_out_raw)
    walk_out_raw = Dropout(0.3)(walk_out_raw)
    # walk_out_raw = BatchNormalization()(walk_out_raw)
    walk_out_raw = Flatten()(walk_out_raw)


    model = concatenate([rest_out, walk_out_raw])

    model=Dense(256, kernel_initializer='he_uniform')(model)
    model=Activation('relu')(model)
    model=Dropout(0.1)(model)
    model=BatchNormalization()(model)

    model=Dense(128, kernel_initializer='he_uniform')(model)
    model=Activation('relu')(model)
    model=Dropout(0.5)(model)
    model=BatchNormalization()(model)

    model=Dense(16, kernel_initializer='he_uniform')(model)
    model=Activation('relu')(model)
    model=BatchNormalization()(model)

    model_out = Dense(1, activation='sigmoid', name='main_output')(model)

    model = Model(inputs=[rest_input, walk_input_raw], outputs=[model_out])
    opt = keras.optimizers.nadam()
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model




def basicConvLSTM_merged_rest_walk_wavenet(rest_dim, walk_dim, walk_dim_raw):

    rest_input = Input(rest_dim, name="rest_input")
    rest_model = LSTM(32, return_sequences=True, input_shape=rest_dim)(rest_input)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.1)(rest_model)
    rest_model = BatchNormalization()(rest_model)


    rest_model = Conv1D(48, 5, kernel_initializer='he_uniform', padding='same')(rest_model)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.1)(rest_model)
    rest_model = BatchNormalization()(rest_model)

    rest_model = Conv1D(48, 5, kernel_initializer='he_uniform', padding='same')(rest_model)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.1)(rest_model)
    rest_model = BatchNormalization()(rest_model)

    rest_model = Conv1D(48, 5, kernel_initializer='he_uniform', padding='same')(rest_model)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.1)(rest_model)
    rest_model = BatchNormalization()(rest_model)

    rest_model = Conv1D(48, 5, kernel_initializer='he_uniform', padding='same')(rest_model)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.4)(rest_model)
    # rest_model = BatchNormalization()(rest_model)
    rest_out = Flatten()(rest_model)

    walk_input = Input(walk_dim, name="walk_input")

    walk_model=LSTM(48, return_sequences=True, input_shape=walk_dim)(walk_input)
    walk_model=Activation('relu')(walk_model)
    walk_model=Dropout(0.1)(walk_model)
    walk_model=BatchNormalization()(walk_model)

    walk_model=LSTM(48, return_sequences=True)(walk_model)
    walk_model=Activation('relu')(walk_model)
    walk_model=Dropout(0.1)(walk_model)
    walk_model=BatchNormalization()(walk_model)

    walk_model=Conv1D(48, 5, kernel_initializer='he_uniform', padding='same')(walk_model)
    walk_model=Activation('relu')(walk_model)
    walk_model=Dropout(0.1)(walk_model)
    walk_model=BatchNormalization()(walk_model)

    walk_model=Conv1D(48, 5, kernel_initializer='he_uniform', padding='same')(walk_model)
    walk_model=Activation('relu')(walk_model)
    walk_model=Dropout(0.1)(walk_model)
    walk_model=BatchNormalization()(walk_model)

    walk_model=Conv1D(48, 5,kernel_initializer='he_uniform', padding='same')(walk_model)
    walk_model=Activation('relu')(walk_model)
    walk_model=Dropout(0.5)(walk_model)
    walk_model=BatchNormalization()(walk_model)

    walk_model=Conv1D(48, 5, kernel_initializer='he_uniform', padding='same')(walk_model)
    walk_model=Activation('relu')(walk_model)
    walk_model=Dropout(0.2)(walk_model)
    # walk_model=BatchNormalization()(walk_model)
    walk_out=Flatten()(walk_model)


    walk_input_raw = Input(walk_dim_raw, name="walk_raw_input")
    A, B = wavenetBlock(48,3,2)(walk_input_raw)
    skip_connections = [B]
    for i in range(16):
        A, B = wavenetBlock(48, 3, 2**((i+2)%7))(A)
        skip_connections.append(B)
    walk_out_raw = add(skip_connections)
    walk_out_raw = Activation('relu')(walk_out_raw)
    walk_out_raw = Convolution1D(1, 1, activation='relu')(walk_out_raw)
    walk_out_raw = Convolution1D(1, 1)(walk_out_raw)
    walk_out_raw = Dropout(0.5)(walk_out_raw)
    walk_out_raw = Flatten()(walk_out_raw)


    model = concatenate([rest_out, walk_out, walk_out_raw])

    model=Dense(256, kernel_initializer='he_uniform')(model)
    model=Activation('relu')(model)
    model=Dropout(0.1)(model)
    model=BatchNormalization()(model)

    model=Dense(128, kernel_initializer='he_uniform')(model)
    model=Activation('relu')(model)
    model=Dropout(0.5)(model)
    model=BatchNormalization()(model)

    model=Dense(16, kernel_initializer='he_uniform')(model)
    model=Activation('relu')(model)
    model=BatchNormalization()(model)

    model_out = Dense(1, activation='sigmoid', name='main_output')(model)

    model = Model(inputs=[rest_input, walk_input, walk_input_raw], outputs=[model_out])
    opt = keras.optimizers.nadam()
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def evalNN(model, train, test = None, fsel=None):
    X_train, y_train = patientsToXy(train, fsel=fsel)
    if not (test is None):
        X_test, y_test = patientsToXy(test, fsel=fsel, norm=train)

    train_pred_prob =np.array(model.predict(X_train)).flatten()
    train_pred = train_pred_prob > 0.5
    tp = np.sum([(a == b == True) for (a, b) in zip(train_pred, y_train)])
    fp = np.sum([(a == True and b == False) for (a, b) in zip(train_pred, y_train)])
    tn = np.sum([(a == b == False) for (a, b) in zip(train_pred, y_train)])
    fn = np.sum([(a == False and b == True) for (a, b) in zip(train_pred, y_train)])
    try:
        auc = metrics.roc_auc_score(y_train, train_pred_prob)
    except:
        print("AUC undefined")
    print("***********TRAIN************")
    print("ACC: ",(tp + tn) / (tp + tn + fp + fn))
    print("SEN: ", (tp) / (tp + fn) )
    print("SPE: ",(tn) / (fp + tn))
    print("TOT: ",tp + fp + tn + fn)
    print("TP : ",tp)
    print("FP : ", fp)
    print("TN : ",tn)
    print("FN : ", fn)
    try:
        print("AUC: ", auc)
    except:
        """"""


    if not (test is None):
        test_pred_prob =np.array(model.predict(X_test)).flatten()
        test_pred = test_pred_prob > 0.5
        tp = np.sum([(a == b == True) for (a, b) in zip(test_pred, y_test)])
        fp = np.sum([(a == True and b == False) for (a, b) in zip(test_pred, y_test)])
        tn = np.sum([(a == b == False) for (a, b) in zip(test_pred, y_test)])
        fn = np.sum([(a == False and b == True) for (a, b) in zip(test_pred, y_test)])
        try:
            auc = metrics.roc_auc_score(y_test, test_pred_prob)
        except:
            print("auc is undefined")
        print("***********TEST************")
        print("ACC: ", (tp + tn) / (tp + tn + fp + fn))
        print("SEN: ", (tp) / (tp + fn))
        print("SPE: ", (tn) / (fp + tn))
        print("TOT: ", tp + fp + tn + fn)
        print("TP : ", tp)
        print("FP : ", fp)
        print("TN : ", tn)
        print("FN : ", fn)
        try:
            print("AUC: ", auc)
        except:
            """"""

def loadNNFromFile(path):
    model = load_model("model/" + path, custom_objects={"binary_crossentropy_weighted_keras": binary_crossentropy_weighted_keras,
                                                        'fmeasure': fmeasure})
    return model

def transferLearn(nnet_path, train, test, fsel=None, measure="acc"):
    return getBestNeuralNetAfterNEvals(train,fsel=fsel, test=test, transfer=nnet_path, min_acc=0.86, measure=measure)
