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


def data():
    with open("hyperas_train.pickle", "rb") as f:
        train = pickle.load(f)

    dim = len(talk_all_features_names()) - 1
    eval_fn = lambda model, epochs: cross_validate(train, selectAllPatients(train), 3, talk_all_features_without(["gender"]), model, probability=False,
                                           isNN=True,# pca_components=dim, pca_kernel="rbf", proportions='equal',
                                           train_prediction=False, epochs = epochs)

    epochChoices = [4, 6, 8, 10, 13, 18, 24]
    return eval_fn, epochChoices

def huber_loss(y_true, y_pred, clip_value=1):
    # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
    # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
    # for details.
    assert clip_value > 0.

    x = y_true - y_pred
    if np.isinf(clip_value):
        # Spacial case for infinity since Tensorflow does have problems
        # if we compare `K.abs(x) < np.inf`.
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if hasattr(tf, 'select'):
            return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
        else:
            return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
    elif K.backend() == 'theano':
        from theano import tensor as T
        return T.switch(condition, squared_loss, linear_loss)
    else:
        raise RuntimeError('Unknown backend "{}".'.format(K.backend()))

def nnetReLU(input_dim, layers = (80, 20)):
    nnet = Sequential()
    for i, nodes in enumerate(layers):
        if i == 0:
            nnet.add(Dense(units=nodes, input_dim=input_dim, kernel_initializer='he_normal', activation='relu'))
        else:
            nnet.add(Dense(units=nodes, kernel_initializer='he_normal', activation='relu'))
    nnet.add(Dense(1, kernel_initializer='normal', activation='relu'))
    nnet.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return nnet


def nnetDenseHyperas3(eval_fn, dim, epochChoices):
    model = Sequential()
    model.add(Dense({{choice([150, 125, 100, 75, 50])}}, kernel_initializer="he_normal", activation="relu", input_dim=dim))
    model.add(Dense({{choice([75, 50, 30, 24])}},kernel_initializer="he_normal", activation="relu"))
    model.add(Dense({{choice([40, 30, 20, 10, 5])}}, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    epochs = {{choice(epochChoices)}}
    score = eval_fn(model, epochs)[0]
    print('Test accuracy:', score)
    return {'loss': -score, 'status': STATUS_OK, 'model': model}

def nnetDenseBigHyperas4(eval_fn, dim, epochChoices):
    model = Sequential()
    model.add(
        Dense({{choice([800, 400, 300, 200, 150])}}, kernel_initializer="he_normal", activation="relu", input_dim=dim))
    model.add(Dense({{choice([300, 250, 170,100])}}, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense({{choice([200, 150, 100, 50])}}, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense({{choice([150, 120, 70, 40, 25])}}, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    epochs = {{choice(epochChoices)}}
    score = eval_fn(model, epochs)[0]
    print('Test accuracy:', score)
    return {'loss': -score, 'status': STATUS_OK, 'model': model}

def nnetDenseBig4_opt(dim):
    """
    {'Activation': 0, 'Dense': 0, 'Dense_1': 3, 'Dense_2': 2, 'Dense_3': 2, 'epochs': 3, 'optimizer': 1}
    optimal epochs: 8
    """
    model = Sequential()
    model.add(Dense(600, kernel_initializer="he_normal", activation="relu", input_dim=dim))
    model.add(Dense(100, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense(100, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense(70, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.add(Activation('relu'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer='adam')
    return model



def nnDense3_150(dim):
    model = Sequential()
    model.add(Dense(125, kernel_initializer="he_normal", activation="relu", input_dim=dim))
    model.add(Dense(50,kernel_initializer="he_normal", activation="relu"))
    model.add(Dense(35, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.add(Activation('relu'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer='adam')
    return model

def nnetDenseHyperas4(eval_fn, dim, epochChoices):
    model = Sequential()
    model.add(Dense({{choice([150, 125, 100, 75, 50])}}, kernel_initializer="he_normal", activation="relu", input_dim=dim))
    model.add(Dense({{choice([75, 50, 30, 24])}},kernel_initializer="he_normal", activation="relu"))
    model.add(Dense({{choice([40, 30, 20])}}, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense({{choice([25, 20, 15, 10, 5])}}, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    epochs = {{choice(epochChoices)}}
    score = eval_fn(model, epochs)[0]
    print('Test accuracy:', score)
    return {'loss': -score, 'status': STATUS_OK, 'model': model}

def nnetDenseHyperas5(eval_fn, dim,epochChoices):
    model = Sequential()
    model.add(Dense({{choice([150, 125, 100, 75, 50])}}, kernel_initializer="he_normal", activation="relu", input_dim=dim))
    model.add(Dense({{choice([100, 75, 50, 30, 24])}},kernel_initializer="he_normal", activation="relu"))
    model.add(Dense({{choice([50, 40, 30, 20])}}, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense({{choice([30, 20, 15, 10])}}, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense({{choice([30, 15, 10, 5, 3])}}, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    epochs = {{choice(epochChoices)}}
    score = eval_fn(model, epochs)[0]
    print('Test accuracy:', score)
    return {'loss': -score, 'status': STATUS_OK, 'model': model}

def nnetDropoutHyperas5_1(eval_fn, dim):
    model = Sequential()
    model.add(Dense({{choice([150, 125, 100, 75, 50])}}, kernel_initializer="he_normal", activation="relu", input_dim=dim))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([100, 75, 50, 30, 24])}},kernel_initializer="he_normal", activation="relu"))
    model.add(Dense({{choice([50, 40, 30, 20])}}, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense({{choice([30, 20, 15, 10])}}, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense({{choice([30, 15, 10, 5, 3])}}, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    score = eval_fn(model)[0]
    print('Test accuracy:', score)
    return {'loss': -score, 'status': STATUS_OK, 'model': model}


def hyperasNnetSearch(model, train, evals = 5):
    with open("hyperas_train.pickle", "wb") as f:
        pickle.dump(train, f)
    best_run, best_model = optim.minimize(model=model,
                                          data= data,
                                          algo=tpe.suggest,
                                          max_evals=evals,
                                          trials=Trials(),
                                          verbose=0)
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print("Evalutation of best performing model:")
    eval_fn, epochChoices = data()
    print(eval_fn(best_model, epochChoices[best_run['epochs']]))

def nnet_balanced(dim):
    model = Sequential()
    model.add(Dense(800, kernel_initializer="he_normal", activation="relu", input_dim=dim, kernel_regularizer=l1(0.015)))
    model.add(Dense(200, kernel_initializer="he_normal", activation="relu", kernel_regularizer=l1(0.001)))
    model.add(Dense(200, kernel_initializer="he_normal", activation="relu", kernel_regularizer=l1(0.0005)))
    model.add(Dense(100, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense(100, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense(80, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense(60, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense(40, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense(30, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense(1, kernel_initializer='normal', activation='relu'))
    model.compile(loss=binary_crossentropy_weighted_keras, metrics=[fmeasure],
                  optimizer='nadam')
    return model

def basicDenseNN_small(dim):
    model = Sequential()
    model.add(Dense(64, kernel_initializer="he_uniform", input_dim=dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(64, kernel_initializer="he_uniform"))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(64, kernel_initializer="he_uniform"))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Dense(1, kernel_initializer="he_uniform"))
    model.add(Activation('sigmoid'))
    opt = keras.optimizers.nadam()
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)
    return model

def basicDenseNN(dim):
    model = Sequential()
    model.add(Dense(400, kernel_initializer="he_uniform", input_dim=dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(400, kernel_initializer="he_uniform"))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(200, kernel_initializer="he_uniform"))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Dense(200, kernel_initializer="he_uniform"))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Dense(100, kernel_initializer="he_uniform"))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Dense(64, kernel_initializer="he_uniform"))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, kernel_initializer="he_uniform"))
    model.add(Activation('sigmoid'))
    opt = keras.optimizers.nadam()
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)
    return model

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

def basicConvLSTM2D_wavenet_regression(rest_dim, walk_dim_raw):

    rest_input = Input(rest_dim, name="rest_input")
    rest_model = LSTM(64, return_sequences=True, input_shape=rest_dim)(rest_input)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.1)(rest_model)
    rest_model = BatchNormalization()(rest_model)


    rest_model = Conv1D(64, 5, kernel_initializer='he_uniform', padding='same')(rest_model)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.1)(rest_model)
    rest_model = BatchNormalization()(rest_model)

    rest_model = Conv1D(64, 5, kernel_initializer='he_uniform', padding='same')(rest_model)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.1)(rest_model)
    rest_model = BatchNormalization()(rest_model)

    rest_model = Conv1D(64, 5, kernel_initializer='he_uniform', padding='same')(rest_model)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.1)(rest_model)
    rest_model = BatchNormalization()(rest_model)

    rest_model = Conv1D(64, 5, kernel_initializer='he_uniform', padding='same')(rest_model)
    rest_model = Activation('relu')(rest_model)
    rest_model = BatchNormalization()(rest_model)
    rest_model = Dropout(0.4)(rest_model)
    rest_out = Flatten()(rest_model)

    walk_input_raw = Input(walk_dim_raw, name="walk_raw_input")
    A, B = wavenetBlock(64,3,2)(walk_input_raw)
    skip_connections = [B]
    for i in range(16):
        A, B = wavenetBlock(64, 3, 2**((i+2)%8))(A)
        skip_connections.append(B)
    walk_out_raw = add(skip_connections)
    walk_out_raw = Activation('relu')(walk_out_raw)
    # walk_out_raw = BatchNormalization()(walk_out_raw)
    walk_out_raw = Convolution1D(1, 1, activation='relu')(walk_out_raw)
    # walk_out_raw = BatchNormalization()(walk_out_raw)
    walk_out_raw = Convolution1D(1, 1)(walk_out_raw)
    # walk_out_raw = BatchNormalization()(walk_out_raw)
    walk_out_raw = Dropout(0.3)(walk_out_raw)
    walk_out_raw = Flatten()(walk_out_raw)


    model = concatenate([rest_out, walk_out_raw])

    model=Dense(256, kernel_initializer='he_uniform')(model)
    model=Activation('relu')(model)
    model=Dropout(0.2)(model)
    model=BatchNormalization()(model)

    model=Dense(128, kernel_initializer='he_uniform')(model)
    model=Activation('relu')(model)
    model=Dropout(0.5)(model)
    model=BatchNormalization()(model)

    model=Dense(24, kernel_initializer='he_uniform')(model)
    model=Activation('relu')(model)
    model=BatchNormalization()(model)

    model_out = Dense(1, activation='linear', name='main_output')(model)

    model = Model(inputs=[rest_input, walk_input_raw], outputs=[model_out])
    opt = keras.optimizers.nadam()
    model.compile(loss=huber_loss,
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def basicConvLSTM2D_wavenet(rest_dim, walk_dim_raw, num_classes=1):

    rest_input = Input(rest_dim, name="rest_input")
    rest_model = LSTM(64, return_sequences=True, input_shape=rest_dim)(rest_input)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.1)(rest_model)
    rest_model = BatchNormalization()(rest_model)


    rest_model = Conv1D(64, 5, kernel_initializer='he_uniform', padding='same')(rest_model)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.1)(rest_model)
    rest_model = BatchNormalization()(rest_model)

    rest_model = Conv1D(64, 5, kernel_initializer='he_uniform', padding='same')(rest_model)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.1)(rest_model)
    rest_model = BatchNormalization()(rest_model)

    rest_model = Conv1D(64, 5, kernel_initializer='he_uniform', padding='same')(rest_model)
    rest_model = Activation('relu')(rest_model)
    rest_model = Dropout(0.1)(rest_model)
    rest_model = BatchNormalization()(rest_model)

    rest_model = Conv1D(64, 5, kernel_initializer='he_uniform', padding='same')(rest_model)
    rest_model = Activation('relu')(rest_model)
    rest_model = BatchNormalization()(rest_model)
    rest_model = Dropout(0.4)(rest_model)
    rest_out = Flatten()(rest_model)

    walk_input_raw = Input(walk_dim_raw, name="walk_raw_input")
    A, B = wavenetBlock(64,3,2)(walk_input_raw)


def challenge2ConvLSTM_wavenet(spectral_dim, walk_dim_norm, walk_dim, num_classes=1):

    spectral_input = Input(spectral_dim, name="spectral_input")
    spectral_model = LSTM(32, return_sequences=True, input_shape=spectral_dim)(spectral_input)
    spectral_model = Activation('relu')(spectral_model)
    spectral_model = Dropout(0.1)(spectral_model)
    spectral_model = BatchNormalization()(spectral_model)


    spectral_model = Conv1D(48, 5, kernel_initializer='he_uniform', padding='same')(spectral_model)
    spectral_model = Activation('relu')(spectral_model)
    spectral_model = Dropout(0.1)(spectral_model)
    spectral_model = BatchNormalization()(spectral_model)

    spectral_model = Conv1D(48, 5, kernel_initializer='he_uniform', padding='same')(spectral_model)
    spectral_model = Activation('relu')(spectral_model)
    spectral_model = Dropout(0.1)(spectral_model)
    spectral_model = BatchNormalization()(spectral_model)

    spectral_model = Conv1D(48, 5, kernel_initializer='he_uniform', padding='same')(spectral_model)
    spectral_model = Activation('relu')(spectral_model)
    spectral_model = Dropout(0.1)(spectral_model)
    spectral_model = BatchNormalization()(spectral_model)

    spectral_model = Conv1D(48, 5, kernel_initializer='he_uniform', padding='same')(spectral_model)
    spectral_model = Activation('relu')(spectral_model)
    spectral_model = Dropout(0.2)(spectral_model)
    spectral_model = BatchNormalization()(spectral_model)

    spectral_model = Conv1D(48, 5, kernel_initializer='he_uniform', padding='same')(spectral_model)
    spectral_model = Activation('relu')(spectral_model)
    spectral_model = BatchNormalization()(spectral_model)
    spectral_model = Dropout(0.5)(spectral_model)
    spectral_out = Flatten()(spectral_model)

    walk_input_norm = Input(walk_dim_norm, name="walk_raw_input")
    A, B = wavenetBlock(32,5,2)(walk_input_norm)
    skip_connections = [B]
    for i in range(16):
        A, B = wavenetBlock(32, 5, 2**((i+2)%8))(A)
        skip_connections.append(B)
    walk_out_norm = add(skip_connections)
    walk_out_norm = Activation('relu')(walk_out_norm)
    # walk_out_norm = BatchNormalization()(walk_out_norm)
    walk_out_norm = Convolution1D(1, 1, activation='relu')(walk_out_norm)
    # walk_out_norm = BatchNormalization()(walk_out_norm)
    walk_out_norm = Convolution1D(1, 1)(walk_out_norm)
    # walk_out_norm = BatchNormalization()(walk_out_norm)
    walk_out_norm = Dropout(0.5)(walk_out_norm)
    walk_out_norm = Flatten()(walk_out_norm)


    walk_input = Input(walk_dim, name="walk_input")
    walk_model = LSTM(32, return_sequences=True, input_shape=walk_dim)(walk_input)
    walk_model = Activation('relu')(walk_model)
    walk_model = Dropout(0.1)(walk_model)
    walk_model = BatchNormalization()(walk_model)
    walk_model = LSTM(32, return_sequences=True, input_shape=walk_dim)(walk_input)
    walk_model = Activation('relu')(walk_model)
    walk_model = Dropout(0.1)(walk_model)
    walk_model = BatchNormalization()(walk_model)
    walk_model = LSTM(32, return_sequences=True, input_shape=walk_dim)(walk_model)
    walk_model = Activation('relu')(walk_model)
    walk_model = BatchNormalization()(walk_model)


    walk_model = Conv1D(48, 5, kernel_initializer='he_uniform', padding='same')(walk_model)
    walk_model = Activation('relu')(walk_model)
    walk_model = Dropout(0.1)(walk_model)
    walk_model = BatchNormalization()(walk_model)

    walk_model = Conv1D(48, 5, kernel_initializer='he_uniform', padding='same')(walk_model)
    walk_model = Activation('relu')(walk_model)
    walk_model = Dropout(0.2)(walk_model)
    walk_model = BatchNormalization()(walk_model)

    walk_model = Conv1D(48, 5, kernel_initializer='he_uniform', padding='same')(walk_model)
    walk_model = Activation('relu')(walk_model)
    walk_model = Dropout(0.1)(walk_model)
    walk_model = BatchNormalization()(walk_model)

    walk_model = Conv1D(48, 5, kernel_initializer='he_uniform', padding='same')(walk_model)
    walk_model = Activation('relu')(walk_model)
    walk_model = BatchNormalization()(walk_model)
    walk_model = Dropout(0.5)(walk_model)
    walk_out = Flatten()(walk_model)

    model = concatenate([spectral_out, walk_out_norm, walk_out])

    model=Dense(256, kernel_initializer='he_uniform')(model)
    model=Activation('relu')(model)
    model=Dropout(0.2)(model)
    model=BatchNormalization()(model)

    model=Dense(128, kernel_initializer='he_uniform')(model)
    model=Activation('relu')(model)
    model=Dropout(0.5)(model)
    model=BatchNormalization()(model)

    model=Dense(24, kernel_initializer='he_uniform')(model)
    model=Activation('relu')(model)
    model=BatchNormalization()(model)

    model_out = Dense(num_classes, activation='softmax', name='main_output')(model)

    model = Model(inputs=[spectral_input, walk_input_norm, walk_input], outputs=[model_out])
    opt = keras.optimizers.nadam()
    model.compile(loss='categorical_crossentropy',
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


def predictNN(model, train, pat, fsel=None):
    X, y = patientsToXy([pat], fsel=fsel, norm=train)
    return np.array(model.predict(X)).flatten()[0]

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

def trainAndTestNN(train, nnet, fsel=None, test=None, filename="", round=0, min_acc=0.88, transfer=None, measure="acc"):
    """
    Similar to cross_validate, however uses 
    :param patients:    
    :param nnet: 
    :return: 
    """
    X, y= patientsToXy(train, fsel=fsel)
    dim = X.shape[1:]
    if transfer is None:
        model = nnet(dim)
    else:
        model = loadNNFromFile(transfer)
        # model.layers[0].trainable = False


    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=0, patience=200, verbose=0, mode='auto'),
        EarlyStopping(monitor="val_"+measure, min_delta=0, patience=300, verbose=0, mode='auto'),
        ModelCheckpoint("model/best_acc-{val_" +measure+ ":.8f}.tmp", monitor="val_"+measure, save_best_only=False, verbose=0),
    ]
    history = model.fit(X,y, batch_size=5000, epochs=10000, shuffle=True, validation_split = 0.25, callbacks=callbacks)
    print("Best Train:", max(history.history[measure]))
    print("Best Val:", max(history.history["val_"+measure]))
    if not (test is None):
        X_test, y_test =  patientsToXy(test, norm=train, fsel=fsel)
        results = []
        for filename in os.listdir("model"):
            if filename.endswith(".tmp") and float(filename[9:18]) > min_acc:
                model_acc = loadNNFromFile(filename)
                test_res = model_acc.evaluate(X_test,y_test, batch_size=5000)
                train_res = model_acc.evaluate(X,y, batch_size=5000)
                print("acc test", test_res)
                print("acc train", train_res)
                results.append((test_res[1], train_res[1], filename))
        results.sort(reverse=True)
        if len(results) > 0:
            os.system("mv model/" + results[0][2] +" model/best"+str(round)+".mdl")
        os.system("rm model/*.tmp")
        if len(results) > 0:
            return results[0][0], results[0][1], "best"+str(round)+".mdl"
    return (0, 0, "failed to achieve baseline acc")

def getBestNeuralNetAfterNEvals(train, nnet=None, evals=10, fsel=None, test=None, filename="", min_acc=0.88, transfer=None, measure="acc"):
    results = []
    for i in range(evals):
        result = trainAndTestNN(train, nnet, fsel=fsel, test=test, filename=filename, round=i, min_acc=min_acc, transfer=transfer, measure=measure)
        results.append(result)
    results.sort(reverse=True)
    print(results)
    return loadNNFromFile(results[0][2])

def transferLearn(nnet_path, train, test, fsel=None, measure="acc"):
    return getBestNeuralNetAfterNEvals(train,fsel=fsel, test=test, transfer=nnet_path, min_acc=0.86, measure=measure)


if __name__ == "__main__":
    from eeglearn import eeg_cnn_lib
    import python_speech_features as psf
    from audio_helpers import *
    import theano.tensor as T
    import lasagne
    import theano
    import time
    from short_time_fourier import *

    def mfcc_to_image(d, samplerate):
        mfcc = psf.mfcc(d, samplerate)
        # mfcc = mfcc[:mfcc.shape[0]//13*13]
        # mfcc = mfcc.reshape(-1, 13, 13)
        return mfcc


    samplerate, data1 = loadAudioAsArray('dataSamples/audio/talk_pd1.wav')
    samplerate, data2 = loadAudioAsArray('dataSamples/audio/talk_pd2.wav')
    samplerate, data3 = loadAudioAsArray('dataSamples/audio/talk_ctrl1.wav')
    samplerate, data4 = loadAudioAsArray('dataSamples/audio/talk_ctrl2.wav')
    batch_size = 100

    mfcc = np.array([mfcc_to_image(d, samplerate) for d in [data1, data3, data2, data4]])
    # mfcc = np.array([np.squeeze(getShortTimeFourier1D(d, 500),0) for d in [data1, data3, data2, data4]])
    X_train=mfcc[0:2]
    X_val = mfcc[2:4]
    y_train = np.array([0,1])
    y_val = np.array([0,1])
    ####TAKE THE SECOND LAST LAYER AS FEATURE!!!!
    print(X_train.shape)

    model = Sequential()
    model.add(LSTM(128, input_shape=X_train.shape[1:]))

    # model.add(Conv1D(32, 3, padding='same',
    #                  input_shape=X_train.shape[1:]))
    #
    # model.add(Activation('relu'))
    # model.add(Conv1D(32, 3, padding='same'))
    #
    # model.add(Activation('relu'))
    # model.add(Conv1D(32, 3, padding='same'))
    #
    # model.add(Activation('relu'))
    # model.add(Conv1D(64, 13, padding='same'))
    #
    # model.add(Activation('relu'))
    # model.add(Conv2D(32, 3))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(64, 3, padding='same'))
    # model.add(Activation('relu'))
    # model.add(Conv2D(64, 3))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(64))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('relu'))

    opt = keras.optimizers.nadam(lr=0.00001)

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.fit(mfcc, [0,1,0,1], epochs=100)
    print(model.predict(mfcc))
