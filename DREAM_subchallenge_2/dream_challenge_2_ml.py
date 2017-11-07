import dream_challenge_2 as dream
import synapse_helper as synapse
import numpy as np
from collections import defaultdict
import random
from sklearn.model_selection import RepeatedStratifiedKFold
from keras_nets import basicLSTM, basicCNN, basicConvLSTM, ModelCheckpoint, basicConvLSTM_merged_rest_walk, loadNNFromFile, basicConvLSTM_merged_rest_walk_wavenet, basicConvLSTM_merged_rest_walk_wavenet2, challenge2ConvLSTM_wavenet
from keras.utils import to_categorical
from sklearn import metrics
from short_time_fourier import getShortTimeFourier1D
import math



def getAccel(r):
    accel = r['accel']
    accel[accel>15] = 15
    accel[accel<-15] = -15
    # accel_fourier = np.swapaxes(getShortTimeFourier1D(accel, 199, 40),0,2)
    accel_fourier = getShortTimeFourier1D(accel, 199, 40)
    accel_fourier = np.linalg.norm(accel_fourier, axis=0).T
    accel_norm = np.linalg.norm(accel, axis=1).reshape(-1,1)
    return accel_fourier, accel_norm, accel


def trainNetworkFeatureExtraction(model, task="tremor"):
    random.seed(0)
    records = dream.restore_basic_features("train")
    records = dream.filter_subchallenge(records, task)

    patients = defaultdict(list)
    for p in records:
        patients[p["patient"]].append(p)
    patients = list([patients[k] for k in sorted(patients.keys())])
    test = patients[11:15]
    train = patients[:11] + patients[15:]

    train_X_raw = []
    train_X_spectral = []
    train_X_norm = []
    train_y = []
    test_X_raw = []
    test_X_spectral = []
    test_X_norm = []
    test_y = []

    for tasks in test:
        for t in tasks:
            spectral, norm, raw = getAccel(t)
            test_X_raw.append(raw)
            test_X_spectral.append(spectral)
            test_X_norm.append(norm)
            test_y.append(int(t[task+"Score"]))

    for tasks in train:
        for t in tasks:
            spectral, norm, raw = getAccel(t)
            train_X_raw.append(raw)
            train_X_spectral.append(spectral)
            train_X_norm.append(norm)
            train_y.append(int(t[task+"Score"]))

    train_X_raw = np.array(train_X_raw)
    train_X_spectral = np.array(train_X_spectral)
    train_X_norm = np.array(train_X_norm)
    train_y = np.array(train_y)
    test_X_raw = np.array(test_X_raw)
    test_X_spectral = np.array(test_X_spectral)
    test_X_norm = np.array(test_X_norm)
    test_y = np.array(test_y)


    std_raw = np.std(train_X_raw)
    std_spectral = np.std(train_X_spectral)
    std_norm = np.std(train_X_norm)
    mean_raw = np.mean(train_X_raw)
    mean_spectral = np.mean(train_X_spectral)
    mean_norm = np.mean(train_X_norm)
    print(mean_spectral, std_spectral)
    print(mean_norm, std_norm)
    print(mean_raw, std_raw)

    train_X_raw = (train_X_raw - mean_raw) / std_raw
    test_X_raw = (test_X_raw - mean_raw) / std_raw
    train_X_spectral = (train_X_spectral - mean_spectral) / std_spectral
    test_X_spectral = (test_X_spectral - mean_spectral) / std_spectral
    train_X_norm = (train_X_norm - mean_norm) / std_norm
    test_X_norm = (test_X_norm - mean_norm) / std_norm

    X_train = [train_X_spectral, train_X_norm, train_X_raw]
    X_test = [test_X_spectral, test_X_norm, test_X_raw]

    num_classes = 2
    class_weight = None
    if task == "tremor":
        num_classes = 5
        class_weight = 'auto'
    train_y = to_categorical(train_y, num_classes)
    test_y = to_categorical(test_y, num_classes)
    print(np.sum(train_y, axis=0))
    print(np.sum(test_y, axis=0))
    train_model = model(X_train[0][0].shape, X_train[1][0].shape, X_train[2][0].shape, num_classes=num_classes)
    callbacks = [
        ModelCheckpoint("model/accel-{val_fmeasure:.8f}-{fmeasure:.8f}.tmp", monitor="val_fmeasure",
                        save_best_only=False, verbose=0),
    ]
    train_model.fit(X_train, train_y, epochs=100, batch_size=100,
          validation_data=(X_test, test_y), shuffle=True, callbacks=callbacks,
                              class_weight=class_weight)


def getModelSemifinalFunctor(model_path):
    from keras import backend as K
    model = loadNNFromFile(model_path)
    inp = model.input
    outputs = [model.layers[-3].output]
    functor = K.function(inp + [K.learning_phase()], outputs )
    return functor

def loadNorm(norm_file_path):
    with open(norm_file_path) as f:
        mean_spectral, std_spectral = map(float, f.readline().split(" "))
        mean_norm, std_norm = map(float, f.readline().split(" "))
        mean_raw, std_raw = map(float, f.readline().split(" "))
        return mean_spectral, std_spectral, mean_norm, std_norm, mean_raw, std_raw

def getFunctorOutput(functor, X):
    """
    :param functor: functor from getModel*Functor
    :param X: [X1, X2, ...] where X1 is shape [n, d1, d2, ...]
    :return: [[output]]
    """

    layer_outs = functor(X + [1])
    return layer_outs

def getNeuralNetworkFeatures(model_path: str = "brady_796_874", challenge: str = "bradykinesia", output_file: str = "results.csv",
                             batch_size: int = 4000) -> None:
    train = dream.restore_basic_features("train")
    test = dream.restore_basic_features("test")
    all_records = train + test
    all_records = dream.filter_subchallenge(all_records,challenge)
    mean_spectral, std_spectral, mean_norm, std_norm, mean_raw, std_raw = loadNorm("model/" + model_path + ".norm")
    outputs = []
    for b in range(int(math.ceil(len(all_records)/batch_size))):
        batch = all_records[b*batch_size: (b+1)*batch_size]
        print("Batch")
        functor = getModelSemifinalFunctor(model_path+".mdl")
        ids = []
        spectral_input = []
        norm_input = []
        raw_input = []
        for r in batch:
            spectral, norm, raw = getAccel(r)
            spectral = (spectral - mean_spectral)/std_spectral
            norm = (norm-mean_norm)/std_norm
            raw = (raw-mean_raw)/std_raw
            spectral_input.append(spectral)
            norm_input.append(norm)
            raw_input.append(raw)
            ids.append(r["dataFileHandleId"])

        res = getFunctorOutput(functor,[spectral_input, norm_input, raw_input])[0]
        joint = [[a] + b.tolist() for a,b in zip(ids, res)]
        outputs.extend(joint)

    import pandas as pd
    df = pd.DataFrame(outputs)
    df.to_csv(output_file, index=False)
