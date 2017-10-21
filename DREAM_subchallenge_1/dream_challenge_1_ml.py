import dream_challenge_1 as dream
import synapse_helper as synapse
import numpy as np
from collections import defaultdict
import random
from sklearn.model_selection import RepeatedStratifiedKFold
from keras_nets import basicLSTM, basicCNN, basicConvLSTM, ModelCheckpoint, basicConvLSTM_merged_rest_walk, loadNNFromFile, basicConvLSTM_merged_rest_walk_wavenet, basicConvLSTM_merged_rest_walk_wavenet2
from sklearn import metrics
from short_time_fourier import getShortTimeFourier1D
import math


def getPatDetails(pat, demog):
    diagnosis = demog[pat['healthCode']]['professional-diagnosis']
    rest, walk = dream.rawAccelValues(pat)
    rest = getShortTimeFourier1D(rest, 319, 100)
    rest = np.linalg.norm(rest, axis=0).T
    walk_fourier = getShortTimeFourier1D(walk, 99, 20)
    walk_fourier = np.linalg.norm(walk_fourier, axis=0).T
    return diagnosis, rest, walk_fourier, walk


def trainNetworkFeatureExtraction(model):
    random.seed(0)
    train = dream.loadTrainBasicFeatures()
    train_supp = dream.loadSupplementaryBasicFeatures()

    train.extend(train_supp)
    # random.shuffle(train)
    _, _, demog =  synapse.restoreSynapseTablesDream()
    demog = demog.set_index('healthCode').T.to_dict()

    patients = defaultdict(list)
    for p in train:
        patients[p["healthCode"]].append(p)

    train_X = []
    train_y = []
    test_X = []
    test_y = []
    train_ratio = [1,1]
    test_ratio = [1,1]
    for p in patients:
        pat = patients[p]
        num = min(len(pat), 40)
        if sum(test_ratio) < 5000:
            test_X.extend(pat[:num])
            diagnosis = demog[p]['professional-diagnosis']
            if diagnosis is True:
                test_ratio[1] += num
                test_y.extend([1] * num)
            else:
                test_ratio[0] += num
                test_y.extend([0] * num)
        else:
            train_X.extend(pat[:num])
            diagnosis = demog[p]['professional-diagnosis']
            if diagnosis is True:
                train_ratio[1] += num
                train_y.extend([1] * num)
            else:
                train_ratio[0] += num
                train_y.extend([0] * num)
    print(train_ratio)
    print(test_ratio)

    X_test_rest = []
    X_test_walk = []
    X_test_walk_raw = []
    X_train_rest = []
    X_train_walk = []
    X_train_walk_raw = []
    for r in train_X:
        diagnosis, rest, walk_fourier, walk = getPatDetails(r, demog)
        X_train_rest.append(rest)
        X_train_walk.append(walk_fourier)
        X_train_walk_raw.append(np.linalg.norm(walk, axis=1).reshape(-1, 1))

    for r in test_X:
        diagnosis, rest, walk_fourier, walk = getPatDetails(r, demog)
        X_test_rest.append(rest)
        X_test_walk.append(walk_fourier)
        X_test_walk_raw.append(np.linalg.norm(walk, axis=1).reshape(-1, 1))

    X_train_rest = np.array(X_train_rest)
    X_train_walk = np.array(X_train_walk)
    X_train_walk_raw = np.array(X_train_walk_raw)
    X_test_rest = np.array(X_test_rest)
    X_test_walk = np.array(X_test_walk)
    X_test_walk_raw = np.array(X_test_walk_raw)
    y_train = np.array(train_y)
    y_test = np.array(test_y)

    std_rest = np.std(X_train_rest)
    std_walk = np.std(X_train_walk)
    std_walk_raw = np.std(X_train_walk_raw)
    mean_rest = np.mean(X_train_rest)
    mean_walk = np.mean(X_train_walk)
    mean_walk_raw = np.mean(X_train_walk_raw)
    X_train_walk = (X_train_walk - mean_walk) / std_walk
    X_train_walk_raw = (X_train_walk_raw - mean_walk_raw) / std_walk_raw
    X_train_rest = (X_train_rest - mean_rest) / std_rest
    X_test_walk = (X_test_walk - mean_walk) / std_walk
    X_test_walk_raw = (X_test_walk_raw - mean_walk_raw) / std_walk_raw
    X_test_rest = (X_test_rest - mean_rest) / std_rest

    X_train = [X_train_rest, X_train_walk, X_train_walk_raw]
    X_test = [X_test_rest, X_test_walk, X_test_walk_raw]

    print(mean_rest, std_rest)
    print(mean_walk, std_walk)
    print(mean_walk_raw, std_walk_raw)
    train_model = model()
    callbacks = [
        ModelCheckpoint("model/accel-{val_acc:.8f}-{acc:.8f}.tmp", monitor="val_acc",
                        save_best_only=False, verbose=0),
    ]
    history = train_model.fit(X_train, y_train, epochs=100, batch_size=100,
          validation_data=(X_test, y_test), shuffle=True, callbacks=callbacks)

# Norm values:  mean          std
# rest_fourier: 13.9213510908    21.2956919417
# walk_fourier: 6.51941760998    10.4690972001
# walk_raw:     0.00949577175042 0.90424573505

# (5, 160), (8, 53), (849, 3)
# 13.9213510908 21.2956919417
# 9.0683239849 15.1479590001
# -1.60525017931e-05 0.373344411831

def getModelSemifinalFunctor(model_path):
    from keras import backend as K
    model = loadNNFromFile(model_path)
    inp = model.input
    outputs = model.layers[-2].output
    functor = K.function(inp + [K.learning_phase()], outputs )
    return functor

def getFunctorOutput(functor, X):
    """
    :param functor: functor from getModel*Functor
    :param X: [X1, X2, ...] where X1 is shape [n, d1, d2, ...]
    :return: [[output]]
    """

    layer_outs = functor(X + [1])
    return layer_outs

def getWavenetFeatures(model_path="wavenet_6964.mdl", output_file="results.csv", batch_size=4000):
    train = dream.loadTrainBasicFeatures()
    test = dream.loadTestBasicFeatures()
    supp = dream.loadSupplementaryBasicFeatures()
    all_records = train + test + supp
    outputs = []
    for b in range(int(math.ceil(len(all_records)/batch_size))):
        batch = all_records[b*batch_size: (b+1)*batch_size]
        print("Batch")
        functor = getModelSemifinalFunctor(model_path)
        ids = []
        rest_input = []
        walk_input = []
        for r in batch:
            rest, walk = dream.rawAccelValues(r)
            rest = getShortTimeFourier1D(rest, 319, 100)
            rest = np.linalg.norm(rest, axis=0).T
            rest = (rest - 15.1302989726) / 21.0846270793

            walk = np.linalg.norm(walk, axis=1).reshape(-1, 1)
            walk = (walk - 1.30678417463) / 0.874501359381
            rest_input.append(rest)
            walk_input.append(walk)
            ids.append(r["recordId"])

        res = getFunctorOutput(functor,[rest_input, walk_input])
        joint = [[a] + b.tolist() for a,b in zip(ids, res)]
        outputs.extend(joint)

    import pandas as pd
    df = pd.DataFrame(outputs)
    df.to_csv(output_file, index=False)

def getLSTMConvFeatures(model_path="convlstm_1_7173.mdl",output_file="results.csv", batch_size=2000):
    train = dream.loadTrainBasicFeatures()
    test = dream.loadTestBasicFeatures()
    supp = dream.loadSupplementaryBasicFeatures()
    all_records = train + test + supp
    outputs = []
    for b in range(int(math.ceil(len(all_records)/batch_size))):
        batch = all_records[b*batch_size: (b+1)*batch_size]
        print("Batch")
        functor = getModelSemifinalFunctor(model_path)
        ids = []
        rest_input = []
        walk_fourier_input = []
        walk_input = []
        for r in batch:
            rest, walk = dream.rawAccelValues(r)
            rest = getShortTimeFourier1D(rest, 319, 100)
            rest = np.linalg.norm(rest, axis=0).T
            rest = (rest - 15.1445603779) / 21.0774958262

            walk_fourier = getShortTimeFourier1D(walk, 99, 20)
            walk_fourier = np.linalg.norm(walk_fourier, axis=0).T
            walk_fourier = (walk_fourier -8.94668024273) / 14.7980954131

            walk = np.linalg.norm(walk, axis=1).reshape(-1, 1)
            walk = (walk - 1.30608647478) /0.872846015991
            rest_input.append(rest)
            walk_input.append(walk)
            walk_fourier_input.append(walk_fourier)
            ids.append(r["recordId"])

        res = getFunctorOutput(functor,[rest_input, walk_fourier_input, walk_input])
        joint = [[a] + b.tolist() for a,b in zip(ids, res)]
        outputs.extend(joint)

    import pandas as pd
    df = pd.DataFrame(outputs)
    df.to_csv(output_file, index=False)


if __name__== "__main__":
    model = lambda: basicConvLSTM_merged_rest_walk((12, 160), (37, 50), (850, 1))
    # model = lambda: basicConvLSTM_merged_rest_walk_wavenet2((12, 160), (850, 1))
    # model = lambda: basicConvLSTM_merged_rest_walk_wavenet((12, 160), (37, 50), (850, 1))
    trainNetworkFeatureExtraction(model)
    # getLSTMConvFeatures("dream_lstm.csv")
