"""
    cross_validate.py

    Our cross_validation code on which all models are evaluated.

    Using this code is not recommended as it is a bit convoluted with
    all of the usage shortcuts. However it has been included for
    verification purposes.
"""

import numpy as np
from collections import defaultdict
from copy import copy
from sklearn.decomposition import PCA, KernelPCA
from sklearn import metrics
from sklearn.base import clone
import GPy
import math
import random
from sklearn.model_selection import RepeatedStratifiedKFold

def patientsToXy(patients, norm=None, fsel=None, healthCodes=False, nan_values=None):
    """

    :param patients:
    :param norm: Use alternate norms.
    :param fsel: Feature subset.
    """
    X = []
    y = []
    hc = []
    if type(patients) is dict:
        patients = list(patients.values())

    for pat in patients:
        feats = fsel(pat)
        X.extend(feats)
        y.extend([pat["isPD"]] * len(feats))
        hc.extend([pat["healthCode"]] * len(feats))

    nan_values_to = None
    X_finite = np.array(X)
    X_finite = X_finite[np.all(np.isfinite(X), axis=1)]
    minimums = np.nanmin(X_finite, axis=0)
    maximums = np.nanmax(X_finite, axis=0)
    if nan_values == 'mean':
        nan_values_to = np.nanmean(X_finite, axis=0)
    elif nan_values == "min":
        nan_values_to = minimums
    elif nan_values == "max":
        nan_values_to = maximums
    elif nan_values == "remove":
        nancols = np.where(np.isnan(X))[1]
        X = np.delete(X, nancols, axis=1)

    if not (nan_values_to is None or nan_values == "remove"):
        for i in range(len(X)):
            for j in range(len(X[i])):
                if np.isnan(X[i][j]):
                    X[i][j] = nan_values_to[j]
                elif np.isinf(X[i][j]):
                    if X[i][j] > 0:
                        X[i][j] = maximums[j]
                    else:
                        X[i][j] = minimums[j]

    X, y = np.array(X).astype(np.float), np.array(y)
    if norm is None:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif not (norm is False):
        X_norm = []
        for pat in norm:
            X_norm.append(fsel(pat))
        X = (X - np.mean(X_norm, axis=0)) / np.std(X_norm, axis=0)

    if healthCodes:
        return X, y, hc
    return X, y


def filterPatientProportions(patients, proportions):
    ratio = 0.5
    if proportions is None:
        return patients
    if proportions == "NCVS":
        ratio = 0.7675
    result = []
    for pat in patients:
        if pat["isPD"]:
            result.append(pat)

    num_pd = len(result)
    for pat in patients:
        if not pat["isPD"] and num_pd/len(result) > ratio:
            result.append(pat)

    return result

def cross_validate(patients,  test_patients, n_folds, feature_selector, model, proportions = None,
                   train_prediction = True, probability = False, feature_selection = None, pca_components = None,
                   pca_kernel = None, isNN = False, nn_batch_size= 400, epochs=5, normalize=True, nan_values='mean',
                   train_model = True, n_repeats=10):
    """
    Perform cross validation and return accuracy, sensitivity and specificity.
    Supports any sklearn-api library.
    :param patients: All patients in the dataset
    :param test_patients: A list of patients that are acceptable for use in test set
    :param n_folds: int - number of cross validation folds
    :param feature_selector: A feature selector from feature_subset.py
    :param model: The ML model with a .fit() and .predict() function
    :param **optional:
                    proportions: default None | equal | NCVS (33-10)
                    feature_selection: an array of feature indices to use
                    pca_components: int -> number of PCA components to use
    :return: (accuracy, sensitivity, specificity, TP, FP, TN, FN)
    """
    X_base, X_all, y_base = [], [], []
    if type(patients) is dict:
        patients = patients.values()

    if not (proportions is None) and len(patients) != len(test_patients):
        raise ("options.proportions are only valid if all patients are selected!")

    for pat in patients:
        features = feature_selector(pat)
        X_all.extend(features)
        if pat not in test_patients:
            X_base.extend(features)
            y_base.append(pat["isPD"])

    X_all = np.array(X_all)

    tpatients = np.array(filterPatientProportions(test_patients, proportions))
    results = []
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=0)
    X_cv, y_cv = patientsToXy(tpatients, fsel=feature_selector, norm=False)
    step = X_cv.shape[0]//len(tpatients)
    X_no_duplicate = X_cv[0::step]
    y_no_duplicate= y_cv[0::step]
    for train_idx, test_idx in rskf.split(X_no_duplicate, y_no_duplicate):
        train_model = model()
        # train, test = tpatients[train_ind], tpatients[test_ind]
        X_train = copy(X_base)
        y_train = copy(y_base)

        train_ind = np.empty((train_idx.size*step), dtype=train_idx.dtype)
        test_ind = np.empty((test_idx.size*step), dtype=test_idx.dtype)
        for i in range(step):
            train_ind[i::step] = train_idx*step+i
            test_ind[i::step] = test_idx*step+i
        y_train.extend(y_cv[train_ind])
        X_train.extend(X_cv[train_ind])

        X_test, y_test = X_cv[test_ind], y_cv[test_ind]

        X_finite = np.array(X_train)
        X_finite = X_finite[np.all(np.isfinite(X_train), axis=1)]

        nan_values_to = None
        minimums = np.nanmin(X_finite, axis=0)
        maximums = np.nanmax(X_finite, axis=0)
        if nan_values=='mean':
            nan_values_to = np.nanmean(X_finite, axis=0)
        elif nan_values=="min":
            nan_values_to = minimums
        elif nan_values=="max":
            nan_values_to = maximums
        elif nan_values=="remove_feature":
            nancols = np.unique(np.where(np.isnan(X_all))[1])
            X_train = np.delete(X_train, nancols, axis=1)
            X_test = np.delete(X_test, nancols, axis=1)
        elif nan_values=="remove_value":
            nanrows = np.union1d(np.where(np.isnan(X_train))[0], np.where(np.isinf(X_train))[0])
            X_train = np.delete(X_train, nanrows, axis=0)
            y_train = np.delete(y_train, nanrows, axis=0)
            nanrows = np.union1d(np.where(np.isnan(X_test))[0], np.where(np.isinf(X_test))[0])
            X_test = np.delete(X_test, nanrows, axis=0)
            y_test = np.delete(y_test, nanrows, axis=0)


        if not (nan_values_to is None or "remove" in nan_values) :
            for i in range(len(X_train)):
                for j in range(len(X_train[i])):
                    if np.isnan(X_train[i][j]):
                        X_train[i][j] = nan_values_to[j]
                    elif np.isinf(X_train[i][j]):
                        if X_train[i][j] > 0:
                            X_train[i][j] = maximums[j]
                        else:
                            X_train[i][j] = minimums[j]

            for i in range(len(X_test)):
                for j in range(len(X_test[i])):
                    if np.isnan(X_test[i][j]):
                        X_test[i][j] = nan_values_to[j]
                    elif np.isinf(X_test[i][j]):
                        if X_test[i][j] > 0:
                            X_test[i][j] = maximums[j]
                        else:
                            X_test[i][j] = minimums[j]

        if normalize:
            norm = np.std(X_train, axis=0)
            mean = np.mean(X_train, axis=0)
            X_train = (X_train -mean) / norm
            X_test = (X_test - mean) / norm
        else:
            X_train, X_test = np.array(X_train), np.array(X_test)
        if not (feature_selection is None):
            X_train = X_train[:,feature_selection]
            X_test = X_test[:,feature_selection]

        if not (pca_components is None):
            if not (pca_kernel is None):
                pca = KernelPCA(n_components=pca_components, kernel=pca_kernel)
            else:
                pca = PCA(n_components=pca_components)
            pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)

        if isNN and train_model:
            history = train_model.fit(X_train, y_train, shuffle=True, epochs=epochs, batch_size=nn_batch_size, validation_data=(X_test,  y_test))
            pred = train_model.predict(X_test)
            pred_prob = np.array(pred).flatten()
            pred = np.array(pred).flatten() > 0.5
        elif isNN:
            pred = train_model.predict(X_test)
            pred_prob = np.array(pred).flatten()
            pred = np.array(pred).flatten() > 0.5
        else:
            train_model.fit(X_train, y_train)
            pred = train_model.predict(X_test)

        tp = np.sum([(a == b == True) for (a,b) in zip(pred,y_test)])
        fp = np.sum([(a == True and b == False) for (a,b) in zip(pred,y_test)])
        tn = np.sum([(a == b == False) for (a,b) in zip(pred,y_test)])
        fn = np.sum([(a == False and b == True) for (a,b) in zip(pred,y_test)])

        result = []
        total = tp+fp+tn+fn
        result.extend([
                (tp + tn)/(tp + tn + fp + fn),
                (tp)/(tp+fn),
                (tn)/(fp+tn),
                tp+fp+tn+fn,
                tp/total,
                fp/total,
                tn/total,
                fn/total
            ])


        if train_prediction:
            train_pred = train_model.predict(X_train)
            train_acc = np.sum(train_pred == y_train)/X_train.shape[0]
            result.append(train_acc)

        if isNN:
            auc = metrics.roc_auc_score(y_test, pred_prob)
            result.append(auc)
        elif probability:
            pred_prob = train_model.predict_proba(X_test)[:,1].T
            auc = metrics.roc_auc_score(y_test, pred_prob)
            result.append(auc)
        else:
            auc = metrics.roc_auc_score(y_test, pred)
            result.append(auc)

        results.append(np.array(result))
        print(result)

    return np.array(results)
