from feature_subset import *
import numpy as np
from skfeature.function.information_theoretical_based import MIFS, CIFE, JMI, MRMR, ICAP
from skfeature.function.similarity_based import fisher_score, reliefF
from skfeature.function.sparse_learning_based import ls_l21, RFS
from skfeature.function.statistical_based import f_score
from skfeature.function.wrapper import svm_backward, svm_forward
import pickle
from sklearn import svm, gaussian_process
import gc
from patient_selectors import selectAllPatients
from thread_helper import processIterableInThreads
from cross_validate import patientsToXy


def fs_reliefF(patients, fsel):
    X, y = patientsToXy(patients, fsel=fsel)
    fs = reliefF.reliefF(X, y, k=X.shape[0]-1)
    return fs

def fs_MRMR(patients, num_features, fsel):
    X, y = patientsToXy(patients, fsel=fsel)
    fs = MRMR.mrmr(X,y, n_selected_features=num_features)
    return fs

def fs_CIFE(patients, num_features, fsel):
    X, y = patientsToXy(patients, fsel=fsel)
    fs = CIFE.cife(X, y, n_selected_features=num_features)
    return fs

def fs_MIFS(patients, num_features, fsel):
    X, y = patientsToXy(patients, fsel=fsel)
    fs = MIFS.mifs(X,y, n_selected_features=num_features)
    return fs

def fs_JMI(patients, num_features, fsel):
    X, y = patientsToXy(patients, fsel=fsel)
    fs = JMI.jmi(X,y, n_selected_features=num_features)
    return fs

def fs_ICAP(patients, num_features, fsel):
    X, y = patientsToXy(patients, fsel=fsel)
    fs = ICAP.icap(X,y, n_selected_features=num_features)
    return fs

def fs_fisher(patients, fsel):
    X, y = patientsToXy(patients, fsel=fsel)
    fs = fisher_score.fisher_score(X,y)
    return fs

def fs_f_score(patients, fsel):
    X, y = patientsToXy(patients, fsel=fsel)
    fs = f_score.f_score(X,y)
    return [int(f) - 1 for f in fs]

def fs_RFS(patients, gamma, fsel):
    X, y = patientsToXy(patients, fsel=fsel)
    fs = RFS.rfs(X,y, gamma=gamma)
    return fs

def fs_svm_backward(patients, num_features, fsel):
    X, y = patientsToXy(patients, fsel=fsel)
    fs = svm_backward.svm_backward(X,y,num_features)
    return fs

def fs_svm_forward(patients, num_features, fsel):
    X, y = patientsToXy(patients, fsel=fsel)
    fs = svm_forward.svm_forward(X,y,num_features)
    return fs

def fs_l21(patients, gamma, fsel):
    X, y = patientsToXy(patients, fsel=fsel)
    fs = ls_l21.proximal_gradient_descent(X, y, gamma, verbose=False)
    return fs


def getFSResults(train, fsfn, fsname, fsel=None):
    from cross_validate import cross_validate
    from ml_models import randomForest, svmDefault
    fcount = 350#len(all_features_names())
    fs = fsfn(train, fcount, fsel)
    with open('features/' +fsname + str(fcount) + '.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
        pickle.dump(fs, f)
    with open('features/' + fsname + str(fcount) + '.pickle', 'rb') as f:
        fs = pickle.load(f)

    def procThread(group, theadIdx):
        for i in group:
            gc.collect()
            SVM = svmDefault
            # GP = gaussian_process.GaussianProcessClassifier()
            RF = randomForest
            resultsvm = cross_validate(train, selectAllPatients(train), 5, fsel, SVM, proportions="equal",
                                    feature_selection=fs[:i])
            resultgp = cross_validate(train, selectAllPatients(train), 5, fsel, RF, proportions="equal",
                                    feature_selection=fs[:i])

            resultsvm = (np.mean(resultsvm, axis=0))
            resultgp = (np.mean(resultgp, axis=0))

            with open("fs_results/"+fsname, 'a') as log:
                log.write(str(i) + ';' + str(resultsvm) + str(resultgp) + "\n")
            print( fsname, i, resultsvm[0], resultgp[0])
    procThread(list(range(1,fcount+1)), 0)
    # processIterableInThreads(list(range(1, 301)), procThread, 8)

    # thread_helper.processIterableInThreads(list(range(1, len(all_features_names()) + 1)), procThread, 8)

def getFSWithoutFeatureParam(train, fsfn, fsname, isGamma= False, gammaToUse = None, fsel=None):
    from cross_validate import cross_validate
    from ml_models import randomForest, svmDefault
    if not isGamma:
        fs = fsfn(train, fsel)
        with open('features/'+fsname+'.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
            pickle.dump(fs, f)
        with open('features/'+fsname+'.pickle', 'rb') as f:
            fs = pickle.load(f)

    if isGamma:
        def evaluateWithGamma(gamma):
            fs = fsfn(train, gamma, fsel)
            with open('features/' + fsname + str(gamma) + '.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
                pickle.dump(fs, f)
            with open('features/' + fsname + str(gamma) + '.pickle', 'rb') as f:
                fs = pickle.load(f)

            def procThread(group, theadIdx):
                for i in group:
                    gc.collect()

                    SVM = svmDefault
                    RF = randomForest

                    resultsvm = cross_validate(train, selectAllPatients(train), 5, fsel, SVM, proportions="equal",
                                               feature_selection=fs[:i])
                    resultgp = cross_validate(train, selectAllPatients(train), 5, fsel, RF, isNN=False, proportions="equal",
                                              feature_selection=fs[:i])
                    resultsvm = (np.mean(resultsvm, axis=0))
                    resultgp = (np.mean(resultgp, axis=0))

                    with open("fs_results/"+fsname, 'a') as log:
                        log.write(str(i) + ';' + str(gamma) + ';' + str(resultsvm) + str(resultgp) + "\n")
                    print(fsname, i, resultsvm[0], resultgp[0])

            procThread(list(range(1, 350)), 0)
            # processIterableInThreads(list(range(1, 301)), procThread, 8)
        if gammaToUse is None:
            for gamma in [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]:
                evaluateWithGamma(gamma)
        else:
            evaluateWithGamma(gammaToUse)
    else:
        def procThread(group, theadIdx):
            for i in group:
                gc.collect()
                SVM = svmDefault
                RF = randomForest

                # GP = gaussian_process.GaussianProcessClassifier()
                resultsvm = cross_validate(train, selectAllPatients(train), 5, fsel, SVM, proportions="equal",
                                        feature_selection=fs[:i])
                resultgp = cross_validate(train, selectAllPatients(train), 5, fsel, RF, proportions="equal",
                                        feature_selection=fs[:i])

                resultsvm = (np.mean(resultsvm, axis=0))
                resultgp = (np.mean(resultgp, axis=0))

                with open("fs_results/"+fsname, 'a') as log:
                    log.write(str(i) + ';' + str(resultsvm) + str(resultgp) + "\n")
                print(fsname, i, resultsvm[0], resultgp[0])
        procThread(list(range(1, 350)), 0)
        # processIterableInThreads(list(range(1, 301)), procThread, 8)

def fs_reader(fname, gamma=False, size=282):
    with open(fname, 'r') as file:
        svmret = []
        gpret = []
        gammas = []
        curGamma = None
        for line in file:
            if gamma:
                features, gamma, results = line.split(";")
                if gamma != curGamma:
                    gammas.append(gamma)
                    svmret.append([])
                    gpret.append([])
                    curGamma = gamma
            else:
                features, results = line.split(';')
            results = [r + "]" for r in results.split("]")]
            res_svm = eval(results[0])
            res_gp = eval(results[1])
            if gamma:
                svmret[-1].append(res_svm[0])
                gpret[-1].append(res_gp[0])
            else:
                svmret.append(res_svm[0])
                gpret.append(res_gp[0])
        if gamma:
            svmret = np.array([np.array(r) for r in svmret]).T
            gpret = np.array([np.array(r) for r in gpret]).T
            svm_max = np.argmax(svmret, axis=1)
            gp_max = np.argmax(gpret, axis=1)
            svmres = [], []
            gpres = [], []
            for i in range(size):
                svmm = svm_max[i]
                gpm = gp_max[i]
                svmres[0].append(svmret[i, svmm])
                gpres[0].append(gpret[i, gpm])
                svmres[1].append(gammas[svmm])
                gpres[1].append(gammas[gpm])
            return svmres, gpres
        return svmret, gpret
