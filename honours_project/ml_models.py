import GPy
from sklearn import svm, naive_bayes, linear_model, neural_network, model_selection, metrics, feature_selection, decomposition
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
from external_libs.stacked_clf.stacking import StackedClassifier, FWLSClassifier
import numpy as np

def svmDefault():
    return svm.SVC(probability=True)

def elasticNet():
    return linear_model.ElasticNet(alpha=0.1, fit_intercept=False)

def randomForest():
    return RandomForestClassifier(n_estimators=500)

def nnetSklearn():
    return neural_network.MLPClassifier(hidden_layer_sizes=(5, 3), warm_start=False)

def naiveBayes():
    return naive_bayes.GaussianNB()

def GPyRBFKernel(input_dim):
    return GPy.kern.RBF(input_dim=input_dim)

def GPyMatern52Kernel(input_dim):
    return GPy.kern.Matern52(input_dim=input_dim)

def nn1():
    return (neural_network.MLPClassifier(
        hidden_layer_sizes=(16,4),
        activation="relu",
        nesterovs_momentum=True,
        early_stopping=True,
        shuffle=True,
        random_state=1,
        validation_fraction=0.2,
        max_iter=400
    ))

def nn2():
    return (neural_network.MLPClassifier(
        hidden_layer_sizes=(128,16),
        activation="relu",
        nesterovs_momentum=True,
        early_stopping=True,
        shuffle=True,
        random_state=1,
        validation_fraction=0.2,
        max_iter=400
    ))

def nn3():
    return (neural_network.MLPClassifier(
        hidden_layer_sizes=(256,256,128,64),
        activation="relu",
        nesterovs_momentum=True,
        early_stopping=True,
        shuffle=True,
        random_state=1,
        validation_fraction=0.2,
        max_iter=400
    ))

def nn4():
    return (neural_network.MLPClassifier(
        hidden_layer_sizes=(512,256,256,256,160,80),
        activation="relu",
        nesterovs_momentum=True,
        early_stopping=True,
        shuffle=True,
        random_state=1,
        validation_fraction=0.2,
        max_iter=400
    ))

def nn5():
    return (neural_network.MLPClassifier(
        hidden_layer_sizes=(512,512,512,256,256,256,160,80),
        activation="relu",
        nesterovs_momentum=True,
        early_stopping=True,
        shuffle=True,
        random_state=1,
        validation_fraction=0.2,
        max_iter=400
    ))

def nn6():
    return (neural_network.MLPClassifier(
        hidden_layer_sizes=(512,512,512,256,256,256,160,128),
        activation="relu",
        nesterovs_momentum=True,
        early_stopping=True,
        shuffle=True,
        random_state=1,
        validation_fraction=0.2,
        max_iter=400
    ))

def nn7():
    return (neural_network.MLPClassifier(
        hidden_layer_sizes=(1024,512,512,512,256,256,256,256,256),
        activation="relu",
        nesterovs_momentum=True,
        early_stopping=True,
        shuffle=True,
        random_state=1,
        validation_fraction=0.2,
        max_iter=400
    ))

def gp_default():
    return GaussianProcessClassifier()

def ensembleModel_all_hard():
    rf = (RandomForestClassifier(n_estimators=1000))
    svm = (SVC(probability=True, kernel='rbf', gamma=1e-5, C=1000))
    gp = (GaussianProcessClassifier(
        max_iter_predict=200
    ))
    nn = nn4()
    nn_deep = nn7()
    knn = (KNeighborsClassifier(3))

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('svm', svm),
        ('gp', gp), ('nn', nn), ('nn_deep', nn_deep), ('knn', knn)], voting='hard')

    return ensemble

def ensembleModel_all_soft():
    rf = (RandomForestClassifier(n_estimators=1000))
    svm = (SVC(probability=True, kernel='rbf', gamma=1e-5, C=1000))
    gp = (GaussianProcessClassifier(
        max_iter_predict=200
    ))
    nn_small = nn3()
    nn = nn4()
    nn_deep = nn7()
    knn = (KNeighborsClassifier(3))
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('svm', svm), ('nn_small', nn_small),
        ('gp', gp), ('nn', nn), ('nn_deep', nn_deep), ('knn', knn)], voting='soft')

    return ensemble

def ensembleModel_all_stacked():
    rf = (RandomForestClassifier(n_estimators=4000))
    svm = (SVC(probability=True, kernel='rbf', gamma=1e-5, C=1000))
    gp = (GaussianProcessClassifier(
        max_iter_predict=100
    ))
    nn_small = nn3()
    nn = nn4()
    nn_deep = nn6()
    knn = (KNeighborsClassifier(3))
    bclf = (GaussianProcessClassifier(
        max_iter_predict=200
    ))

    clfs=[svm, nn_small,
         gp, nn, nn_deep, knn,rf]


    ensemble = StackedClassifier(bclf, clfs)

    return ensemble

def ensembleModel_all_FWLS():
    rf = (RandomForestClassifier(n_estimators=4000))
    svm = (SVC(probability=True, kernel='rbf', gamma=1e-5, C=1000))
    gp = (GaussianProcessClassifier(
        max_iter_predict=100
    ))
    nn_small = nn3()
    nn = nn4()
    nn_deep = nn6()
    knn = (KNeighborsClassifier(3))
    bclf = (GaussianProcessClassifier(
        max_iter_predict=200
    ))

    clfs=[svm, nn_small,
         gp, nn, nn_deep, knn,rf]

    feature_func = lambda x: np.c_[np.ones((x.shape[0], 1)),]


    ensemble = FWLSClassifier(bclf, clfs, feature_func)

    return ensemble

def ensembleModel_minimal():
    rf = (RandomForestClassifier(n_estimators=2000))
    svm = (SVC(probability=True, kernel='rbf', gamma=1e-5, C=1000))
    gp = (GaussianProcessClassifier(
        max_iter_predict=200
    ))
    nn = (neural_network.MLPClassifier(
        hidden_layer_sizes=(128,16),
        activation="relu",
        nesterovs_momentum=True,
        early_stopping=True,
        shuffle=True,
        random_state=1,
        validation_fraction=0.2,
        max_iter=400
    ))

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('svm', svm),
        ('gp', gp), ('nn', nn)], voting='soft')

    return ensemble
