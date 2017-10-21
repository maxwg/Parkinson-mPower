"""
This performs feature selection using skfeature.
MIFS performs best on this dataset it appears.
"""

import dream_challenge_1 as dream
import synapse_helper as synapse
import numpy as np
from collections import defaultdict
import random
from skfeature.function.information_theoretical_based import MIFS, CIFE, JMI, MRMR, ICAP
from skfeature.function.similarity_based import fisher_score, reliefF
from skfeature.function.sparse_learning_based import ls_l21, RFS
from skfeature.function.statistical_based import f_score
from skfeature.function.wrapper import svm_backward, svm_forward
from keras_nets import basicLSTM, basicCNN, basicConvLSTM, ModelCheckpoint, basicConvLSTM_merged_rest_walk, \
    loadNNFromFile, basicConvLSTM_merged_rest_walk_wavenet, basicConvLSTM_merged_rest_walk_wavenet2
from sklearn import metrics
from ml_models import svmDefault
from short_time_fourier import getShortTimeFourier1D
import math
import pickle

random.seed(0)
train = dream.loadTrainBasicFeatures()
train_supp = dream.loadSupplementaryBasicFeatures()

train.extend(train_supp)
# random.shuffle(train)
_, _, demog = synapse.restoreSynapseTablesDream()
demog = demog.set_index('healthCode').T.to_dict()

patients = defaultdict(list)
for p in train:
    patients[p["healthCode"]].append(p)

used_records = []
cur_PD, cur_Ctr = 0, 0
for healthCode in patients:
    if len(patients[healthCode]) > 0:
        diagnosis = demog[healthCode]['professional-diagnosis']
        if diagnosis is True:
            cur_PD += 1
        else:
            cur_Ctr += 1
        used_records.append(patients[healthCode][0])
print(cur_Ctr, cur_PD)

X = []
y = []
for r in used_records:
    feats = dream.signal_processing_features_to_array(r)
    X.append(feats)
    y.append(1 if diagnosis is True else 0)
y = np.array(y)
X = np.array(X)

fs = MIFS.mifs(X, y, n_selected_features=350)
fsname = "dream_walk_mifs"
with open('features/' +fsname + '.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
    pickle.dump(fs, f)

