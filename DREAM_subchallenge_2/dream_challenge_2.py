from synapse_helper import getLDopaChallengeTrain, restoreLDopaChallengeTrain, getLDopaChallengeTest, restoreLDopaChallengeTest
import numpy as np
from dream_challenge_2_features import featuresNames, loadMotionIntoArray, getFeatures
from thread_helper import processIterableInProcesses
from keras_nets import loadNNFromFile
import pickle

def mergeWithHeaders(records, files, headers):
    res = []
    for r in records:
        r = dict({h: val for h, val in zip(headers, r)})
        r['file'] = files[str(r['dataFileHandleId'])]
        res.append(r)
    return res

def get_train_data():
    ldopa, files = restoreLDopaChallengeTrain()
    ldopa_headers = ldopa.columns.values.tolist()
    ldopa = ldopa.as_matrix()
    ldopa = mergeWithHeaders(ldopa, files, ldopa_headers)
    return ldopa

def get_test_data():
    ldopa, files = restoreLDopaChallengeTest()
    ldopa_headers = ldopa.columns.values.tolist()
    ldopa = ldopa.as_matrix()
    ldopa = mergeWithHeaders(ldopa, files, ldopa_headers)
    return ldopa

def is_not_nan_safe(s):
    return isinstance(s, str) or (not np.isnan(s))

def filter_subchallenge(records, task='tremor'):
    """
    :param records: array from get_data()
    :param task: one of {"tremor", "dyskinesia", "bradykinesia"}
    :return:
    """
    return [l for l in records if is_not_nan_safe(l[task+"Score"])]

def filter_task(records, task="drnkg"):
    """
    :param records: array from get_data()
    :param task: One of {'drnkg' 'fldng' 'ftnl1' 'ftnl2' 'ftnr1' 'ftnr2' 'ntblt' 'orgpa' 'raml1' 'raml2' 'ramr1' 'ramr2'}
    :return:
    """
    return [l for l in records if l['task'] == task]

def _extract_features(records, thread_id, queue):
    count = 0
    for r in records:
        count += 1
        print(thread_id, count, "of", len(records))
        sample_time, time_of_day, accel = loadMotionIntoArray(r['file'])
        r["len"] = accel.shape[0]
        while accel.shape[0] < 1000:
            accel = np.concatenate((accel, accel[:min(accel.shape[0], 1000 - len(accel))]))
        accel[accel>15] = 15
        accel[accel<-15] = -15
        r["accel"] = accel[:1000]
        r["time"] = time_of_day
        r['features'] = getFeatures(accel, sample_time, step=20)
        queue.put(r)

def extract_basic_features(records, name="train", threads=6):
    features = processIterableInProcesses(records, _extract_features, threads)
    results = []
    while not features.empty():
        f = features.get()
        results.append(f)

    with open('ldopa_'+name+'_features.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
        pickle.dump(results, f)

def restore_basic_features(ftype="train"):
    with open('ldopa_' + ftype + '_features.pickle', 'rb') as f:  # Python 4: open(..., 'rb')
        return pickle.load(f)


def signal_processing_features_to_array(record, functor=None):
    features = []
    features.extend(record['moments'])
    features.extend(record['entropy'])
    features.extend(record['fourier'])
    features.extend(record['tkeo'])
    features.extend(record['dynamic'])
    features.extend(record['info_dynamic'])
    features.extend(record['hjorth'])

    return features

def meta_feautures(r):
    features = []
    features.append(r["time"])
    features.append(r["len"])
    features.append(r["visit"])
    features.append(r["session"])
    side = 0 if r['deviceSide'].lower() == "left" else 1
    site = 0 if r['site'].lower() == "boston" else 1
    task = ['drnkg', 'fldng', 'ftnl1', 'ftnl2', 'ftnr1', 'ftnr2', 'ntblt', 'orgpa', 'raml1', 'raml2', 'ramr1', 'ramr2'].index(r['task'])
    patient = ['10_BOS', '10_NYC', '11_BOS', '11_NYC', '12_BOS', '12_NYC', '13_BOS', '14_BOS', '15_BOS', '16_BOS', '17_BOS', '18_BOS', '19_BOS', '2_NYC', '3_BOS', '3_NYC', '4_NYC', '5_BOS', '5_NYC', '6_BOS', '6_NYC', '7_BOS', '7_NYC', '8_BOS', '8_NYC', '9_BOS', '9_NYC'].index(r['patient'])
    features.append(side)
    features.append(site)
    features.append(task)
    features.append(patient)
    return features

def meta_features_names():
    return ["time", "len", "visit", "session", "side", "site", "task", "patient"]

def basic_features(r):
    features = []
    features.extend(meta_feautures(r))
    features.extend(signal_processing_features_to_array(r["features"][0]))
    return features

def basic_features_names():
    r = featuresNames()
    features = []
    features.extend(meta_features_names())
    features.extend(signal_processing_features_to_array(r))
    return features

def mean_var_features(r):
    features = []
    features.extend(meta_feautures(r))
    engineered = [signal_processing_features_to_array(r["features"][i]) for i in range(len(r['features']))]
    features.extend(np.mean(engineered, axis=0).tolist())
    features.extend(np.var(engineered, axis=0).tolist())
    return features

def mean_var_features_names():
    r = featuresNames()
    features = []
    features.extend(meta_features_names())
    features.extend(["mean_" + nm for nm in signal_processing_features_to_array(r)])
    features.extend(["var_" + nm for nm in signal_processing_features_to_array(r)])
    return features



def write_features_to_csv(records, path, features=basic_features, feature_names = basic_features_names):
    all_features = []
    for r in records:
        feat = np.array(features(r))
        feat = feat.tolist()
        all_features.append([r['dataFileHandleId']] + feat)

    all_features_np = np.array(all_features)[:,1:].astype(np.float)
    all_features_np = all_features_np[np.all(np.isfinite(all_features_np), axis=1)]
    mean_feat = np.nanmean(all_features_np, axis=0)
    std_feat = np.nanstd(all_features_np, axis=0)
    max_feat = np.nanmax(all_features_np, axis=0)
    min_feat = np.nanmin(all_features_np, axis=0)

    for i in range(len(all_features)):
        for j in range(1, len(all_features[i])):
            if np.isnan(all_features[i][j]):
                all_features[i][j] = mean_feat[j-1]
            elif np.isinf(all_features[i][j]):
                if all_features[i][j] > 0:
                    all_features[i][j] = max_feat[j-1]
                else:
                    all_features[i][j] = min_feat[j-1]
            all_features[i][j] = (all_features[i][j] - mean_feat[j-1])/std_feat[j-1]
            all_features[i][j] = min(1001, max(-1001, all_features[i][j]))

    all_features = np.array(all_features)
    valid_feat_idx = (np.mean(np.abs(all_features) < 1000, axis=0) > 0.995).tolist() #Remove features with a range that is too big after normalisation (=> high variance, uninformative)
    valid_feat_idx[0] = True #always include ID
    all_features = all_features[:,valid_feat_idx]
    hnames = feature_names()
    headers = np.array(["dataFileHandleId"] + hnames)
    headers= headers[valid_feat_idx]
    import pandas as pd
    df = pd.DataFrame(all_features, columns=headers)
    df.to_csv(path, index=False)
    

if __name__ == "__main__":
    getLDopaChallengeTrain()
    getLDopaChallengeTest()
    extract_basic_features(get_train_data(), name="train")
    extract_basic_features(get_test_data(), name="test")
    records = restore_basic_features("train")
    records = restore_basic_features("test")
    # print(np.unique([t['task'] for t in records]))
    # print(np.unique([t['site'] for t in records]))
    # print(np.unique([t['patient'] for t in records]))
    # print(np.unique([t['deviceSide'] for t in records]))
    patients = np.unique([t['patient'] for t in records])
    records = restore_basic_features("train")
    records_test = restore_basic_features("test")
    records = records + records_test
    write_features_to_csv(filter_subchallenge(records, "tremor"), "tremor_mv.csv", features=mean_var_features, feature_names=mean_var_features_names)
    write_features_to_csv(filter_subchallenge(records, "bradykinesia"), "brady_mv.csv", features=mean_var_features, feature_names=mean_var_features_names)
    write_features_to_csv(filter_subchallenge(records, "dyskinesia"), "dysk_mv.csv", features=mean_var_features, feature_names=mean_var_features_names)
    from dream_challenge_2_ml import trainNetworkFeatureExtraction,  challenge2ConvLSTM_wavenet, getNeuralNetworkFeatures
    np.random.seed(0)
    # # trainNetworkFeatureExtraction(challenge2ConvLSTM_wavenet, task='tremor')
    getNeuralNetworkFeatures(model_path="brady_796_874", challenge="bradykinesia", output_file="brady1.csv", batch_size=2000)
    getNeuralNetworkFeatures(model_path="brady_800_820", challenge="bradykinesia", output_file="brady2.csv", batch_size=2000)
    getNeuralNetworkFeatures(model_path="dysk_759_858", challenge="dyskinesia", output_file="dysk1.csv", batch_size=2000)
    getNeuralNetworkFeatures(model_path="dysk_830_862", challenge="dyskinesia", output_file="dysk2.csv", batch_size=2000)
    getNeuralNetworkFeatures(model_path="tremor_618_802", challenge="tremor", output_file="tremor1.csv", batch_size=2000)
    getNeuralNetworkFeatures(model_path="tremor_771_780", challenge="tremor", output_file="tremor2.csv", batch_size=2000)