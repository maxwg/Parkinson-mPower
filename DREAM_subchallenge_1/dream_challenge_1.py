import synapse_helper as synapse
import thread_helper
import pickle
from math import isnan, nan
import numpy as np

def _getFeatures(walk, thread_id, queue):
    """
    Extracts all specified features specified in the write-up.
    See getTrainBasicFeatures() for usage.
    
    Designed for threading.
    
    :param walk: The walk file after mergeWithHeaders
    :param thread_id: An arbitrary ID assigned 
    :param queue: A multiprocessing queue to store the result.
    :return: 
    """
    import dream_challenge_1_features as dream
    count = 0
    for w in walk:
        print(thread_id, count, "of", len(walk))
        count += 1
        if isinstance(w['deviceMotion_walking_rest.json.items'], str):
            try:
                features, errors, accel = dream.getAllFeaturesRest(w['deviceMotion_walking_rest.json.items'], highPass=False)
                w['rest_features'] = features
                w['rest_accel'] = accel
            except:
                """ the deviceMotion is sometimes (rarely) empty,
                    throwing linalg faults.
                """
                w['rest_features'] = None
                w['rest_accel'] = None

            try:
                bpfeatures, errors, bpaccel = dream.getAllFeaturesRest(w['deviceMotion_walking_rest.json.items'], highPass=True)
                w['bp_rest_features'] = bpfeatures
            except:
                """ the deviceMotion is sometimes (rarely) empty,
                    throwing linalg faults.
                """
                w['bp_rest_features'] = None
        else:
            """ Deal with these cases in post-processing.
            """
            w['rest_features'] = None
            w['rest_accel'] = None
            w['bp_rest_features'] = None

        if isinstance(w['deviceMotion_walking_outbound.json.items'], str):
            try:
                features, errors, accel = dream.getAllFeaturesWalking(
                    w['deviceMotion_walking_outbound.json.items'])
                w['walk_features'] = features
                w['walk_accel'] = accel
            except:
                w['walk_features'] = None
                w['walk_accel'] = None
        else:
            w['walk_features'] = None
            w['walk_accel'] = None

        if isinstance(w['pedometer_walking_outbound.json.items'], str):
            try:
                pedo = dream.getPedometerFeatures(w['pedometer_walking_outbound.json.items'])
                w['pedo'] = pedo
            except:
                w['pedo'] = None
        else:
            w['pedo'] = None
        queue.put(w)


def mergeWithHeaders(walk, files, headers):
    res = []
    mergeToKey =lambda h, val: files[str(int(val))] if "json.items" in h and not isnan(val) else val
    for w in walk:
        res.append({h: mergeToKey(h, val) for val,h in zip(w,headers)})
    return res

def getTrainBasicFeatures(threads = 4):
    """
    Get features for the training dataset
    NOTE:: Pickle from synapse.getDreamChallengeData() must previously exist
    :return: None. Saves a pickle for later use. Use loadTrainBasicFeatures()
    """
    walk, walk_files, _ = synapse.restoreSynapseTablesDream()
    walk_headers = walk.columns.values.tolist()
    walk = walk.as_matrix()
    walk_files = walk_files
    walk = mergeWithHeaders(walk, walk_files, walk_headers)
    features = thread_helper.processIterableInProcesses(walk, _getFeatures, threads)
    results = []
    while not features.empty():
        f = features.get()
        results.append(f)

    with open('dream_train_features.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
        pickle.dump(results, f)

def getTestBasicFeatures(threads = 4):
    walk, walk_files = synapse.restoreSynapseTablesDreamTest()
    walk_headers = walk.columns.values.tolist()
    walk = walk.as_matrix()
    walk_files = walk_files
    walk = mergeWithHeaders(walk, walk_files, walk_headers)
    features = thread_helper.processIterableInProcesses(walk, _getFeatures, threads)
    results = []
    while not features.empty():
        f = features.get()
        results.append(f)

    with open('dream_test_features.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
        pickle.dump(results, f)

def getSupplementaryBasicFeatures(threads = 4):
    walk, walk_files = synapse.restoreSynapseTablesDreamSupplementary()
    walk_headers = walk.columns.values.tolist()
    walk = walk.as_matrix()
    walk_files = walk_files
    walk = mergeWithHeaders(walk, walk_files, walk_headers)
    features = thread_helper.processIterableInProcesses(walk, _getFeatures, threads)
    results = []
    while not features.empty():
        f = features.get()
        results.append(f)

    with open('dream_supp_features.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
        pickle.dump(results, f)

def loadTrainBasicFeatures():
    with open('dream_train_features.pickle', 'rb') as f:  # Python 4: open(..., 'rb')
        return pickle.load(f)

def loadTestBasicFeatures():
    with open('dream_test_features.pickle', 'rb') as f:  # Python 4: open(..., 'rb')
        return pickle.load(f)

def loadSupplementaryBasicFeatures():
    with open('dream_supp_features.pickle', 'rb') as f:  # Python 4: open(..., 'rb')
        return pickle.load(f)

def signal_processing_features_normed_to_array(record):
    """
    OLD function (numbers are off).
    This norms (x,y) feature pairs and returns them as one feature.
    However this reduced performance interestingly. This implies that
    the directional data contains information.
    """
    features = []
    rest = record['rest_features']
    bp_rest = record['bp_rest_features']
    walk = record['walk_features']
    pedo = record['pedo']

    #replace missing values with nan for later processing
    if walk is None:
        features.extend([nan] * 98)
    else:
        mp = walk['mpower']
        features.extend(mp[66:])
        features.extend(walk['moments'][:5])
        features.append(np.linalg.norm(walk['moments'][-3:]))
        ent = walk['entropy']
        features.append(np.linalg.norm([ent[0], ent[1], ent[6]]))
        features.extend(ent[2:5])
        features.extend(ent[7:10])
        features.extend(ent[12:15])
        features.extend(walk['fourier'][7:])
        features.extend(walk['tkeo'])
        features.extend(walk['area'])
        dyn = walk['dynamic']
        features.append(np.linalg.norm([dyn[0],dyn[1]]))
        features.append(np.linalg.norm([dyn[2],dyn[3]]))
        features.append(np.linalg.norm([dyn[4],dyn[5]]))
        features.append(np.linalg.norm([dyn[6],dyn[7]]))
        features.append(np.linalg.norm([dyn[8],dyn[9]]))
        features.append(np.linalg.norm([dyn[10],dyn[11]]))
        features.append(np.linalg.norm([dyn[12],dyn[13]]))
        id = walk["info_dynamic"]
        features.append(np.linalg.norm([id[0],id[1]]))
        features.append(np.linalg.norm([id[2],id[3]]))
        features.append(np.linalg.norm([id[4],id[5]]))
        features.append(np.linalg.norm([id[6],id[7]]))
        features.append(np.linalg.norm([id[8],id[12]]))
        features.append(np.linalg.norm([id[9],id[13]]))
        features.append(np.linalg.norm([id[10],id[14]]))
        features.append(np.linalg.norm([id[11],id[15]]))
        hj = walk["hjorth"]
        features.append(np.linalg.norm([hj[0],hj[3]]))
        features.append(np.linalg.norm([hj[1],hj[4]]))
        features.append(np.linalg.norm([hj[2],hj[5]]))
    if rest is None:
        features.extend([nan] * 63)
    else:
        features.extend(rest['mpower'])
        features.extend(rest['moments'][:5])
        ent = rest['entropy']
        features.append(np.linalg.norm([ent[0], ent[1]]))
        features.extend(ent[2:])
        features.extend(rest['fourier'][7:])
        features.extend(rest['tkeo'])
        features.extend(rest['area'])
        dyn = rest['dynamic']
        features.append(np.linalg.norm([dyn[0],dyn[1]]))
        features.append(np.linalg.norm([dyn[2],dyn[3]]))
        features.append(np.linalg.norm([dyn[4],dyn[5]]))
        features.append(np.linalg.norm([dyn[6],dyn[7]]))
        features.append(np.linalg.norm([dyn[8],dyn[9]]))
        features.append(np.linalg.norm([dyn[10],dyn[11]]))
        features.append(np.linalg.norm([dyn[12],dyn[13]]))
        id = rest["info_dynamic"]
        features.append(np.linalg.norm([id[0],id[1]]))
        features.append(np.linalg.norm([id[2],id[3]]))
        features.append(np.linalg.norm([id[4],id[5]]))
        features.append(np.linalg.norm([id[6],id[7]]))
        features.append(np.linalg.norm([id[8],id[12]]))
        features.append(np.linalg.norm([id[9],id[13]]))
        features.append(np.linalg.norm([id[10],id[14]]))
        features.append(np.linalg.norm([id[11],id[15]]))
        hj = rest["hjorth"]
        features.append(np.linalg.norm([hj[0],hj[3]]))
        features.append(np.linalg.norm([hj[1],hj[4]]))
        features.append(np.linalg.norm([hj[2],hj[5]]))
    if bp_rest is None:
        features.extend([nan] * 49)
    else:
        features.extend(bp_rest['mpower'])
        features.extend(bp_rest['moments'][:5])
        ent = bp_rest['entropy']
        features.append(np.linalg.norm([ent[0], ent[1]]))
        features.extend(ent[2:])
        # features.extend(bp_rest['fourier'][7:])
        features.extend(bp_rest['tkeo'])
        features.extend(bp_rest['area'])
        dyn = bp_rest['dynamic']
        features.append(np.linalg.norm([dyn[0],dyn[1]]))
        features.append(np.linalg.norm([dyn[2],dyn[3]]))
        features.append(np.linalg.norm([dyn[4],dyn[5]]))
        features.append(np.linalg.norm([dyn[6],dyn[7]]))
        features.append(np.linalg.norm([dyn[8],dyn[9]]))
        features.append(np.linalg.norm([dyn[10],dyn[11]]))
        features.append(np.linalg.norm([dyn[12],dyn[13]]))
        id = bp_rest["info_dynamic"]
        features.append(np.linalg.norm([id[0],id[1]]))
        features.append(np.linalg.norm([id[2],id[3]]))
        features.append(np.linalg.norm([id[4],id[5]]))
        features.append(np.linalg.norm([id[6],id[7]]))
        features.append(np.linalg.norm([id[8],id[12]]))
        features.append(np.linalg.norm([id[9],id[13]]))
        features.append(np.linalg.norm([id[10],id[14]]))
        features.append(np.linalg.norm([id[11],id[15]]))
        hj = bp_rest["hjorth"]
        features.append(np.linalg.norm([hj[0],hj[3]]))
        features.append(np.linalg.norm([hj[1],hj[4]]))
        features.append(np.linalg.norm([hj[2],hj[5]]))

    if pedo is None:
        features.extend([nan] * 3)
    else:
        features.extend(pedo)

    return features

def signal_processing_features_to_array(record):
    features = []
    rest = record['rest_features']
    bp_rest = record['bp_rest_features']
    walk = record['walk_features']
    pedo = record['pedo']

    # replace missing values with nan for later processing
    if walk is None:
        features.extend([nan] * 235)
    else:
        features.extend(walk['mpower'])
        features.extend(walk['moments'])
        features.extend(walk['entropy'])
        features.extend(walk['fourier'])
        features.extend(walk['tkeo'])
        features.extend(walk['area'])
        features.extend(walk['dynamic'])
        features.extend(walk['info_dynamic'])
        features.extend(walk['hjorth'])
    if rest is None:
        features.extend([nan] * 107)
    else:
        features.extend(rest['mpower'])
        features.extend(rest['moments'])
        features.extend(rest['entropy'])
        features.extend(rest['fourier'])
        features.extend(rest['tkeo'])
        features.extend(rest['area'])
        features.extend(rest['dynamic'])
        features.extend(rest['info_dynamic'])
        features.extend(rest['hjorth'])
    if bp_rest is None:
        features.extend([nan] * 86)
    else:
        features.extend(bp_rest['mpower'])
        features.extend(bp_rest['moments'])
        features.extend(bp_rest['entropy'])
        features.extend(bp_rest['tkeo'])
        features.extend(bp_rest['area'])
        features.extend(bp_rest['dynamic'])
        features.extend(bp_rest['info_dynamic'])
        features.extend(bp_rest['hjorth'])
    if pedo is None:
        features.extend([nan] * 3)
    else:
        features.extend(pedo)

    return features


def signal_processing_features_names():
    import dream_challenge_1_features as dream

    features_walk = []
    features_rest = []
    features_bp_rest = []
    walk = dream.featuresWalkNames()
    bp_rest = dream.featuresRestNames()
    rest = dream.featuresRestNames()

    pedo = ["steps_time", "number_steps", "distance"]

    features_walk.extend(walk['mpower'])
    features_walk.extend(walk['moments'])
    features_walk.extend(walk['entropy'])
    features_walk.extend(walk['fourier'])
    features_walk.extend(walk['tkeo'])
    features_walk.extend(walk['area'])
    features_walk.extend(walk['dynamic'])
    features_walk.extend(walk['info_dynamic'])
    features_walk.extend(walk['hjorth'])


    features_rest.extend(rest['mpower'])
    features_rest.extend(rest['moments'])
    features_rest.extend(rest['entropy'])
    features_rest.extend(rest['fourier'])
    features_rest.extend(rest['tkeo'])
    features_rest.extend(rest['area'])
    features_rest.extend(rest['dynamic'])
    features_rest.extend(rest['info_dynamic'])
    features_rest.extend(rest['hjorth'])

    features_bp_rest.extend(bp_rest['mpower'])
    features_bp_rest.extend(bp_rest['moments'])
    features_bp_rest.extend(bp_rest['entropy'])
    features_bp_rest.extend(bp_rest['tkeo'])
    features_bp_rest.extend(bp_rest['area'])
    features_bp_rest.extend(bp_rest['dynamic'])
    features_bp_rest.extend(bp_rest['info_dynamic'])
    features_bp_rest.extend(bp_rest['hjorth'])

    features_walk = ["walk_" + name for name in features_walk]
    features_rest = ["rest_" + name for name in features_rest]
    features_bp_rest = ["bp_rest" + name for name in features_bp_rest]


    return features_walk + features_rest + features_bp_rest + pedo

def rawAccelValues(record):
    rest = np.array([(0,0,0)]*1600) if record['rest_accel'] is None else record['rest_accel']
    walk = np.array([(0,0,0)]*850) if record['walk_accel'] is None else record['walk_accel']

    while len(rest) < 1600:
        rest = np.concatenate((rest, rest[:min(len(rest), 1600-len(rest))]))
    while len(walk) < 850:
        walk = np.concatenate((walk, walk[:min(len(walk), 850-len(walk))]))

    return rest, walk

def write_features_to_csv(records, path, features=signal_processing_features_to_array, feat_sel=None, feat_count=350):
    all_features = []
    if not (feat_sel is None):
        with open("features/" + feat_sel + ".pickle", 'rb') as f:
            fsel = pickle.load(f)

    for r in records:
        feat = np.array(features(r))
        if not (feat_sel is None):
            feat = feat[fsel]
        feat = feat.tolist()
        all_features.append([r['recordId']] + feat)

    all_features_np = np.array(all_features)[:,1:].astype(np.float)
    all_features_np = all_features_np[np.all(np.isfinite(all_features_np), axis=1)]
    mean_feat = np.nanmean(all_features_np, axis=0)
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
    hnames = np.array(signal_processing_features_names())
    if not (feat_sel is None):
        hnames = hnames[fsel]
    hnames = hnames.tolist()
    headers = ["recordId"] + hnames
    import pandas as pd 
    df = pd.DataFrame(all_features, columns=headers)
    df.to_csv(path, index=False)



if __name__ == "__main__":
    # getSupplementaryBasicFeatures(threads=8)
    # signal_processing_features_names()
    train = loadTrainBasicFeatures()
    test = loadTestBasicFeatures()
    supp = loadSupplementaryBasicFeatures()
    all = train + test + supp
    write_features_to_csv(all, "dream_mifs.csv", feat_sel="dream_walk_mifs", feat_count=350)
