"""
Feature selectors must take in a patient record and return a numerical list of features
"""
import numpy as np
from copy import copy
from voicetoolbox_helper import toolbox_feature_names
from opensmile import eGeneva_feature_idx, eGeneva_feature_names
import pickle
import math

def toolbox_all(pat):
    return np.array(pat["toolbox"])

def eGeneva_all(pat):
    return np.array(pat["best_audio"]["eGeneva"][1:])

def talk_all_features(pat):
    features = copy(pat["toolbox"])
    # Remove duplicate features
    for feature in eGeneva_feature_idx:
        if not any([feature.startswith(fname) for fname in ['name','mfcc', 'jitter', 'shimmer', 'HNR', 'frameTime']]):
            features.append(pat["best_audio"]["eGeneva"][eGeneva_feature_idx[feature]])
    features.extend(demographics_features(pat))
    return [np.array(features)]

def speech_tsanas_dynamic(pat):
    return [np.array(pat["toolbox"] + pat['best_audio']['dynamic'])]


def talk_all_features_segments(pat):
    features_all = []
    for seg in pat['audio_segs']:
        features = copy(seg["toolbox"])
        features += pat['best_audio']['dynamic']
        # Remove duplicate features
        features_all.append(np.array(features))
    return features_all


def talk_all_features_segments_eGeneva_dynamic(pat):
    features_all = []
    for seg in pat['audio_segs']:
        features = copy(seg["toolbox"])
        # Remove duplicate features
        for feature in eGeneva_feature_idx:
            if not any([feature.startswith(fname) for fname in ['name','mfcc', 'jitter', 'shimmer', 'HNR', 'frameTime']]):
                features.append(float(seg["eGeneva"][eGeneva_feature_idx[feature]]))
        features.extend(pat['best_audio']['dynamic'])
        features_all.append(np.array(features))
    return features_all

def talk_all_features_segments_mean(pat):
    features_all = []
    for seg in pat['audio_segs']:
        features = copy(seg["toolbox"])
        features += pat['best_audio']['dynamic']
        # Remove duplicate features
        features_all.append(np.array(features))
    features_all = np.mean(features_all, axis=0)
    return [features_all]


def talk_all_features_segments_concat(pat):
    features_all = []
    for seg in pat['audio_segs']:
        features = copy(seg["toolbox"])
        # Remove duplicate features
        features_all.append(np.array(features))
    features_all = np.concatenate(features_all)
    features_all=np.concatenate((features_all, pat['best_audio']['dynamic']))
    return [features_all]

def talk_all_features_segments_eGeneva_mean(pat):
    features_all = []
    for seg in pat['audio_segs']:
        features = copy(seg["toolbox"])
        features += pat['best_audio']['dynamic']
        # Remove duplicate features
        for feature in eGeneva_feature_idx:
            if not any([feature.startswith(fname) for fname in ['name','mfcc', 'jitter', 'shimmer', 'HNR', 'frameTime']]):
                features.append(float(seg["eGeneva"][eGeneva_feature_idx[feature]]))
        features_all.append(np.array(features))
    features_all = np.mean(features_all, axis=0)
    return [features_all]

def talk_all_features_segments_eGeneva_mean_demo(pat):
    features_all = []
    for seg in pat['audio_segs']:
        features = copy(seg["toolbox"])
        features += pat['best_audio']['dynamic']
        # Remove duplicate features
        for feature in eGeneva_feature_idx:
            if not any([feature.startswith(fname) for fname in ['name','mfcc', 'jitter', 'shimmer', 'HNR', 'frameTime']]):
                features.append(float(seg["eGeneva"][eGeneva_feature_idx[feature]]))
        features.extend(demographics_features(pat))
        features_all.append(np.array(features))
    features_all = np.mean(features_all, axis=0)
    return [features_all]


def talk_all_features_segments_eGeneva_mean_walk(pat):
    features_all = []
    for seg in pat['audio_segs']:
        features = copy(seg["toolbox"])
        features += pat['best_audio']['dynamic']
        # Remove duplicate features
        for feature in eGeneva_feature_idx:
            if not any([feature.startswith(fname) for fname in ['name','mfcc', 'jitter', 'shimmer', 'HNR', 'frameTime']]):
                features.append(float(seg["eGeneva"][eGeneva_feature_idx[feature]]))
        features.extend(walk_all(pat)[0])
        features.extend(demographics_features(pat))
        features_all.append(np.array(features))
    features_all = np.mean(features_all, axis=0)
    return [features_all]


def talk_all_features_segments_ComPARe_mean(pat):
    features_all = []
    for seg in pat['audio_segs']:
        features = copy(seg["toolbox"])
        features += pat['best_audio']['dynamic']
        # Remove duplicate features
        for feature in eGeneva_feature_idx:
            if not any([feature.startswith(fname) for fname in ['name','mfcc', 'jitter', 'shimmer', 'HNR', 'frameTime']]):
                features.append(seg["best_audio"]["eGeneva"][eGeneva_feature_idx[feature]])
        features_all.append(np.array(features))
    features_all = np.mean(features_all, axis=0)
    return [features_all]



def talk_all_features_without_demo(pat):
    features = copy(pat["toolbox"])
    # Remove duplicate features
    for feature in eGeneva_feature_idx:
        if not any([feature.startswith(fname) for fname in ['name','mfcc', 'jitter', 'shimmer', 'HNR', 'frameTime']]):
            features.append(pat["best_audio"]["eGeneva"][eGeneva_feature_idx[feature]])
    return [np.array(features)]

def talk_all_features_without(names_to_exclude):
    excl_idx=[]
    allf = talk_all_features_names()
    for name in names_to_exclude:
        excl_idx.append(allf.index(name))
    excl_idx.sort()
    num_f = len(allf)

    fidxs = []
    ci = 0
    for i in range(num_f):
        if len(excl_idx) <= ci or i != excl_idx[ci]:
            fidxs.append(i)
        else:
            ci+=1

    if ci != len(excl_idx):
        raise ("FS ERROR!")
    return lambda pat: [talk_all_features(pat)[fidxs]]


def talk_all_features_names():
    feature_names = copy(toolbox_feature_names)
    for feature in eGeneva_feature_idx:
        if not any([feature.startswith(fname) for fname in ['name', 'mfcc', 'jitter', 'shimmer', 'HNR', 'frameTime']]):
            feature_names.append(feature)
    feature_names.extend(demographics_feature_names())
    return feature_names

def gait_only(pat):
    #remove 19, 24(23) but why
    rf = copy(pat['walk']['rest_features'])
    rf.pop(19)
    rf.pop(23)
    return [np.array(rf)]

def age_only(pat):
    return np.array([pat["age"]])

def all_features_with_duplicates(pat):
    return [np.concatenate((np.array(pat["toolbox"]), np.array(pat["best_audio"]["eGeneva"][1:])))]

def all_features_with_duplicates_names():
    return [np.concatenate((np.array(toolbox_feature_names), np.array(eGeneva_feature_names[1:])))]

def demographics_features(pat):
    result = []
    result.append(pat["phone"])
    result.append(pat["age"])
    result.append(pat["gender"])
    result.extend(pat["race"])
    return result

def demographics_feature_names():
    return ['phone', 'age', 'gender',"South Asian","White or Caucasian","Mixed","Middle Eastern","East Asian","Latino/Hispanic","Black or African","Caribbean","Native American","Other","Pacific Islander"]

def tsanas_RELIEF(pat):
    feature_idx = [71, 73, 74, 72, 82, 80, 63, 62, 69, 70]
    return [np.array(np.array(pat["toolbox"])[feature_idx])]

def tsanas_nonlinear_2012(pat):
    feature_idx = [i for i in range(0, 112)]
    feature_idx.extend([336, 337, 338])
    return [np.array(np.array(pat["toolbox"])[feature_idx])]

def speech_MFCC(pat):
    return [np.array(pat['best_audio']['mfcc'])]

def speech_dynamic(pat):
    return [np.array(pat['best_audio']['dynamic'])]

def tsanas_RRCT(pat):
    feature_idx = [32, 63, 73, 52, 124, 82, 145, 89, 233, 223, 243, 49, 62]
    return [np.array(np.array(pat["toolbox"])[feature_idx])]

def getBestWalk(walks):
    walks_valid = []
    for walk in walks:
        if not (walk['bp_rest_features'] is None):
            walks_valid.append(walk)
    # walks_valid.sort(key=lambda w: w['bp_rest_features']['area'][0], reverse=False)
    return walks_valid[0]

def walk_arora(pat):
    features = []
    rest = getBestWalk(pat['walks'])['rest_features']#pat['walks'][0]['rest_features']#pat['best_walk']['rest_features']
    walk = getBestWalk(pat['walks'])['walk_features']#pat['walks'][0]['rest_features']#pat['best_walk']['rest_features']

    features.extend(rest['moments'])
    features.extend(rest['area'])
    features.extend(rest['entropy'])
    features.append(rest['dynamic'][6]) #DFA x
    features.append(rest['dynamic'][7]) #DFA y
    features.extend(walk['moments'])
    features.extend(walk['entropy'])
    features.append(walk['dynamic'][6]) #DFA x
    features.append(walk['dynamic'][7]) #DFA y
    return [np.array(features)]

def walk_accel(pat):
    features = []
    best = getBestWalk(pat['walks'])
    rest = best['rest_features']
    walk = best['bp_rest_features']
    features.extend()

def walk_dream_base(pat):
    features = []
    best = getBestWalk(pat['walks'])
    rest = best['rest_features']
    bp_rest = best['bp_rest_features']
    walk = best['walk_features']
    pedo = best['pedo']
    features.extend(rest['mpower'])
    features.extend(walk['mpower'])

    def tryFloat(val):
        try:
            return float(val)
        except:
            return math.nan

    pedo = [tryFloat(v) for v in pedo]
    features.extend(pedo)

    return [features]

def speech_dynamic_only(pat):
    # if not isinstance(pat['best_audio']['dynamic'], list):
    #     return np.array([np.nan]* 17)
    return [np.array(pat['best_audio']['dynamic'] + pat["toolbox"][-3:])]

def walk_dynamic_only(pat):
    features = []
    best = getBestWalk(pat['walks'])
    rest = best['rest_features']
    bp_rest = best['bp_rest_features']
    walk = best['walk_features']
    pedo = best['pedo']
    features.extend(rest['dynamic'])
    features.extend(rest['info_dynamic'])
    features.extend(bp_rest['dynamic'])
    features.extend(bp_rest['info_dynamic'])
    features.extend(walk['dynamic'])
    features.extend(walk['info_dynamic'])
    features.extend(pedo)

    def tryFloat(val):
        try:
            return float(val)
        except:
            return math.nan

    features = np.array([tryFloat(f) for f in features])
    return [features]


def walk_all_normed(pat):
    features = []
    best = getBestWalk(pat['walks'])
    rest = best['rest_features']
    bp_rest = best['bp_rest_features']
    walk = best['walk_features']
    pedo = best['pedo']

    #replace missing values with nan for later processing
    if walk is None:
        features.extend([np.nan] * 98)
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
        # features.append(np.linalg.norm([dyn[12],dyn[13]]))
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
        features.extend([np.nan] * 63)
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
        # features.append(np.linalg.norm([dyn[12],dyn[13]]))
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
        features.extend([np.nan] * 30)
    else:
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
        # features.append(np.linalg.norm([dyn[12],dyn[13]]))
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
        features.extend([np.nan] * 3)
    else:
        features.extend(pedo)

    def tryFloat(val):
        try:
            return float(val)
        except:
            return math.nan

    features = np.array([tryFloat(f) for f in features])

    return [features]


def walk_all(pat):
    features = []
    best = getBestWalk(pat['walks'])
    rest = best['rest_features']
    bp_rest = best['bp_rest_features']
    walk = best['walk_features']
    pedo = best['pedo']
    features.extend(rest['mpower'])
    features.extend(rest['moments'])
    features.extend(rest['entropy'])
    features.extend(rest['fourier'])
    features.extend(rest['tkeo'])
    features.extend(rest['area'])
    features.extend(rest['dynamic'])
    features.extend(rest['info_dynamic'])
    features.extend(rest['hjorth'])
    features.extend(bp_rest['moments'])
    features.extend(bp_rest['entropy'])
    features.extend(bp_rest['fourier'])
    features.extend(bp_rest['tkeo'])
    features.extend(bp_rest['area'])
    features.extend(bp_rest['dynamic'])
    features.extend(bp_rest['info_dynamic'])
    features.extend(bp_rest['hjorth'])
    features.extend(walk['mpower'])
    features.extend(walk['moments'])
    features.extend(walk['entropy'])
    features.extend(walk['fourier'])
    features.extend(walk['tkeo'])
    features.extend(walk['area'])
    features.extend(walk['dynamic'])
    features.extend(walk['info_dynamic'])
    features.extend(walk['hjorth'])
    features.extend(pedo)

    def tryFloat(val):
        try:
            return float(val)
        except:
            return math.nan

    features = list([tryFloat(f) for f in features])
    return [features]

def walk_all_demo(pat):
    features = []
    features.extend(walk_all(pat)[0])
    features.extend(demographics_features(pat))
    return [features]

def from_file(file_name, num_features, fsel):
    with open("features/" + file_name + '.pickle','rb') as f:
        features = pickle.load(f)
        return lambda pat: [f[features[:num_features]]  for f in fsel(pat)]

def from_file_names(fname, num_features):
    with open("features/" + fname + '.pickle','rb') as f:
        features = pickle.load(f)
        return np.array(talk_all_features_names())[features[:num_features]]
