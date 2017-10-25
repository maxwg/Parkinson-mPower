"""
dataExtractor.py
A congolmeration of tools to work with the mPower data.

This file has become very convoluted with the iterative development
approach of the project.
"""

from synapse_helper import *
import gridsearch_svm
import pickle
import thread_helper
import opensmile
from queue import Queue
from collections import defaultdict
import voicetoolbox_helper
import time
import gc
import numpy as np
from cross_validate import cross_validate
from feature_subset import *
from feature_selection import *
from patient_selectors import *
from gridsearch_svm import gridsearch_svm
import random
import math
from ml_models import *
from sklearn_modelselector import getBestModelHyperopt
from keras_nets import *
from audio_helpers import *
from sklearn.model_selection import train_test_split

def restoreWAVAudio():
    with open('talk_wav.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
        return pickle.load(f)

def hasSpeechDifficulties(updrs_list, demog_list):
    """
    MDS-UPDRS2.1: Over the past week, have you had problems with your speech?
    one of: {"Normal", "Slight", "Mild", "Moderate", "Severe"} mapping to {0, 1, 2, 3, 4}

    There is an issue where "Normal" is the default, and a number of participants leave it
    at that. We forcibly filter
    'MDS-UPDRS2.10'
    'MDS-UPDRS2.12'
    'MDS-UPDRS2.13'
    """
    isNotAnsweringMDSProperly = all([all([u[updrs] == 0 for updrs in u if updrs.startswith("MDS-UPDRS2")]) for u in updrs_list])
    isAdvancedStage = max([ sum([u['MDS-UPDRS2.10'], u['MDS-UPDRS2.12'], u['MDS-UPDRS2.13']]) for u in updrs_list]) > 2
    isOldDiagnosis = all([u["diagnosis-year"] < 2013 for u in demog_list])
    return len(updrs_list) == 0 or len(demog_list) == 0 or isOldDiagnosis or isAdvancedStage or isNotAnsweringMDSProperly or not all([u['MDS-UPDRS2.1'] == 0 for u in updrs_list])

def hasPD(demog_list):
    """
    Determine if patient has PD from demographics data
    Currently ignores self diagnosis - only professional diagnosis considered.
    """
    isPD = len(demog_list) > 0 and any([d['professional-diagnosis'] for d in demog_list]) or all([u["diagnosis-year"] < 2020 for u in demog_list])
    return 1 if isPD else 0

def joinPatientsTalkUPRDS(talk, updrs, demog):
    """ Convert contents of talk to a dictionary
        talk:: recordId :-> talk details
        patient:: patients[healthCode]["talk"] :-> [talk]
        patient:: patients[healthCode]["updrs"] :-> updrs
        patient:: patients[healthCode]["isPD"] :-> bool
        patient:: patients[healthCode]["hasSpeechDisorder"] :-> bool
    """
    tlk = talk.to_dict(orient='records')
    talk = {}
    for t in tlk:
        talk[t["recordId"]] = t

    patients = {}
    patients = defaultdict(dict)
    for t in tlk:
        if "talk" not in patients[t['healthCode']]:
            patients[t['healthCode']]['talk'] = []
        patients[t['healthCode']]['talk'].append(t)

    demog = demog.to_dict(orient='records')
    for d in demog:
        if "demog" not in patients[d['healthCode']]:
            patients[d['healthCode']]['demog'] = []
        patients[d['healthCode']]["demog"].append(d)
        patients[d['healthCode']]["isPD"] = hasPD(patients[d['healthCode']]["demog"])

    updrs = updrs.to_dict(orient='records')
    for u in updrs:
        if "updrs" not in patients[u['healthCode']]:
            patients[u['healthCode']]['updrs'] = []
        patients[u['healthCode']]["updrs"].append(u)
        if "demog" not in patients[u['healthCode']]:
            patients[u['healthCode']]['hasSpeechDisorder'] = True
        else:    
            patients[u['healthCode']]["hasSpeechDisorder"] = hasSpeechDifficulties(patients[u['healthCode']]["updrs"], patients[u['healthCode']]["demog"])

    with open('patients.pickle', 'wb') as f:  # Python 3: open(..., 'rb')
        pickle.dump(patients, f)

def patientsArrayToDict(patients):
    ret = {}
    for pat in patients:
        ret[pat["healthCode"]] = pat
    return ret

def rerun_UPDRS(patients, save=True):
    """
    Discovered issues with UPDRS detail extraction post-processing.
    This script refreshes the detail.
    You should not need to run this anymore.
    """
    if type(patients) is dict:
        patients = list(patients.values())
    retpats = []
    raceidx = {'"South Asian"': 0, '"White or Caucasian"': 1, '"Mixed"': 2, '"Middle Eastern"': 3, '"East Asian"': 4, '"Latino/Hispanic"': 5, '"Black or African"': 6, '"Caribbean"': 7, '"Native American"': 8, '"Other"': 9, '"Pacific Islander"': 10}
    for pat in patients:
        if isExcluded(pat["best_audio"]) or ("valid" in pat and pat["valid"] == 0):
            continue

        pat["healthCode"] = pat["best_audio"]["healthCode"]
        pat["hasSpeechDisorder"] = True if "updrs" not in pat else hasSpeechDifficulties(pat["updrs"], pat["demog"])
        pat["isPD"] = None if "demog" not in pat else hasPD(pat["demog"])
        pat["toolbox"] = [float(num) for num in pat["toolbox"]]
        pat["best_audio"]["eGeneva"] = [None] + [float(num) for num in pat["best_audio"]["eGeneva"][1:]]

        age = pat["demog"][len(pat["demog"]) - 1 ]['age']
        phone = pat["demog"][len(pat["demog"]) - 1 ]['phoneInfo']
        gender = pat["demog"][len(pat["demog"]) - 1 ]['gender']
        race = pat["demog"][len(pat["demog"]) - 1 ]['race']

        phoneModel = None
        if phone.startswith("iPhone 7"):
            phoneModel = 0
        elif phone == "iPhone 6 Plus":
            phoneModel = 0
        elif phone == "iPhone 6":
            phoneModel = 1
        elif phone.startswith("iPhone 5s"):
            phoneModel = 2
        elif phone.startswith("iPhone 5c"):
            phoneModel = 3
        elif phone.startswith("iPhone 5"):
            phoneModel = 4
        elif phone.startswith("iPhone 4S"):
            phoneModel = 5

        pat["phone"] = phoneModel
        pat["age"] = age
        pat["gender"] = 1 if gender == "Male" else 0

        if (gender == "Male" or gender == "Female") and phoneModel != None and isinstance(race, str) and not math.isnan(age):
            #Only consider M/F due to lack of data
            racearr = [0] * 11
            race = race.split(',')
            for r in race:
                racearr[raceidx[r]] = 1
            pat["race"] = racearr
            retpats.append(pat)
    if save:
        with open('patients_features_final.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
            pickle.dump(retpats, f)

    print(len(retpats))
    return retpats

def restorePatientsDict():
    with open('patients.pickle', 'rb') as f:
        return pickle.load(f)

def filterTalk(patients):
    return {k: v for k, v in patients.items() if 'talk' in v and len(v["talk"]) > 0}

def filterDemog(patients):
    return {k: v for k, v in patients.items() if 'demog' in v and len(v['demog']) > 0}

def filterUPDRS(patients):
    return {k: v for k, v in patients.items() if 'updrs' in v and len(['updrs']) > 0}

def filterUsable(patients):
    return {k: v for k, v in patients.items() if 'best_audio' in v and len(['best_audio']) > 0}

def getGenevaEnergyFeatures(patients):
    """ Appends short time energy as talk[record]["energy"] = [float]
        Appends eGeneva attributes as talk[eGeneva]
    """
    patients = filterTalk(filterDemog(patients)) #Save processing by only considering those with a known PD diagnosis
    talks = {}
    for p in patients:
        pat = patients[p]
        for tlk in pat["talk"]:
            talks[tlk["recordId"]] = tlk

    features = Queue() #Threadsafe Queue object to store resuls.

    def getFeatures(group, thread):
        idx = 0
        for i, talk in group.items():
            idx += 1
            with open("log", 'a') as log:
                log.write(str(thread) + "|" + talk["countdown"] + "|" + talk["audio"] + "\n")
            tk_ste = opensmile.getShortTimeEnergy(talk["audio"], "output" + str(thread))
            cd_ste = opensmile.getShortTimeEnergy(talk["countdown"], "output" + str(thread))
            (gen_head, gen) = opensmile.getGenevaExtended(talk["audio"], "output" + str(thread))
            features.put((talk["recordId"], tk_ste, cd_ste, gen))
            print(idx,"of",len(group))

    thread_helper.processIterableInThreads(talks, getFeatures, 8)
    while not features.empty():
        k, tk_ste, cd_ste, gen  = features.get()
        talks[k]["energy"] = tk_ste
        talks[k]["cd_energy"] = cd_ste
        talks[k]["eGeneva"] = gen

    with open('patients_opensmile_features.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
        pickle.dump(patients, f)

def getComParEFeaturesSegments(patients):
    """
        Appends eGeneva attributes as talk[ComParE]
    """
    patients = restorePatientsFinal()
    patients = patientsArrayToDict(patients)
    features = Queue() #Threadsafe Queue object to store results.

    def procThread(group, thread_id):
        idx = 0
        for p in group:
            idx += 1
            print(idx, "of", len(group))
            pat = group[p]
            pat['ComParE'] = []
            paths = convertAudioToSegments(pat['best_audio']['audio'][:-4], thread_id)
            segments = []
            for path in paths:
                print(idx, path)
                res = opensmile.getComParE(path, 'output' + str(thread_id))[1]
                segments.append(res)

            features.put((pat["best_audio"]["healthCode"], segments))

    thread_helper.processIterableInThreads(patients, procThread, 8)

    while not features.empty():
        k, segments= features.get()
        patients[k]['ComParE'] = segments

    with open('patients_compare_features.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
        pickle.dump(patients, f)

def restorePatientsCompareFeatures():
    with open('patients_compare_features.pickle', 'rb') as f:  # Python 4: open(..., 'rb')
        return pickle.load(f)


def restorePatientsOpenSmileFeatures():
    with open('patients_opensmile_features.pickle', 'rb') as f:
        return pickle.load(f)

def isExcluded(talk_sample):
    if "cd_energy" not in talk_sample or "energy" not in talk_sample:
        return True
    cd = talk_sample["cd_energy"]
    tk = talk_sample["energy"]
    if cd is None or tk is None:
        return True
    cd = cd[1:-1]
    tk = tk[1:-1]
    return opensmile.getVariance(tk, normalize=True) > 0.3 or opensmile.audioFileIsWeird(tk) or opensmile.noiseExceedsLimit(cd, 0.05) > 20

def rateAudioFiles(talk_sample):
    cd = talk_sample["cd_energy"][1:-1]
    tk = talk_sample["energy"][1:-1]
    #print(opensmile.getVariance(cd), opensmile.getVariance(tk), opensmile.getMeanVolume(tk))
    rank = -opensmile.getVariance(cd) - opensmile.getVariance(tk) + (opensmile.getMeanVolume(tk)/10)
    return rank

def getBestAudioSamples(patients):
    for p in patients:
        pat = patients[p]
        usable = []
        for talk in pat["talk"]:
            if not isExcluded(talk):
                usable.append((rateAudioFiles(talk), talk))
        if len(usable) == 0:
            pat["talk_usable"] = False
            continue

        pat["talk_usable"] = True
        usable.sort(key = lambda i: -i[0])
        pat["best_audio"] = usable[0][1]

def getVoiceToolboxFeatures(patients):
    """
        Calculates voice toolbox features (expensive)
    """
    patients = filterUsable(filterTalk(filterDemog(patients))) #Save processing by only considering those with a known PD diagnosis
    print(len(patients))
    talks = []
    for p in patients:
        talks.append(patients[p]['best_audio'])

    features = Queue() #Threadsafe Queue object to store resuls.

    def getFeatures(group, thread):
        eng = voicetoolbox_helper.newEngine()
        idx = 0
        def attemptExtractFeatures(talk, eng, features):
            try:
                with open("log", 'a') as log:
                    log.write(str(thread) + "|" + "|" + talk["audio"] + "\n")
                (val, name, f0) = voicetoolbox_helper.getAudioFeatures(talk['audio'], eng, "male" if patients[talk['healthCode']]['gender'] == 1 else "female")
                features.put((talk["healthCode"], val, f0))
                return True
            except:
                with open("error_log", 'a') as elog:
                    elog.write(str(thread) + " - MATLAB - " + talk['audio'] + "\n")
                return False

        for talk in group:
            idx += 1
            print(idx, "of", len(group))
            tries = 0
            while(tries < 10):
                tries += 1
                gc.collect()
                if attemptExtractFeatures(talk, eng, features):
                    break
                else:
                    #reset matlab engine
                    try:
                        eng.quit()
                    except:
                        """"""
                    time.sleep(tries * 10)
                    eng = voicetoolbox_helper.newEngine()


            if idx % 20 == 0:
                # restart matlab to fix memory leak
                eng.quit()
                time.sleep(20)
                eng = voicetoolbox_helper.newEngine()

        eng.quit()

    thread_helper.processIterableInThreads(talks, getFeatures, 7)
    while not features.empty():
        k, val, f0 = features.get()
        patients[k]["toolbox"] = val
        patients[k]["f0"] = f0

    for p in patients:
        pat = patients[p]
        pat["toolbox"] = list(pat["toolbox"][0])
        pat["f0"] = [f[0] for f in list(pat["f0"])]

    with open('patients_features.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
        pickle.dump(patients, f)

def getAndProcessAudioSegments():
    patients = restorePatientsFinal()
    patients = patientsArrayToDict(patients)
    features = Queue() #Threadsafe Queue object to store results.

    def procThread(group, thread_id):
        idx = 0
        eng = voicetoolbox_helper.newEngine()

        def attemptExtractFeatures(talk, eng, gender):
            try:
                with open("log", 'a') as log:
                    log.write(str(thread_id) + "|" + "|" + talk + "\n")
                if gender == 1: #male
                    (val, name, f0) = voicetoolbox_helper.getAudioFeatures(talk, eng, "male")
                else:
                    (val, name, f0) = voicetoolbox_helper.getAudioFeatures(talk, eng, "female")
                return val, f0
            except:
                with open("error_log", 'a') as elog:
                    elog.write(str(thread_id) + " - MATLAB - " + talk + "\n")
                print("error with", talk)
                return False

        for p in group:
            idx += 1
            print(idx, "of", len(group))
            pat = group[p]
            pat['audio_segs'] = []
            paths = convertAudioToSegments(pat['best_audio']['audio'][:-4], thread_id)
            segments = []
            for path in paths:
                print(idx, path)
                tries = 0
                while (tries < 10):
                    tries += 1
                    gc.collect()
                    toolbox = attemptExtractFeatures(path, eng, pat['gender'])

                    if toolbox is False:
                        # reset matlab engine
                        try:
                            eng.quit()
                        except:
                            """"""
                        time.sleep(tries * 3)
                        eng = voicetoolbox_helper.newEngine()
                    else:
                        val, f0 = toolbox
                        eGeneva = opensmile.getGenevaExtended(path, 'output' + str(thread_id))[1]
                        res = {}
                        res['eGeneva'] = eGeneva
                        res['toolbox'] = list(val[0])
                        res['f0'] = [f[0] for f in list(f0)]
                        segments.append(res)

            features.put((pat["best_audio"]["healthCode"], segments))
            if idx % 30 == 0:
                # restart matlab to fix memory leak
                eng.quit()
                time.sleep(15)
                eng = voicetoolbox_helper.newEngine()
                try:
                    with open('audio_segments_bak' +str(thread_id)+ '.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
                        pickle.dump(features, f)
                except:
                    """"""
    thread_helper.processIterableInThreads(patients, procThread, 7)

    while not features.empty():
        k, segments= features.get()
        patients[k]['audio_segs'] = segments

    with open('patients_audio_segments.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
        pickle.dump(patients, f)

def restorePatientsAllFeatures():
    with open('patients_features.pickle', 'rb') as f:
        return pickle.load(f)

def restorePatientsFinal():
    with open('patients_features_final.pickle', 'rb') as f:
        return pickle.load(f)

def getBestLinearPredictors(patients):
    fnames = all_features_with_duplicates_names()
    tasks = [(idx, fname) for idx, fname in enumerate(fnames)]
    results = Queue()
    def evaluateLinearPredictor(tasks, threadId):
        for idx, fname in tasks:
            fsel = lambda pat: np.array([all_features_with_duplicates(pat)[idx]])
            gnb = naive_bayes.GaussianNB()
            res = cross_validate(patients, selectAllPatients(patients), 50, fsel, gnb, proportions= "equal")[0] #maximise accuracy
            print(threadId, res, idx, fname)
            results.put((res, fname, idx))
    thread_helper.processIterableInThreads(tasks, evaluateLinearPredictor, 8)
    results = list(results.queue)
    with open('linear_predictors.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
        pickle.dump(results, f)

def getTrainTestSplit(patients, num_test, onlyPD):
    train = []
    test = []
    if type(patients) is dict:
        patients = list(patients.values())
        patients.sort(key = lambda pat: pat["healthCode"])

    ratio = len([p for p in patients if p['isPD']])/len(patients)
    print(ratio)
    curPD, curC = 0,0
    random.seed(2)
    random.shuffle(patients)
    for pat in patients:
        if len(test)<num_test and ((pat["isPD"] and not pat["hasSpeechDisorder"]) if onlyPD else True):
            if onlyPD or pat['isPD'] and (curPD == 0 or curPD/(curPD+curC) <= ratio): # Stratify test set, leaving it at max one off.
                test.append(pat)
            else:
                train.append(pat)
        else:
            train.append(pat)
    return train, test

def getFeatureStats(patients, feature):
    female = [pat for pat in patients if pat["gender"] == 0 ]
    male= [pat for pat in patients if pat["gender"] == 1 ]

    isPD = []
    noPD = []
    for pat in female:
        ppe = talk_all_features(pat)[talk_all_features_names().index(feature)]
        if pat["isPD"]:
            isPD.append(ppe)
        else:
            noPD.append(ppe)
    print(feature, "Female PD",np.mean(isPD), np.std(isPD))
    print(feature, "Female Control",np.mean(noPD), np.std(noPD))

    isPD = []
    noPD = []
    for pat in male:
        ppe = talk_all_features(pat)[talk_all_features_names().index(feature)]
        if pat["isPD"]:
            isPD.append(ppe)
        else:
            noPD.append(ppe)
    print(feature, "Male PD",np.mean(isPD), np.std(isPD))
    print(feature, "Male Control",np.mean(noPD), np.std(noPD))


def mergeUPDRSrerunWithOld():
    """
    I realise a major flaw with the code is that rerun_UPDRS filters patients labelled
    valid = False out of the pickle.
    This fixes it.
    """
    patients_old = patientsArrayToDict(rerun_UPDRS(restorePatientsAllFeatures(), save=False))
    patients_updrs = patientsArrayToDict(restorePatientsFinal())
    unknown = 0
    valid = 0
    invalid = 0
    for p in patients_old:
        if p in patients_updrs:
            patients_old[p] = patients_updrs[p]
            if 'valid' in patients_old[p]:
                valid += 1
            else:
                unknown += 1
        else:
            patients_old[p]['valid'] = False
            invalid += 1

    print(valid,invalid,unknown)
    with open('patients_final_with_invalid.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
        pickle.dump(patients_old, f)

    return patients_old

def filterValidOnly(patients):
    rtn = []
    for pat in patients:
        if 'valid' in pat and pat['valid']:
            rtn.append(pat)
    return rtn

def saveNorm(patients, filename):
    X, y = patientsToXy(patients)
    norm = [np.mean(X, axis=0), np.std(X, axis=0)]
    with open(filename + '.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
        pickle.dump(norm, f)

def joinWalkWithFilesAndPatients(walk, walks, patients):
    headers = list(walk.columns.values)
    walk_matrix = walk.as_matrix()

    def procThread(group, thread_id, queue):
        import path_processors
        count = 0
        for w in group:
            print(thread_id, count, "of", group.shape[0])
            count += 1
            w = {h: wk for wk, h in zip(w, headers)}
            for h in w:
                if "json" in h:
                    if (math.isnan(w[h])):
                        continue
                    w[h] = walks[str(int(w[h]))]

            healthCode = w['healthCode']
            if isinstance(w['deviceMotion_walking_rest.json.items'], str) and isinstance(
                    w['pedometer_walking_outbound.json.items'], str):
                features, errors, accel = path_processors.getAllFeaturesRest(w['deviceMotion_walking_rest.json.items'], highPass=False)
                if errors >= 1000:
                    print("ERR Rest")
                    continue
                bpfeatures, bperrors, bpaccel = path_processors.getAllFeaturesRest(w['deviceMotion_walking_rest.json.items'],
                                                                   highPass=True)
                if bperrors >= 1000:
                    print("ERR Rest BP")
                    continue
                walkfeatures, walkerrors, walkaccel = path_processors.getAllFeaturesWalking(
                    w['deviceMotion_walking_outbound.json.items'])
                if walkerrors >= 1000:
                    print("ERR Walk")
                    continue

                pedo = path_processors.getPedometerFeatures(w['pedometer_walking_outbound.json.items'])
                print(pedo)

                w['rest_features'] = features
                w['bp_rest_features'] = bpfeatures
                w['walk_features'] = walkfeatures
                w['pedo'] = pedo
                w['errors'] = errors
                w['rest_accel'] = accel
                w['walk_accel'] = accel
                queue.put((healthCode, w))

    extracted_features = thread_helper.processIterableInProcesses(walk_matrix, procThread, 6)

    while not extracted_features.empty():
        k, features = extracted_features.get()
        if k in patients:
            if "walks" not in patients[k]:
                patients[k]["walks"] = []
            patients[k]["walks"].append(features)

    with open('patients_final_with_walk.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
        pickle.dump(patients, f)

    return patients

def restorePatientsWalk():
    with open('patients_final_with_walk.pickle', 'rb') as f:  # Python 4: open(..., 'rb')
        return pickle.load(f)


def filterTalkAndWalkAvailable(patients):
    ret = []
    for p in patients:
        pat = patients[p]
        if "best_audio" in pat and "walks" in pat:# and any([not (w['bp_rest_features'] is None) for w in pat['walks']]):
            ret.append(pat)
    return ret

medStates = ["I don't take Parkinson medications", 'Immediately before Parkinson medication','Another time', 'Just after Parkinson medication (at your best)']
def isMedicated(pat):
    if pat['medTimepoint'] == "I don't take Parkinson medications" or pat['medTimepoint'] == 'Immediately before Parkinson medication' or pat['medTimepoint'] == 'Another time':
        return False
    elif pat['medTimepoint'] =='Just after Parkinson medication (at your best)':
        return True
    elif math.isnan(pat['medTimepoint']):
        return None
    else:
        raise Exception("Invalid Medication State!")

def getBestMedicatedAudio(patients):
    """
    Adds a ['med_audio'] field to patients which contains the
    best medicated audio as rated by rateAudioFiles()
    
    :param patients: a list of patients
    :return: a list of patients with the ['med_audio'] field.
    """
    for pat in patients:
        if not pat['isPD']:
            pat['med_audio'] = None
        else:
            usable = []
            for talk in pat["talk"]:
                if not isExcluded(talk) and isMedicated(talk) == True:
                    usable.append((rateAudioFiles(talk), talk))
            if len(usable) == 0:
                pat["med_audio"] = None
                continue
            usable.sort(key = lambda i: -i[0])
            pat["med_audio"] = usable[0][1]

def listenToSpeechFiles(patients):
    """
    Plays patients best speech samples while printing out
    a parameter
    
    :param patients: Input of patients to be played
    """
    import os
    print(len(patients))
    for pat in patients:
        os.system("mplayer " + pat["best_audio"]["audio"])
        print(pat["best_audio"]["energy"])
        break

def manuallyLabelSpeechFiles(patients, duration=1.5):
    """
    Plays speech samples and prompts the user
    to label them as either valid or invalid.
    
    :param patients: Input of patients to be labelled
           duration: (float): Length of voice to play for each subject
    """
    import os
    i =0
    patientsordered = []
    for pat in patients:
        patientsordered.append((rateAudioFiles(pat["best_audio"]), pat))
    patientsordered.sort()

    for pat in patientsordered:
        pat = pat[1]
        if "valid" not in pat and not pat["hasSpeechDisorder"] and pat["isPD"]:
            os.system("mplayer -ss 2 -endpos " + str(duration) + " " + pat["best_audio"]["audio"])
            isusable = None
            while isusable != "1" and isusable != "0":
                isusable = input("1 = good, 0 = bad")
            isusable = int(isusable)
            pat["valid"] = isusable
            i += 1
            if i % 10 == 0:
                print("saving to disk...")
                with open('patients_features_labelled.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
                    pickle.dump(patients, f)

def getSpeechMFCCandDynamical(patients):
    import python_speech_features as psf
    import dynamical_speech_features as dsf
    from short_time_fourier import getShortTimeFourier1D
    total = len(patients)
    i = 0
    for p in patients:
        pat = patients[p]
        print(i, "of", total)
        samprate, audio = loadAudioAsArray(pat['best_audio']['audio'])
        print(len(audio))
        # mfcc = psf.mfcc(audio, samprate, nfilt=40, winlen=0.02, numcep=20, appendEnergy=True)
        # fourier = getShortTimeFourier1D(audio, 2048)
        # fbank = psf.logfbank(audio, samprate, nfilt=40)
        dynamic = dsf.getDynamicalSpeechFeatures(audio, samprate)
        # pat['best_audio']['fbank'] = fbank
        pat['best_audio']['mfcc'] = None
        pat['best_audio']['fourier'] = None
        pat['best_audio']['dynamic'] = dynamic
        # pat['best_audio']['raw'] = audio[1024:1024+samprate]
        i+=1

    with open('patients_mfcc.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
        pickle.dump(patients, f)


def restorePatientsMFCC():
    with open('patients_mfcc.pickle', 'rb') as f:  # Python 4: open(..., 'rb')
        return pickle.load(f)

def fixSpeechMFCCandDynamical():
    patients = restorePatientsMFCC()
    import dynamical_speech_features as dsf
    total = len(patients)
    i = 0
    for p in patients:
        pat = patients[p]
        print(i, "of", total)
        samprate, audio = loadAudioAsArray(pat['best_audio']['audio'])
        dynamic = dsf.fixDynamicalSpeechFeatures(audio, pat['best_audio']['dynamic'])
        pat['best_audio']['dynamic'] = dynamic
        i+=1

    with open('patients_mfcc.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
        pickle.dump(patients, f)

def mergeMFCCFeatures(patients):
    mfcc = restorePatientsMFCC()
    for p in patients:
        patients[p]['best_audio'] = mfcc[p]["best_audio"]
        patients[p]['audio_segs'] = mfcc[p]["audio_segs"]

    with open('patients_mfcc_merged.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
        pickle.dump(patients, f)

def restoreMergedMFCCWalk():
    with open('patients_mfcc_merged.pickle', 'rb') as f:  # Python 4: open(..., 'rb')
        return pickle.load(f)

def restorePatientsSegments():
    with open('patients_audio_segments.pickle', 'rb') as f:  # Python 4: open(..., 'rb')
        return pickle.load(f)

def getDynamicAudioFeatures(patients):
    import dynamical_speech_features as dsf
    results = []
    count = 0
    for p in patients:
        print(count,"of",len(patients))
        count += 1
        pat = patients[p]
        samprate, audio = loadAudioAsArray(pat['best_audio']['audio'])
        results.append((p, dsf.getDynamicalSpeechFeatures(audio, samprate)))

    with open("patients_dynamic_speech.pickle", "wb") as f:
        pickle.dump(results, f)


def fixSegmentFeatures():
    """ I made a major mistake with some of the segment
        features and the same observations had been redundantly
        extracted 10 times.
    """
    with open('patients_mfcc_merged.pickle', 'rb') as f:  # Python 4: open(..., 'rb')
        segs = pickle.load(f)
        i = 0
        for p in segs:
            pat = segs[p]
            pat['audio_segs'] = [pat['audio_segs'][i] for i in range(0, 69, 10)]
            print(1,i)
            i+=1
            
        with open('patients_mfcc_fixed.pickle', 'wb') as f2:  # Python 4: open(..., 'rb')
            pickle.dump(segs, f2)
        del segs

def loadSegmentFeaturesFixed():
    with open('patients_mfcc_fixed.pickle', 'rb') as f2:  # Python 4: open(..., 'rb')
        return pickle.load(f2)

def fixCompareFeatures():
    with open('patients_compare_features.pickle', 'rb') as f:  # Python 4: open(..., 'rb')
        segs = pickle.load(f)
        i = 0
        for p in segs:
            pat = segs[p]
            pat['ComParE'] = [list(map(float, comp)) for comp in pat['ComParE']]
            print(2,i)
            i+=1
        with open('patients_compare_features_fixed.pickle', 'wb') as f2:  # Python 4: open(..., 'rb')
            pickle.dump(segs, f2)

def mergeCompareSegments():
    with open('patients_audio_segments.pickle', 'rb') as f:  # Python 4: open(..., 'rb')
        segs = pickle.load(f)
        with open('patients_compare_features.pickle', 'rb') as f2:  # Python 4: open(..., 'rb')
            compare = pickle.load(f2)
            for p in segs:
                pat = segs[p]
                pat['ComParE'] = compare[p]['ComParE']
                print(p, len(pat['ComParE']), len(pat['audio_segs']))
            with open('patients_all_segments.pickle', 'rb') as f2:  # Python 4: open(..., 'rb')
                pickle.dump(segs, f2)

def count_PD_ratio(patients):
    pd = 0
    if type(patients) is dict:
        patients = list(patients.values())

    print("Total:", len(patients))
    print("PD:", len([p for p in patients if p["isPD"]]))


def init():
    # getAndDownloadSynapse()
    #tap, taps, walk, walks, talk, talks, mem, mems, pdq8, demog, updrs = restoreSynapseTables()
    # convertAudioToWav(talk, talks)
    #talk = restoreWAVAudio()
    #joinPatientsTalkUPRDS(talk, updrs, demog)

    #patients = restorePatientsDict()
    #getGenevaEnergyFeatures(patients)
    # patients = restorePatientsOpenSmileFeatures()
    # getBestAudioSamples(patients)
    # getVoiceToolboxFeatures(patients)
    # patients = restorePatientsAllFeatures())
    # patients = restorePatientsFinal()
    # patients = rerun_UPDRS(patients)
    # patients = restorePatientsWalk()
    # patients = restorePatientsMFCC()
    # patients = restorePatientsSegments()
    patients = restoreMergedMFCCWalk()
    return patients

def doStuff(patients):
    # mergeMFCCFeatures(patients)
    # getSpeechMFCCandDynamical(patients)
    fixSpeechMFCCandDynamical()
    # getAndDownloadSynapse()
    # fixSegmentFeatures()
    #mergeCompareSegments()
    # tap, taps, walk, walks, talk, talks, mem, mems, pdq8, demog, updrs = restoreSynapseTables()
    # patients = joinWalkWithFilesAndPatients(walk, walks, patients)
    # patients = filterTalkAndWalkAvailable(patients)
    # for pat in patients:
    #     # pat = patients[p]
    #     print(pat['audio_segs'])
    #     break
    # getComParEFeaturesSegments(patients)
    # patients = rerun_UPDRS(patients)
    # updrs = [p for p in patients if "updrs" in patients[p] and len(patients[p]['updrs']) > 0 and patients[p]['isPD']]
    # print(len(updrs))
    # count_PD_ratio(patients)
    # train, test = getTrainTestSplit(patients, 50, True)
    #
    # model = lambda : basicDenseNN_small(389)
    # eval_fn = lambda model: cross_validate(patients, selectAllPatients(patients), 10, walk_all, model, probability=True, proportions='equal',
    #        nan_values='mean', isNN=False, epochs=200, nn_batch_size=10000)

    # model = lambda: svm.SVC(kernel='rbf', gamma=1e-4, C=1000, probability=True)
    # results = eval_fn(model)
    # #
    # np.savetxt("results/speech_dynamic_only", results, fmt='%.10f')
    #
    # print(np.mean(results, axis=0))
    # print(np.std(results, axis=0))


    # eval_fn = lambda model: cross_validate(patients, selectAllPatients(patients), 5, walk_all_normed, model, probability=True, n_repeats=1, proportions='equal',
    #        nan_values='mean', isNN=False, epochs=200, nn_batch_size=10000)
    #
    # print(gridsearch_svm(eval_fn, 0))
    # female_train = [pat for pat in train if pat["gender"] == 0 ]
    # male_train= [pat for pat in train if pat["gender"] == 1 ]
    # female_test = [pat for pat in test if pat["gender"] == 0 ]
    # male_test = [pat for pat in test if pat["gender"] == 1 ]
    # fsel_nogender= all_features_without(["gender"])
    # fsel_nogenderage= all_features_without(["gender", "age"])
    # fsel_noage= all_features_without(["age"])

    # patients = getBestMedicatedAudio(patients)
    # meds = []
    # res = [0,0]
    # for pat in patients:
    #     medic = pat['med_audio']
    #     if medic is None:
    #         res[0] +=1
    #     elif medic is True:
    #         res[1] +=1
    # print(res)


    # trainAndTestNN(train, basicLSTM, fsel=speech_MFCC, test=test)
    #Transfer Learning
    # getBestNeuralNetAfterNEvals(train, nnetFinalModel1_balanced, fsel=all_features_without_demo, test=test, evals=15, min_acc=0.56, measure="fmeasure")
    # nn = loadNNFromFile("best0.mdl")
    # transferLearn("goodfemalenodemo.mdl", male_train, male_test, fsel=all_features_without_demo)
    # evalNN(nn, train, test, fsel=all_features_without_demo)
    # svm = svmDefault()
    # print(cross_validate(male_train, selectAllPatients(male_train), 5, all_features_without(["gender"]), svm))
    # print(cross_validate(female_train, selectAllPatients(female_train), 5, all_features_without(["gender"]), svm))
    # hyperasNnetSearch(nnetDenseHyperas5, female_train, evals=100)
    # hyperasNnetSearch(nnetDenseHyperas4, train, evals=150)
    #hyperasNnetSearch(nnetDenseHyperas5, male_train, evals=100)


    # getFeatureStats(patients, "DFA")
    # getFeatureStats(patients, "Jitter->F0_abs_dif")
    # svm = svmDefault()
    # print(cross_validate(patients, selectAllPatients(patients), 5, age_only, svm))
    # print(cross_validate(patients, selectAllPatients(patients), 5, all_features, svm))
    # getBestLinearPredictors(patients)
    #getFSResults(train, fs_MRMR, "mrmrwk", fsel=walk_all)
    #getFSResults(train, fs_CIFE, "cifewk", fsel=walk_all)
    #getFSResults(train, fs_ICAP, "icapwk", fsel=walk_all)
    #getFSResults(train, fs_JMI, "jmiwk", fsel=walk_all)
    #getFSResults(train, fs_MIFS, "mifswk", fsel=walk_all)
    # getFSWithoutFeatureParam(train, fs_reliefF, "relieffwk", fsel=walk_all)
    #getFSWithoutFeatureParam(train, fs_RFS, "rfswk0.0003", True, 0.0003, fsel=walk_all)
    #getFSWithoutFeatureParam(train, fs_RFS, "rfswk0.001", True, 0.001, fsel=walk_all)
    # getFSWithoutFeatureParam(train, fs_RFS, "rfswk0.003", True, 0.003, fsel=walk_all)
    #getFSWithoutFeatureParam(train, fs_RFS, "rfswk0.01", True, 0.01, fsel=walk_all)
    #getFSWithoutFeatureParam(train, fs_RFS, "rfswk0.03", True, 0.03, fsel=walk_all)
    #getFSWithoutFeatureParam(train, fs_RFS, "rfswk0.1", True, 0.1, fsel=walk_all)
    # getFSWithoutFeatureParam(train, fs_RFS, "rfswk0.3", True, 0.3, fsel=walk_all)
    #getFSWithoutFeatureParam(train, fs_RFS, "rfswk1", True, 1, fsel=walk_all)


    # getFSResults(train, fs_svm_backward, "svmbnn", fsel=walk_all)
    # getFSResults(train, fs_svm_forward, "svmfnn", fsel=walk_all)
    # getFSWithoutFeatureParam(train, fs_fisher, "fishernn")
    # getFSWithoutFeatureParam(train, fs_f_score, "fscorenn")
    # getFSWithoutFeatureParam(train, fs_l21, "l21nn0.0003", True, 0.0003)
    # getFSWithoutFeatureParam(train, fs_l21, "l21nn0.001", True, 0.001)
    # getFSWithoutFeatureParam(train, fs_l21, "l21nn0.003", True, 0.003)
    # getFSWithoutFeatureParam(train, fs_l21, "l21nn0.01", True, 0.01)
    # getFSWithoutFeatureParam(train, fs_l21, "l21nn0.03", True, 0.03)
    # getFSWithoutFeatureParam(train, fs_l21, "l21nn0.1", True, 0.1)
    # getFSWithoutFeatureParam(train, fs_l21, "l21nn0.3", True, 0.3)
    # getFSWithoutFeatureParam(train, fs_l21, "l21nn1", True, 1)
    # fs_reliefF(train)
    # nnet = nnetReLU(200, layers=[100, 70, 40, 25, 15])
    # eval_fn = lambda model: cross_validate(patients, selectAllPatients(patients), 5, all_features, model, probability=False, proportions='equal', pca_components = 200, pca_kernel="rbf", isNN= True, train_prediction=True)
    # print(eval_fn(nnet))

    # print(cross_validate_GPy(patients, selectAllPatients(patients), 5, all_features, GPyMatern52Kernel(80),
    #                proportions='equal', pca_components=80, pca_kernel="rbf", train_prediction=True))

    # from keras_nets import hyperasNnetSearch, nnetDenseBigHyperas4
    # hyperasNnetSearch(nnetDenseBigHyperas4, train, evals=100)
    # getBestModelHyperopt(train, max_evals=10, timeout=10)
    # print("matern", eval_fn(gpmatern))
    # print("rbf", eval_fn(gprbf))
    # print("rq", eval_fn(gprq))
    # print(eval_fn(gp))
    # print(gridsearch_svm(eval_fn, 0))
    # print(len(selectNoSpeechDisorderPatients(patients)))

    # svm = svmDefault()
    # print(cross_validate(train, selectAllPatients(train), 5, from_file('svmb300', 23), svm,
    #                 proportions='equal', train_prediction=True))
    # from sklearn.gaussian_process import GaussianProcessClassifier
    # gp = GaussianProcessClassifier()
    # print(cross_validate(train, selectAllPatients(train), 5, from_file('svmb300', 23), gp,
    #                 proportions='equal', train_prediction=True))

    # print(cross_validate_GPy(train, selectAllPatients(train), 5, from_file('svmb300', 23),
    #                      proportions='equal', train_prediction=True))

    #SVMB 23
    # getBestModelHyperopt(train, fsel=from_file('svmb300', 23), max_evals=100, timeout=10)

    #RFS

        #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2858583/
    #getBestModelHyperopt(train, test=test, proportions="equal", fsel=from_file("fisherd", 187), max_evals=100, timeout=10)

    #getBestModelHyperopt(train, fsel=from_file('svmb300', 23), max_evals=100, timeout=10)


    # gp = svmDefault()
    #from keras_nets import genderAgeCombinerNNET1
    #nn = genderAgeCombinerNNET1()
    #print(cross_validate(train, selectAllPatients(train), 5, all_features, nn, feature_selection=[all_features_names().index("PPE"),
                                                                                                  #all_features_names().index(
                                                                                                   #   "gender")],
    #                proportions='equal',epochs=5, train_prediction=True, isNN=True))

    # print(np.mean(isPD))
    # print(np.mean(noPD))



if __name__ == "__main__":
    patients = init()
    doStuff(patients)
