import opensmile
import voicetoolbox_helper
import pickle
from keras_nets import *
PMSR_dir = 'Sakar_2012_data/data/'
import dataExtractor

def process():
    results = [{} for _ in range(28)]
    matlab = voicetoolbox_helper.newEngine()
    for i in range(1,29):
        path = PMSR_dir + str(i) + '.wav'
        print(path)
        results[i-1]['id'] = i
        results[i-1]['path'] = path
        results[i-1]['eGeneva'] = opensmile.getGenevaExtended(path, 'output')[1]
        results[i-1]['toolbox'] = voicetoolbox_helper.getAudioFeatures(path, matlab)

    with open('Sakar_2012_data/features.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
        pickle.dump(results, f)

def get_results():
    with open('Sakar_2012_data/features.pickle', 'rb') as f:  # Python 4: open(..., 'rb')
        results = pickle.load(f)
        for i, r in enumerate(results):
            r['toolbox'] = [float(v) for v in list(r['toolbox'][0][0])]
            r['isPD'] = 1
            r['healthCode'] = i+1
            r["best_audio"] = {"eGeneva": r["eGeneva"]}
        nn = loadNNFromFile("goodfinal1nodemo_balanced60.mdl")

        patients = dataExtractor.init()

        evalNN(nn,patients, results, fsel=speech_tsanas_dynamic)
        X, y, hc = patientsToXy(results, fsel = speech_tsanas_dynamic, norm=patients, healthCodes=True)
        pred_prob = np.array(nn.predict(X)).flatten()
        print(list(zip(pred_prob, hc)))

# process()
get_results()
