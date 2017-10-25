import dataExtractor
import voicetoolbox_helper
import opensmile

def diagnose(audio_path):
    patients = dataExtractor.init()
    train, test = dataExtractor.getTrainTestSplit(patients, 400, False)
    nn = dataExtractor.loadNNFromFile("goodfinal1nodemo_balanced60.mdl")
    eng = voicetoolbox_helper.newEngine()
    pat = {'healthCode': 0, 'isPD': 0, 'best_audio':{}}
    pat['best_audio']['eGeneva'] = opensmile.getGenevaExtended(audio_path, "output")[1]
    pat['toolbox'] = voicetoolbox_helper.getAudioFeatures(audio_path, eng)
    pat['toolbox'] = [float(v) for v in list(pat['toolbox'][0][0])]
    print(pat['best_audio']['eGeneva'])
    return dataExtractor.predictNN(nn, train, pat, fsel=dataExtractor.all_features_without_demo)
