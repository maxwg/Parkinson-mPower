from dataExtractor import *
def getData():
    import python_speech_features as psf
    import dynamical_speech_features as dsf
    from short_time_fourier import getShortTimeFourier1D

    patients = init()

    cv_data = []

    if type(patients) is dict:
        patients = list(patients.values())

    for pat in patients:
        usable = []
        for talk in pat["talk"]:
            if not isExcluded(talk):
                usable.append((rateAudioFiles(talk), talk))
        if len(usable) == 0:
            continue
        usable.sort(key=lambda i: -i[0])
        for i in range(1, len(usable)):
            cv_data.append(usable[i][1])
        break
    total = len(cv_data)
    i = 0
    for talk in cv_data:
        print(i, "of", total)
        samprate, audio = loadAudioAsArray(talk['audio'])
        print(len(audio))
        mfcc = psf.mfcc(audio, samprate, nfilt=40, winlen=0.02, numcep=20, appendEnergy=True)
        fourier = getShortTimeFourier1D(audio, 2048)
        dynamic = dsf.getDynamicalSpeechFeatures(audio, samprate)
        talk['mfcc'] = mfcc
        talk['fourier'] = fourier
        talk['dynamic'] = dynamic
        talk['raw'] = audio[1024:5120]
        i+=1

    with open('patients_audio_ml.pickle', 'wb') as f:  # Python 4: open(..., 'rb')
        pickle.dump(patients, f)