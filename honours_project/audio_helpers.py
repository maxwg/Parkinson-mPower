"""
    audio_helpers.py

    This file contains helpers to crop audio files.

    This is a very hacky file, delegating to os.system.

    Prereq: mplayer.
"""

import thread_helper
import pickle
import os
import numpy as np

audio_sections = [(1.5, 1.5), (2, 1.5), (2.5,1.5), (3, 1.5),
                  (3.5,1.5), (4, 1.5), (4.5,1.5)]

def convertSingleAudio(source, dest, start, length):
    os.system("mplayer -ss " + start + " -endpos " + length + " -ao pcm:file=" + dest + " " + source+ ">/dev/null 2> /dev/null")

def convertAudioToSegments(audio, thread):
    resultpaths = []
    for start, length in audio_sections:
        s, l = str(start), str(length)
        path = "tmp/output_" + s + "." + l + "-" + str(thread) +".wav"
        resultpaths.append(path)
        convertSingleAudio(audio, path, s, l)
    return resultpaths

def convertAudioToWav(talk, talks, start = "2", length = "4"):
    def replaceIdWithPath(row, header, path_table, end):
        fid = row[header]
        row[header] = path_table[str(fid)] + end

    def convertM4AtoWAV(row, header, cutoff):
        import os
        file = row[header]
        if True or not os.path.isfile(file+".wav"):
            os.system("mplayer -ss "+cutoff+" -endpos "+length+" -ao pcm:file=" + file + ".wav " + file +">/dev/null 2> /dev/null")

    def getIDandReplaceM4a(group, thread):
        idx = 0
        for i, t in group.iterrows():
            replaceIdWithPath(t, 'audio_audio.m4a', talks, '')
            convertM4AtoWAV(t, 'audio_audio.m4a', start)
            replaceIdWithPath(t, 'audio_countdown.m4a', talks, '')
            convertM4AtoWAV(t, 'audio_countdown.m4a', "0")
            print(idx, "of", len(group))
            idx += 1

    thread_helper.processIterableInThreads(talk, getIDandReplaceM4a, 8)
    talk = talk.assign(countdown = lambda col: [talks[str(f)] + ".wav" for f in col["audio_countdown.m4a"]])
    talk = talk.assign(audio = lambda col: [talks[str(f)] + ".wav" for f in col["audio_audio.m4a"]])
    with open('talk_wav.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(talk, f)

def loadAudioAsArray(wavpath, scale='normal', crop = None):
    """

    :param wavpath:
    :param scale:
    :param crop: (start, end) - crop audio file in seconds.
    :return: int samplerate, [float] audio
    """
    from scipy.io.wavfile import read
    wavpath = wavpath.replace("/u5584091/", "/users/u5584091/")
    aud = read(wavpath)
    data = None
    if scale == 'normal':
        data = np.array([a/pow(2,31) for a in aud[1]][100:-100]).reshape(-1, 1)
    if scale == 'int':
        data = np.array([a + pow(2,31)  for a in aud[1]][100:-100]).reshape(-1, 1)

    if not (crop is None):
        data = data[aud[0]*crop[0]:aud[0]*crop[1]]

    return aud[0], data
    # with wave.open(wavpath, 'rb') as wav:
    #     channels = wav.getnchannels()
    #     samplerate = wav.getframerate()
    #     frames = wav.getnframes()
    #     framewidth = wav.getsampwidth()
    #     data = wav.readframes(frames)
    #     print(len(data))
    #     data = struct.unpack("%ih" % (frames * channels), data)
    #     data = [float(val) / pow(2, 31) for val in data]
    #
    #     return {
    #         'channels' : channels,
    #         'rate' : samplerate,
    #         'frames' : frames,
    #         'data' : data
    #     }

# samplerate, data = loadAudioAsArray('dataSamples/audio/audio_audio.wav')

# import python_speech_features as psf
# print(psf.mfcc(data, samplerate, nfilt=50, winlen=0.02, numcep=25, appendEnergy=True).shape)
# print(psf.logfbank(data, samplerate, winlen=0.02, nfilt=100).shape)
# from short_time_fourier import *
# getShortTimeFourier1D(data, 1000)
