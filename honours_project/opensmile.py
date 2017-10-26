"""
    opensmile.py

    This file contains various shortcuts for working with the opensmile library.
"""

import subprocess
import csv
from config import SMILE_path, OPENSMILE_log
import numpy as np
import os.path

def getShortTimeEnergy(wav, output):
    if os.path.isfile(wav) and os.path.getsize(wav) > 40000:
        subprocess.Popen("sh "+ SMILE_path + " -C getEnergy.conf -I '" + wav +"' -O " + output, shell=True, stdout=OPENSMILE_log, stderr=OPENSMILE_log).wait()
        energy = []
        with open(output) as energy_out:
            reader = csv.reader(energy_out, delimiter=";")
            for row in reader:
                energy.append(float(row[0]))
        return energy
    return None

def getGenevaExtended(wav, output):
    if os.path.isfile(wav) and os.path.getsize(wav) > 40000:
        subprocess.Popen("rm " + output, shell=True, stdout=OPENSMILE_log, stderr=OPENSMILE_log).wait()
        subprocess.Popen("sh "+ SMILE_path + " -C smile_config/eGeMAPSv01a.conf -I '" + wav +"' -csvoutput " + output, shell=True, stdout=OPENSMILE_log, stderr=OPENSMILE_log).wait()
        try:
            with open(output) as gen:
                res = csv.reader(gen, delimiter=';', quotechar="'")
                res = list(res)
                headers = res[0]
                data = res[1]
                return (headers, data)
        except:
            with open("error_log", 'a') as elog:
                elog.write(wav + "\n")
                return (None, None)
    return (None, None)

def getComParE(wav, output):
    if os.path.isfile(wav) and os.path.getsize(wav) > 40000:
        subprocess.Popen("rm " + output, shell=True, stdout=OPENSMILE_log, stderr=OPENSMILE_log).wait()
        subprocess.Popen("sh "+ SMILE_path + " -C smile_config/ComParE_2016.conf -I '" + wav +"' -csvoutput " + output, shell=True, stdout=OPENSMILE_log, stderr=OPENSMILE_log).wait()
        try:
            with open(output) as gen:
                res = csv.reader(gen, delimiter=';', quotechar="'")
                res = list(res)
                headers = res[0][2:]
                data = res[1][2:]
                return (headers, data)
        except:
            with open("error_log", 'a') as elog:
                elog.write(wav + "\n")
                return (None, None)
    return (None, None)

# print(getComParE('dataSamples/audio/audio_audio.wav', '431')[0])
# print(getGenevaExtended('audio_audio.m4a-6315fd01-f2e1-4a90-b2d2-c765bd1db2b78161460049553722071.tmp.wav', 1))

def noiseExceedsLimit(short_time_energy, threshold):
    """
    Checks if at least 1/N of the sound exceeds a certain limit
    :param wav: path to .wav file
    :param threshold: noise threshold - 0.005 is a good value
    :param output: output file for SMILExtract
    :return: boolean
    """
    exceeded = 0
    for e in short_time_energy:
        if e > threshold:
            exceeded += 1
    return exceeded

def getVariance(short_time_energy, normalize=False):
    ste = np.array(short_time_energy)
    if normalize:
        ste = ste / np.mean(ste)
    if np.var(ste) == 0:
        return float("inf")
    return np.std(ste)

def getMeanVolume(short_time_energy):
    return np.mean(np.array(short_time_energy))

def audioFileIsWeird(short_time_energy):
    ste = np.array(short_time_energy)
    stesort = sorted(short_time_energy)

    if len(stesort) < 10 or (stesort[-2] - stesort[1]) == 0:
        return True

    ste = (ste - stesort[1])/(stesort[-2] - stesort[1]) # normalise but
    if np.mean(ste[10:25]) < 0.05:
        return True

    if stesort[-1]  >  stesort[-10] * 2:
        return True

    return False

def getFirstBreak(short_time_energy, threshold):
    """ Gets the first break in the pronunciation of "aaaah"
        Short Time Energies are intervals of 0.1 seconds.
        TODO
    """
    # sum = 0
    # pmean = short_time_energy[0]
    # for i, e in enumerate(short_time_energy):
    #     sum += e
    #     if i > 10 and abs(e - pmean) > threshold:
    #         return i
    #     pmean = sum/(i+1)
    # return len(short_time_energy)
    ste = np.array(short_time_energy)
    stesort = sorted(short_time_energy)
    if (stesort[-2] - stesort[2]) == 0:
        return True
    ste = (ste - stesort[2])/(stesort[-2] - stesort[2]) # normalise but exclude outliers
    if np.mean(ste[20:30]) < 0.2:
        print(ste)
        return True

eGeneva_feature_names = ['name',
'frameTime',
'F0semitoneFrom27.5Hz_sma3nz_amean',
'F0semitoneFrom27.5Hz_sma3nz_stddevNorm',
'F0semitoneFrom27.5Hz_sma3nz_percentile20.0',
'F0semitoneFrom27.5Hz_sma3nz_percentile50.0',
'F0semitoneFrom27.5Hz_sma3nz_percentile80.0',
'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2',
'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope',
'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope',
'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope',
'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope',
'loudness_sma3_amean',
'loudness_sma3_stddevNorm',
'loudness_sma3_percentile20.0',
'loudness_sma3_percentile50.0',
'loudness_sma3_percentile80.0',
'loudness_sma3_pctlrange0-2',
'loudness_sma3_meanRisingSlope',
'loudness_sma3_stddevRisingSlope',
'loudness_sma3_meanFallingSlope',
'loudness_sma3_stddevFallingSlope',
'spectralFlux_sma3_amean',
'spectralFlux_sma3_stddevNorm',
'mfcc1_sma3_amean',
'mfcc1_sma3_stddevNorm',
'mfcc2_sma3_amean',
'mfcc2_sma3_stddevNorm',
'mfcc3_sma3_amean',
'mfcc3_sma3_stddevNorm',
'mfcc4_sma3_amean',
'mfcc4_sma3_stddevNorm',
'jitterLocal_sma3nz_amean',
'jitterLocal_sma3nz_stddevNorm',
'shimmerLocaldB_sma3nz_amean',
'shimmerLocaldB_sma3nz_stddevNorm',
'HNRdBACF_sma3nz_amean',
'HNRdBACF_sma3nz_stddevNorm',
'logRelF0-H1-H2_sma3nz_amean',
'logRelF0-H1-H2_sma3nz_stddevNorm',
'logRelF0-H1-A3_sma3nz_amean',
'logRelF0-H1-A3_sma3nz_stddevNorm',
'F1frequency_sma3nz_amean',
'F1frequency_sma3nz_stddevNorm',
'F1bandwidth_sma3nz_amean',
'F1bandwidth_sma3nz_stddevNorm',
'F1amplitudeLogRelF0_sma3nz_amean',
'F1amplitudeLogRelF0_sma3nz_stddevNorm',
'F2frequency_sma3nz_amean',
'F2frequency_sma3nz_stddevNorm',
'F2bandwidth_sma3nz_amean',
'F2bandwidth_sma3nz_stddevNorm',
'F2amplitudeLogRelF0_sma3nz_amean',
'F2amplitudeLogRelF0_sma3nz_stddevNorm',
'F3frequency_sma3nz_amean',
'F3frequency_sma3nz_stddevNorm',
'F3bandwidth_sma3nz_amean',
'F3bandwidth_sma3nz_stddevNorm',
'F3amplitudeLogRelF0_sma3nz_amean',
'F3amplitudeLogRelF0_sma3nz_stddevNorm',
'alphaRatioV_sma3nz_amean',
'alphaRatioV_sma3nz_stddevNorm',
'hammarbergIndexV_sma3nz_amean',
'hammarbergIndexV_sma3nz_stddevNorm',
'slopeV0-500_sma3nz_amean',
'slopeV0-500_sma3nz_stddevNorm',
'slopeV500-1500_sma3nz_amean',
'slopeV500-1500_sma3nz_stddevNorm',
'spectralFluxV_sma3nz_amean',
'spectralFluxV_sma3nz_stddevNorm',
'mfcc1V_sma3nz_amean',
'mfcc1V_sma3nz_stddevNorm',
'mfcc2V_sma3nz_amean',
'mfcc2V_sma3nz_stddevNorm',
'mfcc3V_sma3nz_amean',
'mfcc3V_sma3nz_stddevNorm',
'mfcc4V_sma3nz_amean',
'mfcc4V_sma3nz_stddevNorm',
'alphaRatioUV_sma3nz_amean',
'hammarbergIndexUV_sma3nz_amean',
'slopeUV0-500_sma3nz_amean',
'slopeUV500-1500_sma3nz_amean',
'spectralFluxUV_sma3nz_amean',
'loudnessPeaksPerSec',
'VoicedSegmentsPerSec',
'MeanVoicedSegmentLengthSec',
'StddevVoicedSegmentLengthSec',
'MeanUnvoicedSegmentLength',
'StddevUnvoicedSegmentLength',
'equivalentSoundLevel_dBp']

eGeneva_feature_idx = {name:i for i, name in enumerate(eGeneva_feature_names)}
