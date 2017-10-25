"""
    This module extracts features used EEG signal analysis.
"""

from path_processors import *
from audio_helpers import *
import nolds
def getDynamicalSpeechFeatures(signal, sampling_rate=44100):
    signal = signal.flatten()
    dynamic = []
    dynamic.append(pyrem.hfd(signal, 75))  # Gomez 2008 states dimenison should be chosen as point where FD plateus.
    dynamic.append(eeg.pfd(signal))  # Gomez 2008 states dimenison should be chosen as point where FD plateus.
    dynamic.append(pyrem.hurst(signal))
    dynamic.append(pyrem.pfd(signal))
    dynamic.append(eeg.dfa(signal))

    tau = getTau(signal)
    dynamic.append(tau)
    # print(lyapunov.global_false_nearest_neighbors(signal, tau, 1, 20)) >> 9
    # Set Tau with first minimum of mutual info and embedding dimension using False Nearest Neighbours.


    LLE_Eck=nolds.lyap_e(signal[2000:22000], emb_dim=6, matrix_dim=6)
    LLE_Ros=nolds.lyap_r(signal[2000:22000], emb_dim=6)
    dynamic.extend(LLE_Eck)
    dynamic.append(LLE_Ros)

    # dynamic.append(eeg.LLE(signal[:10000], tau, 6, 1/sampling_rate, sampling_rate))
    dynamic.extend(pyrem.hjorth(signal))
    # dynamic.append(pyrem.fisher_info(signal, tau, 4) )

    r = get_ap_samp_entropy_filtering(signal[:10000])
    dynamic.append(pyrem.svd_entropy(signal, tau, 6))
    dynamic.append(pyrem.fisher_info(signal, tau, 6))
    dynamic.append(pyrem.spectral_entropy(signal, sampling_rate))
    dynamic.append(pyrem.ap_entropy(signal[:10000], tau, r))
    dynamic.extend([pyrem.samp_entropy(signal[:10000], m=6, r=r, tau=tau, relative_r=True) for tau in range(1, 5)])
    return dynamic

def fixDynamicalSpeechFeatures(signal, dynamic):
    signal = np.array(signal).flatten().tolist()
    LLE_Eck=nolds.lyap_e(signal[2000:12000], emb_dim=6, matrix_dim=6)
    LLE_Ros=nolds.lyap_r(signal[2000:12000], emb_dim=6)
    dynamic[6] = LLE_Eck[0]
    dynamic[7] = LLE_Eck[1]
    dynamic[8] = LLE_Eck[2]
    dynamic[9] = LLE_Eck[3]
    dynamic[10] = LLE_Eck[4]
    dynamic[11] = LLE_Eck[5]
    dynamic[12] = LLE_Ros
    return dynamic

if __name__ == "__main__":
    samprate, audio = loadAudioAsArray("dataSamples/audio/talk_pd1.wav")
    print(getDynamicalSpeechFeatures(audio[:samprate*2].flatten(), samprate))
