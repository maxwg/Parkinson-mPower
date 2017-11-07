"""
Code is very much research-quality the challenge coincides
with my Bachelor's thesis deadlines.

A higher quality version may be released in the future
given enough demand.

Author: Max Wang
        maxwg@outlook.com.au
"""
import json
import numpy as np
import nolds
import math
from numpy.linalg import norm
import numpy.linalg as la
from scipy.signal import butter, filtfilt
import external_libs.eeg as eeg
import external_libs.pyrem.univariate as pyrem
import external_libs.lyapunov as lyapunov
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
import time
from external_libs.rpde import rpde_main as rpde
from scipy.linalg import norm



def tkeo(a):
    """
    Calculates the TKEO of a given recording by using four samples.
    See Deburchgrave et al., 2008
    Arguments:
    a 			--- 1D numpy array.
    Returns:
    1D numpy array containing the tkeo per sample
    """

    # Create two temporary arrays of equal length, shifted 1 sample to the right
    # and left and squared:

    l = 1
    p = 2
    q = 0
    s = 3

    aTkeo = a[l:-p] * a[p:-l] - a[q:-s] * a[s:]

    return [np.mean(np.abs(aTkeo)), np.std(aTkeo)]


def motionToAcceleration(motion):
    return np.sum(np.abs(motion), axis=1)


def shortTimeFourierWindow(accel, measure_time):
    """
    From acceleration data, return the mean, min and stdev of
    data in short time Fourier bins.

    :param accel: Nxd numpy array
    :param measure_time: The time between each measurement. E.g 0.01s == 100Hz
    :return: [float]
    """
    chunks = accel.shape[0] // (2.5 // measure_time)  # split array into approximately 3 second chunks
    st_accel = np.array_split(accel, chunks)

    def getfftamplitudes(st):
        if (len(st) > 100):
            # mean values and standard deviations for the power spectral density at the frequencies of 1Hz, 3Hz, 6Hz, and 10Hz
            bin_vals = [(0, 3.5), (3.5, 5), (5, 7), (7, 10), (10, float("inf"))]

            def freqToBin(freq):
                absf = abs(freq)
                for i, bound in enumerate(bin_vals):
                    lb, ub = bound
                    if absf >= lb and absf < ub:
                        return i

            bins = [[] for _ in range(len(bin_vals))]
            fftfreq = np.fft.fftfreq(st.shape[0], measure_time)
            fftamplitude = np.array([np.absolute(np.fft.rfft(st[:, 0])), np.absolute(np.fft.rfft(st[:, 1]))]).T
            fftamplitude = np.sqrt(np.sum(fftamplitude ** 2, axis=1))
            for freq, amp in zip(fftfreq, fftamplitude):
                bin = freqToBin(freq)
                if bin is not None:
                    bins[bin].append(amp)
            bins = [np.mean(b) for b in bins]
            return bins

    st_amps = []
    for st in st_accel:
        st_amps.append(getfftamplitudes(st))
    st_amps = np.array(st_amps)
    means = st_amps.mean(axis=0)
    normalizer = means.mean()
    means = means / normalizer
    mins = st_amps.min(axis=0) / normalizer
    stdev = st_amps.std(axis=0) / normalizer
    return np.array([mins, means, stdev]).flatten()


def getMomentsNorm(accel, moments=6):
    """
    Get the moments from acceleration signal
    :param accel: Nxd numpy array
    :param moments: Number of moments
    :return: [float]
    """
    from scipy.stats import moment
    return [np.sqrt(np.sum(mo ** 2)) for mo in [moment(accel, moment=n) for n in range(2, moments)]] + [
        np.sqrt(np.sum(np.mean(accel, axis=0) ** 2))]


def getMomentsIndividual(position, moments=4):
    """
    Get moments from one-dimensional signal.
    More suitable for position rather than acceleration

    :param position: numpy array R^N
    :param moments: number of moments
    :return: [float]
    """
    from scipy.stats import moment
    return [moment(position, moment=n) for n in range(2, moments)]

import csv, time
def loadMotionIntoArray(path):
    """
    Given a file, return the acceleration data
    :param path: ./Path/to/file.json
    """
    with open(path, 'r') as f:
        csvread = csv.reader(f, delimiter='\t')
        headers = next(csvread, None)
        # print(headers)
        arr = []
        for r in csvread:
            r = [float(v) for v in r]
            arr.append(r)

        means = np.nanmean(np.array(arr), axis=0)
        means = list([0 if np.isnan(m) else m for m in means])
        res = []
        for r in arr:
            # A tiny bit of gaussian noise is added in the case
            # that all values are NaN, screwing with the
            # entropy histograms.
            res.append([means[i] + np.random.rand()*0.000001
                if np.isnan(r[i]) else r[i] for i in range(len(r))])
        arr = np.array(res)
        """ The data is solely NYC/Boston so we don't need to worry
            about other time zones for now.
        """
        time_of_day = time.gmtime(arr[0,0])
        time_of_day = time_of_day.tm_hour + time_of_day.tm_min/37 + time_of_day.tm_sec/3600
        return np.mean(arr[1:,0] - arr[:-1,0]), time_of_day, arr[:,1:-1]


def getAccelerationFromPositions(position):
    accel = []
    for i in range(1, position.shape[0]):
        accel.append(position[i] - position[i - 1])
    return accel


def getFirstMinimaIdx(arr):
    """
    Very naive method of returning first minimum
    :param arr: array-like
    :return: int
    """
    cval = arr[0]
    for i, n in enumerate(arr):
        if n > cval:
            return min(i, 100)
        cval = n
    print("ERR NO MINIMA")
    return 100


def getTau(signal):
    # Set Tau with first minimum of mutual info and embedding dimension using False Nearest Neighbours.
    return getFirstMinimaIdx(lyapunov.lagged_ami(signal, 0, 100)[1])


def getZCR(signal):
    """ The zero crossing rate
    """
    return [((signal[:-1] * signal[1:]) < 0).mean()]


def crossEntropyAndMutualInformationAndCorrelation(x, y):
    """
    Given two signals, returns their cross entropy, cross correlation and mutual information.
    """
    xy = np.concatenate((x, y))
    bin_range = xy.min(), xy.max()
    num_bins = int(math.ceil(np.sqrt(x.size / 5)))  # Reccomended by Cellucci et al. (2005)
    px = np.histogram(x, num_bins)[0]
    py = np.histogram(y, num_bins)[0]
    pxy = np.histogram2d(x, y, num_bins)[0]

    x_ent = entropy(px)
    y_ent = entropy(py)
    cross_ent = entropy(px + 0.01, py + 0.01)
    corr = np.correlate(x, y)[0]
    mi = mutual_info_score(None, None, contingency=pxy)

    return x_ent, y_ent, cross_ent, corr, mi


def get_ap_samp_entropy_filtering(y):
    """
    Follows Lu (2008) in determining the optimal r parameter.

    Uses the equation provided for m=4. S
    Should approximately generalize to other parameters of m.

    Select m with false nearest neighbor algorithm
    :param y:
    :return:
    """
    e = y[:-1] - y[1:]
    sd1 = np.std(e)
    sd2 = np.std(y)
    return (-0.11 + 0.65 * np.sqrt(sd1 / sd2)) / np.power(len(y) / 1000, 1 / 4)


def getFeatures(accel_readings, sampling_time, step=50):
    """
    Optimal features for walking data
    :param path: ./path/to/walking_file.json
    :return: {features}, int number_errors, Nx3 numpy array of raw accel.
    """
    shorttime_features = []
    for i in range(0,401, step):
        accel = accel_readings[i:600+i]
        x, y, z = accel[:, 0], accel[:, 1], accel[:, 2]
        features = {}

        moments = getMomentsNorm(accel)
        moments.extend(getMomentsIndividual(x) + getMomentsIndividual(y) + getMomentsIndividual(z))
        moments.extend(getZCR(x) + getZCR(y) + getZCR(z))
        features['moments'] = moments
        features['fourier'] = shortTimeFourierWindow(accel, sampling_time)
        features['tkeo'] = tkeo(accel)
        features['entropy'] = crossEntropyAndMutualInformationAndCorrelation(x, y) + \
                              crossEntropyAndMutualInformationAndCorrelation(x, z)[1:] + \
                              crossEntropyAndMutualInformationAndCorrelation(y, z)[2:]

        dynamic = []
        xTau = getTau(x)
        yTau = getTau(y)
        zTau = getTau(z)
        dynamic.append(xTau)
        dynamic.append(yTau)
        dynamic.append(zTau)
        dynamic.append(pyrem.hfd(x, 60))  # Gomez 2008 states dimenison should be chosen as point where FD plateus.
        dynamic.append(pyrem.hfd(y, 60))  # 60 seems to be an upper-bound for most samples
        dynamic.append(pyrem.hfd(z, 60))  # 60 seems to be an upper-bound for most samples
        dynamic.append(pyrem.hurst(x))
        dynamic.append(pyrem.hurst(y))
        dynamic.append(pyrem.hurst(z))
        dynamic.append(pyrem.pfd(x))
        dynamic.append(pyrem.pfd(y))
        dynamic.append(pyrem.pfd(z))
        dynamic.extend([nolds.dfa(x), nolds.dfa(y), nolds.dfa(z)])
        dynamic.extend([rpde(x.tolist(), 5, xTau), rpde(y.tolist(), 5, yTau), rpde(z.tolist(), 5, zTau)])

        # print(lyapunov.global_false_nearest_neighbors(x, getTau(x), 1, 20))
        # print(lyapunov.global_false_nearest_neighbors(y, getTau(y), 1, 20))

        lyap_x_Eck = nolds.lyap_e(x, emb_dim=5, matrix_dim=5)
        lyap_x_Ros = nolds.lyap_r(x, emb_dim=5)
        lyap_y_Eck = nolds.lyap_e(y, emb_dim=5, matrix_dim=5)
        lyap_y_Ros = nolds.lyap_r(y, emb_dim=5)
        lyap_z_Eck = nolds.lyap_e(z, emb_dim=5, matrix_dim=5)
        lyap_z_Ros = nolds.lyap_r(z, emb_dim=5)
        dynamic.extend(lyap_x_Eck)
        dynamic.append(lyap_x_Ros)
        dynamic.extend(lyap_y_Eck)
        dynamic.append(lyap_y_Ros)
        dynamic.extend(lyap_z_Eck)
        dynamic.append(lyap_z_Ros)
        # dynamic.append(eeg.LLE(x, xTau, 5, sampling_time, 1/sampling_time))
        # dynamic.append(eeg.LLE(y, yTau, 5, sampling_time, 1/sampling_time))
        features['dynamic'] = dynamic
        info_dynamic = []
        rx = get_ap_samp_entropy_filtering(x)
        ry = get_ap_samp_entropy_filtering(y)
        rz = get_ap_samp_entropy_filtering(z)
        info_dynamic.append(pyrem.svd_entropy(x, xTau, 5))
        info_dynamic.append(pyrem.svd_entropy(y, yTau, 5))
        info_dynamic.append(pyrem.svd_entropy(z, zTau, 5))
        info_dynamic.append(pyrem.fisher_info(x, xTau, 5))
        info_dynamic.append(pyrem.fisher_info(y, yTau, 5))
        info_dynamic.append(pyrem.fisher_info(z, zTau, 5))
        info_dynamic.append(pyrem.spectral_entropy(x, 1 / sampling_time, [3, 5, 7, 10, 15]))
        info_dynamic.append(pyrem.spectral_entropy(y, 1 / sampling_time, [3, 5, 7, 10, 15]))
        info_dynamic.append(pyrem.spectral_entropy(z, 1 / sampling_time, [3, 5, 7, 10, 15]))
        info_dynamic.append(pyrem.ap_entropy(x, 5, rx))
        info_dynamic.append(pyrem.ap_entropy(y, 5, ry))
        info_dynamic.append(pyrem.ap_entropy(z, 5, rz))
        info_dynamic.extend([pyrem.samp_entropy(x, m=5, r=rx, tau=tau, relative_r=True) for tau in range(1, 5)])
        info_dynamic.extend([pyrem.samp_entropy(y, m=5, r=ry, tau=tau, relative_r=True) for tau in range(1, 5)])
        info_dynamic.extend([pyrem.samp_entropy(z, m=5, r=rz, tau=tau, relative_r=True) for tau in range(1, 5)])
        features['info_dynamic'] = info_dynamic
        features['hjorth'] = pyrem.hjorth(x) + pyrem.hjorth(y) + pyrem.hjorth(z)
        shorttime_features.append(features)
    return shorttime_features

def featuresNames():
    features = {}

    moments = ["accel_var", "accel_skew", "  _kurt", "accel_moment_5", "accel_mean",
               "accel_x_var", "accel_x_kurt", "accel_y_var", "accel_y_kurt", "accel_z_var", "accel_z_kurt",
               "accel_x_zcr", "accel_y_zcr", "accel_z_zcr"]
    features['moments'] = moments
    # [(0, 1.5), (1.5, 3), (3, 5), (5, 7), (7, 10), (10, 14), (14, float('inf'))]
    features['fourier'] = ["fourier_3hz_min", "fourier_5hz_min", "fourier_7hz_min",
                           "fourier_10hz_min", "fourier_remain_min",
                           "fourier_3hz_mean", "fourier_5hz_mean", "fourier_7hz_mean",
                           "fourier_10hz_mean", "fourier_remain_mean",
                            "fourier_3hz_std", "fourier_5hz_std", "fourier_7hz_std",
                           "fourier_10hz_std", "fourier_remain_std"]

    features['tkeo'] = ["mean_tkeo", "std_tkeo"]

    features['entropy'] = ["x_ent", "y_ent", "xy_cross_ent", "xy_cross_corr", "xy_mutual_info",
                           "z_ent", "xz_cross_ent", "xz_cross_corr", "xz_mutual_info",
                           "yz_cross_ent", "yz_cross_corr", "yz_mutual_info"]

    dynamic = ["xTau", "yTau", "zTau", "higuchi_x", "higuchi_y", "higuchi_z", "hurst_x", "hurst_y", "hurst_z", "pfd_x",
               "pfd_y", "pfd_z", "dfa_x", "dfa_y", "dfa_z", "rpde_x", "rpde_y", "rpde_z",
               "Lyap_Eck_x_1", "Lyap_Eck_x_2", "Lyap_Eck_x_3", "Lyap_Eck_x_4", "Lyap_Eck_x_5",
               "LLE_Rosen_x",
               "Lyap_Eck_y_1", "Lyap_Eck_y_2", "Lyap_Eck_y_3", "Lyap_Eck_y_4", "Lyap_Eck_y_5",
               "LLE_Rosen_y",
               "Lyap_Eck_z_1", "Lyap_Eck_z_2", "Lyap_Eck_z_3", "Lyap_Eck_z_4", "Lyap_Eck_z_5",
               "LLE_Rosen_z"]

    features['dynamic'] = dynamic

    info_dynamic = ["x_svd_ent", "y_svd_ent", "z_svd_ent", "x_fisher", "y_fisher", "z_fisher", "x_spect_ent",
                    "y_spect_ent", "z_spect_ent",
                    "x_ap_ent", "y_ap_ent", "z_ap_ent", "x_samp_ent_1", "x_samp_ent_2", "x_samp_ent_3", "x_samp_ent_4",
                    "y_samp_ent_1", "y_samp_ent_2", "y_samp_ent_3", "y_samp_ent_4", "z_samp_ent_1", "z_samp_ent_2",
                    "z_samp_ent_3", "z_samp_ent_4"]
    features['info_dynamic'] = info_dynamic

    features['hjorth'] = ["x_activity", "x_complexity", "x_morbidity",
                          "y_activity", "y_complexity", "y_morbidity",
                          "z_activity", "z_complexity", "z_morbidity"]

    return features


def getPedometerFeatures(path):
    """
    Get pedometer features using the given challenge baseline
    feature set
    :param path: ./path/to/pedometer_data.json
    """
    pedo = np.asarray(r['getPedometerFeatures'](path))[:-1]

    def tryFloat(val):
        try:
            return float(val)
        except:
            print("err...")
            return math.nan

    mpower = [tryFloat(v) for v in pedo]
    return mpower

