import json
import numpy as np
import math
from numpy.linalg import norm
import numpy.linalg as la
from scipy.signal import butter, filtfilt
import external_libs.eeg as eeg
import external_libs.pyrem.univariate as pyrem
import external_libs.lyapunov as lyapunov
import voicetoolbox_helper
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
import time
import nolds
from external_libs.rpde import rpde_main as rpde
try:
    from rpy2.robjects import r
    from rpy2.robjects.packages import importr
    importr('plyr')
    importr('dplyr', on_conflict="warn")
    importr('ggplot2')
    importr('doMC')
    importr('jsonlite')
    importr('parallel')
    importr('tidyr')
    importr('lubridate')
    importr('stringr')
    importr('sqldf')
    importr('parsedate')
    importr("mpowertools")

except Exception as e:
    print(e)
    print("Could not import R engine")

eng = None
eng_execs = 0
def createOrRepairMatlabEngine():
    global eng, eng_execs
    if eng is None:
        eng = voicetoolbox_helper.newEngine()

    eng_execs += 1
    try:
        if eng_execs % 400 == 0:
            eng.quit()
            time.sleep(15)
            eng = voicetoolbox_helper.newEngine()
        elif eng_execs % 20 == 0:
            eng.clear(nargout=0)

    except:
        try:
            eng.quit()
        except:
            """"""
        eng = voicetoolbox_helper.newEngine()
        print("Matlab Error")


def butter_highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='highpass')
    return b, a


def butter_highpass_filter(data, lowcut, fs, order=5):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


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

def skew(v):
    if len(v) == 4: v = v[:3]/v[3]
    skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
    return skv - skv.T

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def area_np(x, y):
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    n = len(x)
    shift_up = np.arange(-n+1, 1)
    shift_down = np.arange(-1, n-1)
    return (x * (y.take(shift_up) - y.take(shift_down))).sum() / 2.0


def getBoundingEllipse(points, tol = 0.02):
    """
    Finds the ellipse equation in "center form"
    (x-c).T * A * (x-c) = 1
    """
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = la.norm(new_u-u)
        u = new_u
    c = np.dot(u,points)
    A = la.inv(np.dot(np.dot(points.T, np.diag(u)), points)
               - np.multiply.outer(c,c))/d
    centroid = np.transpose(c)

    U, D, V = la.svd(A)
    rx, ry, rz = [1/np.sqrt(d) for d in D]
    u, v = np.mgrid[0:2*np.pi:20j,-np.pi/2:np.pi/2:10j]
    x=rx*np.cos(u)*np.cos(v)
    y=ry*np.sin(u)*np.cos(v)
    z=rz*np.sin(v)
    for idx in range(x.shape[0]):
        for idy in range(y.shape[1]):
            x[idx, idy], y[idx, idy], z[idx, idy] = np.dot(np.transpose(V),
                                                           np.array([x[idx, idy], y[idx, idy], z[idx, idy]])) + centroid
    return (rx, ry, rz), (x, y, z)

def getAreaOfEllipsoidMeasurements(radius):
    area = [radius[0]*radius[1]]
    return area

def accelerationToPos(motion, sampling_time = None):
    xyz = []
    cx, cy, cz = 0, 0,0
    xyz.append((cx, cy, cz))
    for x, y, z in motion:
        cx += x
        cy += y
        cz += z
        xyz.append((cx, cy, cz))
    xyz = np.array(xyz)
    return xyz

def motionToAcceleration(motion):
    return np.sum(np.abs(motion), axis=1)

def shortTimeFourierWindow(accel, measure_time):
    chunks = accel.shape[0] // (2.5 // measure_time) #split array into approximately 3 second chunks
    st_accel = np.array_split(accel, chunks)
    def getfftamplitudes(st):
        if(len(st)>100):
            #mean values and standard deviations for the power spectral density at the frequencies of 1Hz, 3Hz, 6Hz, and 10Hz
            bin_vals = [(0, 1.5),(1.5, 3), (3, 5), (5, 7), (7, 10), (10,14), (14, float('inf'))]

            def freqToBin(freq):
                absf = abs(freq)
                for i, bound in enumerate(bin_vals):
                    lb, ub = bound
                    if absf >= lb and absf < ub:
                        return i

            bins = [[] for _ in range(len(bin_vals))]
            fftfreq = np.fft.fftfreq(st.shape[0], measure_time)
            fftamplitude = np.array([np.absolute(np.fft.rfft(st[:,0])), np.absolute(np.fft.rfft(st[:,1]))]).T
            fftamplitude = np.sqrt(np.sum(fftamplitude**2, axis=1))
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
    means = means/normalizer
    mins = st_amps.min(axis=0)/normalizer
    stdev = st_amps.std(axis=0)/normalizer
    return np.array([mins, means, stdev]).flatten()

def getMomentsNorm(accel, moments=6):
    from scipy.stats import moment
    return [np.sqrt(np.sum(mo**2)) for mo in [moment(accel, moment=n) for n in range(2,moments)]] + [np.sqrt(np.sum(np.mean(accel, axis=0)**2))]

def getMomentsIndividual(position, moments=4):
    from scipy.stats import moment
    return [moment(position, moment=n) for n in range(2,moments)]

from scipy.linalg import norm
def RotAtoB( a,b ):
    x = [a[1]*b[2] - b[1]*a[2],
         a[2]*b[0] - b[2]*a[0],
         a[0]*b[1] - b[0]*a[1]]
    x = np.array(x)/norm(x)
    theta = math.acos(a @ b/(norm(a)*norm(b)))
    A = [[0,-x[2], x[1]],
         [x[2],0,-x[0]],
         [-x[1],x[0],0]]
    A = np.array(A)
    R = np.eye(3) + math.sin(theta)*A + (1-math.cos(theta))*A**2
    return R

def loadMotionIntoArray(path, highPass=False, min_len=2000, crop=(400, 2000)):
    with open(path) as f:
        errors = 0
        data = json.load(f)[1:-1]
        X = []
        znorm = np.array([0, 0, 1])
        xnorm = np.array([1, 0, 0])
        ptime = None
        times = []
        cx, cy, cz = 0, 0, 0
        for j in data:
            if not (ptime is None):
                times.append(j['timestamp'] - ptime)
            ptime = j['timestamp']
            accel = np.array((j['userAcceleration']['x'], j['userAcceleration']['y'], j['userAcceleration']['z']))
            grav = np.array((j['gravity']['x'], j['gravity']['y'], j['gravity']['z']))
            if norm(accel) > 0 :
                # axis = np.cross(grav, znorm)
                # axis = axis / norm(axis)
                # angle = np.arccos(np.dot(grav, znorm) / (np.dot(norm(grav), norm(znorm))))
                #
                # # A = skew(axis)
                # R = rotation_matrix(axis, angle)
                R = RotAtoB(grav, xnorm)
                accel_grav = R @ accel
                X.append(accel_grav)
            else:
                errors += 1

        """ Rotate path in direction of travel.
        """
        if np.std(times) > 0.001:
            print("MAJOR MEASURE ISSUE!", np.std(times), times)
        if errors > 800 or len(X) < min_len:
            print("too many errors", errors, len(X))
            return [100000,1000000], 1, 10000

        X = X[crop[0]:crop[1]]

        X = np.array(X)
        # mean = np.mean(X, axis=0)
        # std = np.std(X, axis=0)
        # psize = X.shape[0]
        # X = np.array([x for x in X if (np.all(x > mean - 2*std) and np.all(x < mean + 2*std))])

        X_displacement = np.sum(X, axis=0)
        R = RotAtoB(X_displacement, xnorm)
        X_rotated = []
        for x in X:
            X_rotated.append(R @ x)

        X_rotated = np.array(X_rotated)

        sampling_time = np.mean(times)
        if highPass:
            if sampling_time is None:
                print("Wavelength required for butterworth filter")
            try:
                X_rotated = butter_highpass_filter(X.T, 1, 1 / sampling_time, order=10).T
            except:
                print("Highpass Exception")
                return [100000, 1000000], 1, 10000

        return X_rotated, sampling_time, errors

def getAccelerationFromPositions(position):
    accel=[]
    for i in range(1, position.shape[0]):
        accel.append(position[i] - position[i-1])
    return accel

def getFirstMinimaIdx(arr):
    cval = arr[0]
    for i, n in enumerate(arr):
        if n > cval:
            return i
        cval = n
    raise("ERR NO MINIMA")

def getTau(signal):
    # Set Tau with first minimum of mutual info
    # set embedding dimension using False Nearest Neighbours.
    return getFirstMinimaIdx(lyapunov.lagged_ami(signal, 0, 200)[1])

def getZCR(signal):
    return [((signal[:-1] * signal[1:]) < 0).mean()]

def crossEntropyAndMutualInformationAndCorrelation(x,y):
    xy = np.concatenate((x,y))
    bin_range = xy.min(), xy.max()
    num_bins = int(math.ceil(np.sqrt(x.size/5))) # Reccomended by Cellucci et al. (2005)
    px = np.histogram(x, num_bins)[0]
    py = np.histogram(y, num_bins)[0]
    pxy = np.histogram2d(x, y, num_bins)[0]

    x_ent = entropy(px)
    y_ent = entropy(py)
    cross_ent = entropy(px+0.01, py+0.01)
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
    return (-0.11 + 0.65*np.sqrt(sd1/sd2))/np.power(len(y)/1000, 1/4)

def getAllFeaturesRest(path, highPass=True):
    accel, sampling_time, errors = loadMotionIntoArray(path, highPass=highPass)
    if errors > 1000:
        return accel, errors, errors
    position = accelerationToPos(accel)
    if highPass:
        accel = np.array(getAccelerationFromPositions(position))
    accel2d = accel[:, 0:2]
    x, y, z = accel[:, 0], accel[:, 1], accel[:, 2]
    px, py, pz = position[:, 0], position[:, 1], position[:, 2]

    features = {}

    moments = getMomentsNorm(accel2d)
    moments.extend(getMomentsIndividual(px) + getMomentsIndividual(py))
    moments.extend(getZCR(x) + getZCR(y))
    features['moments'] = moments
    features['fourier'] = shortTimeFourierWindow(accel2d, sampling_time)
    features['tkeo'] = tkeo(accel)
    features['entropy'] = crossEntropyAndMutualInformationAndCorrelation(x, y)

    radius, surface = getBoundingEllipse(position)
    features['area'] = getAreaOfEllipsoidMeasurements(radius)

    dynamic = []
    xTau = getTau(x)
    yTau = getTau(y)
    dynamic.append(xTau)
    dynamic.append(yTau)
    dynamic.append(pyrem.hfd(x, 60))  # Gomez 2008 states dimenison should be chosen as point where FD plateus.
    dynamic.append(pyrem.hfd(y, 60))  # 60 seems to be an upper-bound for most samples
    dynamic.append(pyrem.hurst(x))
    dynamic.append(pyrem.hurst(y))
    dynamic.append(pyrem.pfd(x))
    dynamic.append(pyrem.pfd(y))
    dynamic.extend([eeg.dfa(x), eeg.dfa(y)])
    dynamic.extend([rpde(x.tolist(), 6, xTau), rpde(y.tolist(), 6, yTau)])
    # print(lyapunov.global_false_nearest_neighbors(x, getTau(x), 1, 20))
    # print(lyapunov.global_false_nearest_neighbors(y, getTau(y), 1, 20))
    # dynamic.append(eeg.LLE(x, xTau, 6, sampling_time, 1/sampling_time))
    # dynamic.append(eeg.LLE(y, yTau, 6, sampling_time, 1/sampling_time))


    lyap_x_Eck = nolds.lyap_e(x, emb_dim=6, matrix_dim=6)
    lyap_x_Ros = nolds.lyap_r(x, emb_dim=6)
    lyap_y_Eck = nolds.lyap_e(y, emb_dim=6, matrix_dim=6)
    lyap_y_Ros = nolds.lyap_r(y, emb_dim=6)
    dynamic.extend(lyap_x_Eck)
    dynamic.append(lyap_x_Ros)
    dynamic.extend(lyap_y_Eck)
    dynamic.append(lyap_y_Ros)

    features['dynamic'] = dynamic

    info_dynamic = []
    rx = get_ap_samp_entropy_filtering(x)
    ry = get_ap_samp_entropy_filtering(y)
    info_dynamic.append(pyrem.svd_entropy(x, xTau, 6))
    info_dynamic.append(pyrem.svd_entropy(y, yTau, 6))
    info_dynamic.append(pyrem.fisher_info(x, xTau, 6))
    info_dynamic.append(pyrem.fisher_info(y, yTau, 6))
    info_dynamic.append(pyrem.spectral_entropy(x, 1 / sampling_time, [3, 5, 7, 10, 15]))
    info_dynamic.append(pyrem.spectral_entropy(y, 1 / sampling_time, [3, 5, 7, 10, 15]))
    info_dynamic.append(pyrem.ap_entropy(x, xTau, rx))
    info_dynamic.append(pyrem.ap_entropy(y, yTau, ry))
    info_dynamic.extend([pyrem.samp_entropy(x, m=6, r=rx, tau=tau, relative_r=True) for tau in range(1, 5)])
    info_dynamic.extend([pyrem.samp_entropy(y, m=6, r=rx, tau=tau, relative_r=True) for tau in range(1, 5)])
    features['info_dynamic'] = info_dynamic
    features['hjorth'] = pyrem.hjorth(x) + pyrem.hjorth(y)
    mpower = np.asarray(r['getRestFeatures'](path))[:-1]

    def tryFloat(val):
        try:
            return float(val)
        except:
            print("err...")
            return math.nan

    mpower = [tryFloat(v) for v in mpower]
    features['mpower'] = mpower

    return features, errors, accel

def featuresRestNames():
    features = {}
    moments = ["accel_var", "accel_skew", "accel_kurt", "accel_moment_5", "accel_mean",
               "accel_x_var", "accel_x_kurt", "accel_y_var", "accel_y_kurt",
               "accel_x_zcr", "accel_y_zcr"]
    features['moments'] = moments
    # [(0, 1.5), (1.5, 3), (3, 5), (5, 7), (7, 10), (10, 14), (14, float('inf'))]
    features['fourier'] = ["fourier_1.5hz_min", "fourier_3hz_min","fourier_5hz_min","fourier_7hz_min","fourier_10hz_min","fourier_14hz_min","fourier_remain_min",
                           "fourier_1.5hz_mean", "fourier_3hz_mean", "fourier_5hz_mean", "fourier_7hz_mean", "fourier_10hz_mean", "fourier_14hz_mean", "fourier_remain_mean",
                           "fourier_1.5hz_std", "fourier_3hz_std", "fourier_5hz_std", "fourier_7hz_std", "fourier_10hz_std", "fourier_14hz_std", "fourier_remain_std"]

    features['tkeo'] = ["mean_tkeo", "std_tkeo"]
    # x_ent, y_ent, cross_ent, corr, mi
    features['entropy'] = ["x_ent", "y_ent", "xy_cross_ent", "xy_cross_corr", "xy_mutual_info"]

    features['area'] = ["bounding_ellipse_area"]

    dynamic = ["xTau", "yTau", "higuchi_x", "higuchi_y", "hurst_x", "hurst_y", "pfd_x", "pfd_y", "dfa_x", "dfa_y",
               "rpde_x", "rpde_y", "LLE_x", "LLE_y"]
    features['dynamic'] = dynamic

    info_dynamic = ["x_svd_ent", "y_svd_ent", "x_fisher", "y_fisher", "x_spect_ent", "y_spect_ent",
                    "x_ap_ent", "y_ap_ent", "x_samp_ent_1", "x_samp_ent_2", "x_samp_ent_3", "x_samp_ent_4",
                    "y_samp_ent_1", "y_samp_ent_2", "y_samp_ent_3", "y_samp_ent_4"]
    features['info_dynamic'] = info_dynamic


    features['hjorth'] = ["x_activity", "x_complxity", "x_morbidity", "y_activity", "y_complxity", "y_morbidity"]

    mpower = ["meanAA", "sdAA", "modeAA","skewAA", "kurAA", "q1AA", "medianAA",
                  "q3AA", "iqrAA", "rangeAA", "acfAA", "zcrAA", "dfaAA", "turningTime",
              "postpeak", "postpower", "alpha", "dVol", "ddVol"
    ]
    features['mpower'] = mpower
    return features

def getAllFeaturesWalking(path):
    """
        Optimal features for walking data
        :param path: ./path/to/walking_file.json
        :return: {features}, int number_errors, Nx3 numpy array of raw accel.
        """
    accel, sampling_time, errors = loadMotionIntoArray(path, highPass=False, crop=(0, 850))
    if errors > 1000:
        return accel, errors, errors
    position = accelerationToPos(accel)
    x, y, z = accel[:, 0], accel[:, 1], accel[:, 2]
    px, py, pz = position[:, 0], position[:, 1], position[:, 2]

    features = {}

    moments = getMomentsNorm(accel)
    moments.extend(getMomentsIndividual(px) + getMomentsIndividual(py) + getMomentsIndividual(pz))
    moments.extend(getZCR(x) + getZCR(y) + getZCR(z))
    features['moments'] = moments
    features['fourier'] = shortTimeFourierWindow(accel, sampling_time)
    features['tkeo'] = tkeo(accel)
    features['entropy'] = crossEntropyAndMutualInformationAndCorrelation(x, y) + \
                          crossEntropyAndMutualInformationAndCorrelation(x, z)[1:] + \
                          crossEntropyAndMutualInformationAndCorrelation(y, z)[2:]

    radius, surface = getBoundingEllipse(position)
    features['area'] = getAreaOfEllipsoidMeasurements(radius)

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
    dynamic.extend([eeg.dfa(x), eeg.dfa(y), eeg.dfa(z)])
    dynamic.extend([rpde(x.tolist(), 6, xTau), rpde(y.tolist(), 6, yTau), rpde(z.tolist(), 6, zTau)])

    # print(lyapunov.global_false_nearest_neighbors(x, getTau(x), 1, 20))
    # print(lyapunov.global_false_nearest_neighbors(y, getTau(y), 1, 20))

    lyap_x_Eck = nolds.lyap_e(x, emb_dim=6, matrix_dim=6)
    lyap_x_Ros = nolds.lyap_r(x, emb_dim=6)
    lyap_y_Eck = nolds.lyap_e(y, emb_dim=6, matrix_dim=6)
    lyap_y_Ros = nolds.lyap_r(y, emb_dim=6)
    lyap_z_Eck = nolds.lyap_e(z, emb_dim=6, matrix_dim=6)
    lyap_z_Ros = nolds.lyap_r(z, emb_dim=6)
    dynamic.extend(lyap_x_Eck)
    dynamic.append(lyap_x_Ros)
    dynamic.extend(lyap_y_Eck)
    dynamic.append(lyap_y_Ros)
    dynamic.extend(lyap_z_Eck)
    dynamic.append(lyap_z_Ros)
    # dynamic.append(eeg.LLE(x, xTau, 6, sampling_time, 1/sampling_time))
    # dynamic.append(eeg.LLE(y, yTau, 6, sampling_time, 1/sampling_time))
    features['dynamic'] = dynamic

    info_dynamic = []
    rx = get_ap_samp_entropy_filtering(x)
    ry = get_ap_samp_entropy_filtering(y)
    rz = get_ap_samp_entropy_filtering(z)
    info_dynamic.append(pyrem.svd_entropy(x, xTau, 6))
    info_dynamic.append(pyrem.svd_entropy(y, yTau, 6))
    info_dynamic.append(pyrem.svd_entropy(z, zTau, 6))
    info_dynamic.append(pyrem.fisher_info(x, xTau, 6))
    info_dynamic.append(pyrem.fisher_info(y, yTau, 6))
    info_dynamic.append(pyrem.fisher_info(z, zTau, 6))
    info_dynamic.append(pyrem.spectral_entropy(x, 1 / sampling_time, [3, 5, 7, 10, 15]))
    info_dynamic.append(pyrem.spectral_entropy(y, 1 / sampling_time, [3, 5, 7, 10, 15]))
    info_dynamic.append(pyrem.spectral_entropy(z, 1 / sampling_time, [3, 5, 7, 10, 15]))
    info_dynamic.append(pyrem.ap_entropy(x, 6, rx))
    info_dynamic.append(pyrem.ap_entropy(y, 6, ry))
    info_dynamic.append(pyrem.ap_entropy(z, 6, rz))
    info_dynamic.extend([pyrem.samp_entropy(x, m=6, r=rx, tau=tau, relative_r=True) for tau in range(1, 5)])
    info_dynamic.extend([pyrem.samp_entropy(y, m=6, r=ry, tau=tau, relative_r=True) for tau in range(1, 5)])
    info_dynamic.extend([pyrem.samp_entropy(z, m=6, r=rz, tau=tau, relative_r=True) for tau in range(1, 5)])
    features['info_dynamic'] = info_dynamic
    features['hjorth'] = pyrem.hjorth(x) + pyrem.hjorth(y) + pyrem.hjorth(z)

    mpower = np.asarray(r['getWalkFeatures'](path))[:-1]

    def tryFloat(val):
        try:
            return float(val)
        except:
            print("err...")
            return math.nan

    mpower = [tryFloat(v) for v in mpower]
    features['mpower'] = mpower
    return features, errors, accel


def getPedometerFeatures(path):
    pedo = np.asarray(r['getPedometerFeatures'](path))[:-1]
    def tryFloat(val):
        try:
            return float(val)
        except:
            print("err...")
            return math.nan

    pedo = [tryFloat(v) for v in pedo]
    return pedo


def featuresWalkNames():
    features = {}

    moments = ["accel_var", "accel_skew", "accel_kurt", "accel_moment_5", "accel_mean",
               "accel_x_var", "accel_x_kurt", "accel_y_var", "accel_y_kurt", "accel_z_var", "accel_z_kurt",
               "accel_x_zcr", "accel_y_zcr", "accel_z_zcr"]
    features['moments'] = moments
    # [(0, 1.5), (1.5, 3), (3, 5), (5, 7), (7, 10), (10, 14), (14, float('inf'))]
    features['fourier'] = ["fourier_1.5hz_min", "fourier_3hz_min","fourier_5hz_min","fourier_7hz_min","fourier_10hz_min","fourier_14hz_min","fourier_remain_min",
                           "fourier_1.5hz_mean", "fourier_3hz_mean", "fourier_5hz_mean", "fourier_7hz_mean", "fourier_10hz_mean", "fourier_14hz_mean", "fourier_remain_mean",
                           "fourier_1.5hz_std", "fourier_3hz_std", "fourier_5hz_std", "fourier_7hz_std", "fourier_10hz_std", "fourier_14hz_std", "fourier_remain_std"]

    features['area'] = ["bounding_ellipse_area"]

    features['tkeo'] = ["mean_tkeo", "std_tkeo"]

    features['entropy'] = ["x_ent", "y_ent", "xy_cross_ent", "xy_cross_corr", "xy_mutual_info",
                           "z_ent", "xz_cross_ent", "xz_cross_corr", "xz_mutual_info",
                           "yz_cross_ent", "yz_cross_corr", "yz_mutual_info"]

    dynamic = ["xTau", "yTau", "zTau", "higuchi_x", "higuchi_y", "higuchi_z", "hurst_x", "hurst_y", "hurst_z", "pfd_x", "pfd_y", "pfd_z", "dfa_x", "dfa_y", "dfa_z", "rpde_x", "rpde_y", "rpde_z",
              "Lyap_Eck_x_1", "Lyap_Eck_x_2", "Lyap_Eck_x_3", "Lyap_Eck_x_4", "Lyap_Eck_x_5", "Lyap_Eck_x_6", "LLE_Rosen_x",
               "Lyap_Eck_y_1", "Lyap_Eck_y_2", "Lyap_Eck_y_3", "Lyap_Eck_y_4", "Lyap_Eck_y_5", "Lyap_Eck_y_6" , "LLE_Rosen_y",
                "Lyap_Eck_z_1", "Lyap_Eck_z_2", "Lyap_Eck_z_3", "Lyap_Eck_z_4", "Lyap_Eck_z_5", "Lyap_Eck_z_6", "LLE_Rosen_z" ]

    features['dynamic'] = dynamic

    info_dynamic = ["x_svd_ent", "y_svd_ent", "z_svd_ent", "x_fisher", "y_fisher", "z_fisher", "x_spect_ent", "y_spect_ent", "z_spect_ent",
                    "x_ap_ent", "y_ap_ent", "z_ap_ent", "x_samp_ent_1", "x_samp_ent_2", "x_samp_ent_3", "x_samp_ent_4",
                    "y_samp_ent_1", "y_samp_ent_2", "y_samp_ent_3", "y_samp_ent_4", "z_samp_ent_1", "z_samp_ent_2", "z_samp_ent_3", "z_samp_ent_4"]
    features['info_dynamic'] = info_dynamic

    features['hjorth'] = ["x_activity", "x_complexity", "x_morbidity",
                          "y_activity", "y_complexity", "y_morbidity",
                          "z_activity", "z_complexity", "z_morbidity"]

    mpower = [
        "meanX", "sdX", "modeX", "skewX", "kurX", "q1X",
                           "medianX", "q3X", "iqrX", "rangeX", "acfX", "zcrX",
                           "dfaX", "cvX", "tkeoX", "F0X", "P0X","F0FX", "P0FX",
                           "medianF0FX", "sdF0FX", "tlagX", "meanY", "sdY", "modeY",
                           "skewY", "kurY", "q1Y", "medianY", "q3Y", "iqrY",
                           "rangeY", "acfY", "zcrY", "dfaY", "cvY", "tkeoY",
                           "F0Y", "P0Y", "F0FY", "P0FY", "medianF0FY", "sdF0FY",
                           "tlagY", "meanZ", "sdZ", "modeZ", "skewZ", "kurZ", "q1Z",
                           "medianZ", "q3Z", "iqrZ", "rangeZ", "acfZ", "zcrZ", "dfaZ",
                           "cvZ", "tkeoZ", "F0Z", "P0Z", "F0FZ", "P0FZ", "medianF0FZ",
                           "sdF0FZ", "tlagZ", "meanAA", "sdAA", "modeAA", "skewAA", "kurAA",
                           "q1AA", "medianAA", "q3AA", "iqrAA", "rangeAA", "acfAA", "zcrAA",
                           "dfaAA", "cvAA", "tkeoAA", "F0AA", "P0AA", "F0FAA", "P0FAA",
                           "medianF0FAA", "sdF0FAA", "tlagAA","meanAJ", "sdAJ", "modeAJ",
                           "skewAJ", "kurAJ", "q1AJ", "medianAJ", "q3AJ", "iqrAJ", "rangeAJ",
                           "acfAJ", "zcrAJ", "dfaAJ", "cvAJ", "tkeoAJ", "F0AJ", "P0AJ",
                           "F0FAJ", "P0FAJ", "medianF0FAJ", "sdF0FAJ", "tlagAJ",
                           "corXY", "corXZ", "corYZ"
    ]
    features['mpower'] = mpower
    return features
