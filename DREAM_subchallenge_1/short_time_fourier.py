import numpy as np

def getShortTimeFourier1D(signal, fourier_length, step=None):
    if step is None:
        step = fourier_length
    stft = []

    num_steps = (signal.shape[0]-fourier_length)//step

    for s in range(0, num_steps):
        fftamplitude = [np.absolute(np.fft.rfft(signal[s*step:s*step+fourier_length, i])) for i in range(signal.shape[1])]
        stft.append(fftamplitude)
    stft = np.array(stft).swapaxes(0,1).swapaxes(1,2)
    return stft