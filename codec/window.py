import numpy as np


def SineWindow(dataSampleArray):
    ''' Modulate the sample data block with a sine window '''
    N = float(len(dataSampleArray))
    n = np.linspace(0.5, N+0.5, N)
    return dataSampleArray * np.sin(n * np.pi / N)


def HanningWindow(dataSampleArray):
    ''' Modulate the sample data block with a Hanning window '''
    N = float(len(dataSampleArray))
    n = np.linspace(0.5, N+0.5, N)
    return dataSampleArray * 0.5 * (1 - np.cos(2*n*np.pi/N))


def KBDWindow(dataSampleArray, alpha=4.):
    ''' Performs a KBD window (with tuning param) on the sample '''
    N = float(len(dataSampleArray))
    window = np.zeros(N)
    s = 0.

    for i in range(int(N/2)):
        s += np.i0(np.pi * alpha * np.sqrt(1.-(4.*i/N-1.)**2))
        window[i] = s

    s += np.i0(np.pi * alpha * np.sqrt(1.-(4.*N/2./N-1.)**2))

    for i in range(int(N/2)):
        window[i] = np.sqrt(window[i]/s)
        window[N-i-1] = window[i]

    return dataSampleArray * window
