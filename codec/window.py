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
    raise NotImplementedError
