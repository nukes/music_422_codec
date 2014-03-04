''' Very dirty routines for peak detection. '''

import numpy as np


def onset_in_block(signal):
    ''' Accepts some block of the signal as input. This function performs an
    an FFT on the block and computes a weighted spectral energy measure. the
    weights favor higher energies, as signal onset usually contains high
    frequency components for sharp signal transitions.

    This function will output a Boolean value. '''
    N = signal.size
    thresh = 0.05
    fft = np.fft.fft(block)[:N/2]
    
    energy = np.sum(np.linspace(0, 1, N/2) * np.abs(fft)**2) / (N/2.)
    return energy > thresh
