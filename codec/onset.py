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
    fft = np.fft.fft(signal)[:N/2]
    
    energy = np.sum(np.linspace(0, 1, N/2) * np.abs(fft)**2) / (N/2.)
    return energy > thresh


class WindowState(object):

    def __init__(self):
        self.state = 0

    def step(self, is_onset):
        ''' External method to advance the state machine's state. '''
        if is_onset:
            return self.transient()
        else:
            return self.no_transient()

    def transient(self):
        ''' Internal method to transition state based on onset presence '''
        if self.state == 0:
            self.state = 1
        elif self.state == 1:
            self.state = 2
        elif self.state == 2:
            self.state = 2
        elif self.state == 3:
            self.state = 2
        return self.state

    def no_transient(self):
        ''' Internal method to transition state when no onset is present. '''
        if self.state == 0:
            self.state = 0
        elif self.state == 1:
            self.state = 2
        elif self.state == 2:
            self.state = 3
        elif self.state == 3:
            self.state = 4
        return self.state


