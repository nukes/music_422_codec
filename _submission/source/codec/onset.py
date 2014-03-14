''' Very dirty routines for peak detection. '''

import numpy as np


def onset_in_block(signal, window_state, short_runs):
    ''' Accepts some block of the signal as input. This function performs an
    an FFT on the block and computes a weighted spectral energy measure. the
    weights favor higher energies, as signal onset usually contains high
    frequency components for sharp signal transitions.

    This function will output a Boolean value. '''
    N = signal.size

    if window_state == 2 and short_runs > 4:
        thresh = 1.0
        short_runs = 0
    elif window_state == 2 and short_runs <= 4:
        thresh = 0.05
        short_runs += 1
    else:
        thresh = 0.05
        short_runs = 0

    fft = np.fft.fft(signal)[:N/2]

    weights = np.zeros(N/2)
    weights[N/4:] = 1.
    
    energy = np.sum(weights * np.abs(fft)**2) / (N/2.)
    return energy > thresh, short_runs


class WindowState(object):

    def __init__(self):
        self.state = 0

    def step(self, is_onset):
        ''' External method to advance the state machine's state. '''
        if is_onset:
            return self._transient()
        else:
            return self._no_transient()

    def _transient(self):
        ''' Internal method to transition state based on onset presence '''
        if self.state == 0:
            self.state = 1
        elif self.state == 1:
            self.state = 2
        elif self.state == 2:
            self.state = 2
        elif self.state == 3:
            self.state = 1
        return self.state

    def _no_transient(self):
        ''' Internal method to transition state when no onset is present. '''
        if self.state == 0:
            self.state = 0
        elif self.state == 1:
            self.state = 2
        elif self.state == 2:
            self.state = 3
        elif self.state == 3:
            self.state = 0
        return self.state
