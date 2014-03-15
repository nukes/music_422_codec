"""
- mdct.py -- Computes reasonably fast MDCT/IMDCT using numpy FFT/IFFT
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import math
import numpy as np


def MDCT(data, a, b, isInverse=False):
    """ Fast MDCT algorithm for window length a+b following pp. 141-143 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    (and where 2/N factor is included in forward transform instead of inverse)
    a: left half-window length
    b: right half-window length
    """
    N = a + b
    n_0 = (b + 1.) / 2.
    n = np.linspace(0, N-1., N)

    out = np.array([])

    if isInverse:
        # Actually, extend K and the data
        K = np.linspace(0, N-1, N)
        mirror = -data[::-1]
        data = np.append(data, mirror)

        twiddle_exponents = np.multiply(K, 1j*2.*np.pi*n_0/N)
        twiddle = np.exp(twiddle_exponents)
        twiddle_signal = np.multiply(data, twiddle)

        ifft_data = np.fft.ifft(twiddle_signal)

        shifted_n = np.add(n, n_0)
        post_twiddle_exponents = np.multiply(shifted_n, 1j * np.pi / N)
        post_twiddle = np.exp(post_twiddle_exponents)

        out = np.multiply(ifft_data, post_twiddle)
        out = np.real(out)
        out = np.multiply(N, out)

    else:
        K = np.linspace(0, N/2. - 1, N/2)
        twiddle_exponents = np.multiply(n, -1j * np.pi / N)
        twiddle = np.exp(twiddle_exponents)
        twiddle_signal = np.multiply(data, twiddle)

        fft_data = np.fft.fft(twiddle_signal)
        
        shifted_K = np.add(K, 0.5)
        post_twiddle_exponents = np.multiply(shifted_K, -1j * (2*np.pi/N) * n_0)
        post_twiddle = np.exp(post_twiddle_exponents)

        out = np.multiply(2./N, fft_data[:N/2.])
        out = np.multiply(post_twiddle, out)
        out = np.real(out)

    return out


def IMDCT(data, a, b):
    return MDCT(data, a, b, isInverse=True)
