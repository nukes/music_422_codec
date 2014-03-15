""" Library of signal-windowing functions. """

import numpy as np


def SineWindow(dataSampleArray):
    """ Modulate the sample data block with a sine window """
    N = float(len(dataSampleArray))
    n = np.linspace(0.5, N+0.5, N)
    return dataSampleArray * np.sin(n * np.pi / N)


def HanningWindow(dataSampleArray):
    """ Modulate the sample data block with a Hanning window """
    N = float(len(dataSampleArray))
    n = np.linspace(0.5, N+0.5, N)
    return dataSampleArray * 0.5 * (1 - np.cos(2*n*np.pi/N))


def KBDWindow(dataSampleArray, alpha=4.):
    """ Performs a Kaiser-Bessel Derived window (with tuning param, alpha)
    on the sample. """
    N = float(len(dataSampleArray))
    
    # Create the numerators for half the KBD
    t = np.arange(int(N/2))
    window = np.i0(np.pi * alpha * np.sqrt(1.-(4.*t/N-1.)**2))
    window = np.cumsum(window)

    # Make sure to add the boundary value for the denominator calculation
    den = window[-1] + np.i0(np.pi * alpha * np.sqrt(1.-(4.*N/2./N-1.)**2))

    # Perform the sqrt. operations to get the 'derived' nature.
    # Afterwards, mirror the window to get a symmetric window.
    window = np.sqrt(window/den)
    window = np.concatenate((window, window[::-1]), axis=0)

    return dataSampleArray * window


def compose_kbd_window(dataSampleArray, left, right, left_alpha=4., right_alpha=4.):
    """ Compose a hybrid Kaiser-Bessel Derived window for block-switched MDCT
    windows. Parameters left, right control the size of the window segments,
    while the alpha parameters tune the frequency selectivity vs. rolloff. """

    # Make sure that left + right is the size of the window provided
    if left + right != len(dataSampleArray):
        msg = 'Signal size, {} , must match the composed size left+right: {}' \
               .format(str(len(dataSampleArray)), str(left+right))
        raise ValueError(msg)
    
    # Create a window for size left
    a_ones = np.ones(2*left)
    a_window = KBDWindow(a_ones, alpha=left_alpha)[:left]

    # Create a window for size right
    b_ones = np.ones(2*right)
    b_window = KBDWindow(b_ones, alpha=right_alpha)[:right]
    b_window = b_window[::-1]

    return dataSampleArray * np.concatenate([a_window, b_window])
