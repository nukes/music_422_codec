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
    
    # Create the numerators for half the KBD
    t = np.arange(int(N/2))
    print ">> ", len(t)
    window = np.i0(np.pi * alpha * np.sqrt(1.-(4.*t/N-1.)**2))
    window = np.cumsum(window)

    # Make sure to add the boundary value for the denominator calculation
    den = window[-1] + np.i0(np.pi * alpha * np.sqrt(1.-(4.*N/2./N-1.)**2))

    print ">>> ", len(window)

    # Actually perform the operations to get the 'derived' nature
    # And then mirror the output to get a symmetric window
    window = np.sqrt(window/den)

    print ">> ", len(window)
    window = np.concatenate((window, window[::-1]), axis=0)
    print ">>> ", len(window)

    return dataSampleArray * window


def compose_kbd_window(dataSampleArray, a, b, a_alpha=4., b_alpha=4.):
    ''' Compose a hybrid KBD window for block-switched MDCT windows. '''

    # Make sure that a + b is the size of the window provided
    if a + b != len(dataSampleArray):
        raise ValueError('Signal size must match the composed size a+b.')
    
    # Create a window for size A
    a_ones = np.ones(2*a)
    a_window = KBDWindow(a_ones, alpha=a_alpha)[:a]

    # Create a window for size b
    b_ones = np.ones(2*b)
    b_window = KBDWindow(b_ones, alpha=b_alpha)[:b]
    b_window = b_window[::-1]

    return dataSampleArray * np.concatenate([a_window, b_window])
