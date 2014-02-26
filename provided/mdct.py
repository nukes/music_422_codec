"""
- mdct.py -- Computes reasonably fast MDCT/IMDCT using numpy FFT/IFFT
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np
import time

### Problem 1.a ###
def MDCTslow(data, a, b, isInverse=False):
    """
    Slow MDCT algorithm for window length a+b following pp. 130 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    (and where 2/N factor is included in forward transform instead of inverse)
    a: left half-window length
    b: right half-window length
    """

    ### YOUR CODE STARTS HERE ###

    # Near perfect reconstruction conditions are met by utilizing an analysis
    # window wa = 1 and a synthesis window ws = 1.2 over the ranges of interest.

    # Reconstruction is possible via the overlap-add method: a signal is decomposed
    # into a certain number of blocks (in the test case, 4 blocks of 8 samples with
    # 4-sample overlap). An MDCT is taken of each block, followed by an imDCT. 
    # By adding all the blocks with a shift of 4-samples, the original signal can be   
    # reconstructed. Since the MDCT is not orthogonal, as in the case of the DFT, 
    # the first and the last four samples are to be eliminated
    
    out = []
    N = a + b
    n0 = (1.0*b+1)/2
    
    if isInverse:
        # iMDCT: x[n] = 1/2 * 2 * sum( X[k]*cos(2*np.pi/N*(n+n0)*(k+1/2)) )
        k, n = np.meshgrid(np.linspace(0,N/2-1,N/2), np.linspace(0,N-1,N))
        xn = np.sum(data * np.cos(np.pi*2/N*(n+n0)*(k+0.5)), axis=1)
        out = xn
    else:
        # MDCT: X[k] = 2/N * sum( x[n]*cos(2*np.pi/N*(n+n0)*(k+1/2)) )
        n, k = np.meshgrid(np.linspace(0,N-1,N), np.linspace(0,N/2-1,N/2))
        Xk = 2.0/N * np.sum(data*np.cos(np.pi*2/N*(n+n0)*(k+0.5)), axis=1)
        out = Xk

    return out

    ### YOUR CODE ENDS HERE ###

### Problem 1.c ###
def MDCT(data, a, b, isInverse=False):
    """
    Fast MDCT algorithm for window length a+b following pp. 141-143 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    (and where 2/N factor is included in forward transform instead of inverse)
    a: left half-window length
    b: right half-window length
    """

    ### YOUR CODE STARTS HERE ###

    N = a + b
    halfN = N/2
    no = (b+1.)/2
        
    if isInverse:
        preTwiddle = np.arange(N,dtype=np.float64)
        phase = 2j*np.pi*no/N
        preTwiddle = np.exp(phase*preTwiddle)
        postTwiddle = np.linspace(no,N+no-1,N)
        phase = 1j*np.pi/N
        postTwiddle = N*np.exp(phase*postTwiddle)
        return (postTwiddle * np.fft.ifft(preTwiddle*np.concatenate((data,-data[::-1])))).real
    else:
        preTwiddle = np.arange(N,dtype=np.float64)
        phase = -1j*np.pi/N
        preTwiddle = np.exp(phase * preTwiddle) * data
        postTwiddle = np.linspace(0.5,halfN-0.5,halfN)
        phase = -2j*np.pi*no/N        
        postTwiddle = np.exp(phase*postTwiddle)*2./N
        return (postTwiddle*np.fft.fft(preTwiddle)[:halfN]).real
        
    return out

    ### YOUR CODE ENDS HERE ###

def IMDCT(data,a,b):

    ### YOUR CODE STARTS HERE ###
    
    return MDCT(data,a,b,isInverse=True)

    ### YOUR CODE ENDS HERE ###

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    ### YOUR TESTING CODE STARTS HERE ###
    x = np.array([4,4,4,4,3,1,-1,-3,0,1,2,3])

    """
    print("Testing MDCT slow: \n")
    y0 = MDCTslow(np.concatenate([[0,0,0,0],x[0:4]]),4,4)
    print("First pass: ", y0)
    y1 = MDCTslow(x[0:8],4,4)
    print("Second pass: ", y1)
    y2 = MDCTslow(x[4:12],4,4)
    print("Third pass: ", y2)
    y3 = MDCTslow(np.concatenate([x[8:],[0,0,0,0]]),4,4)
    print("Fourth pass: ", y3)
    print("\n")
    x0 = MDCTslow(y0,4,4, True)
    print("First pass: ", x0)
    x1 = MDCTslow(y1,4,4, True)
    print("Second pass: ", x1)
    x2 = MDCTslow(y2,4,4, True)
    print("Third pass: ", x2)
    x3 = MDCTslow(y3,4,4,True)
    print("Fourth pass: ", x3)
    """
        
    ### Problem 1.b ###
    """
    N = len(x)
    priorBlock = np.array([0,0,0,0])
    finalX = np.zeros(N+8)
    
    # cycle through blocks
    for i in range(0,N/4):
        newX = np.concatenate([priorBlock, x[i*4:i*4+4]])
        priorBlock = x[i*4:i*4+4]
        y = MDCTslow(MDCTslow(newX, 4, 4),4,4,True)
        finalX[i*4:i*4+8] = y + finalX[i*4:i*4+8]
    
    # final cycle
    newX = np.concatenate([priorBlock, [0,0,0,0]])
    y = MDCTslow(MDCTslow(newX, 4, 4),4,4,True)
    finalX[-8:] = y + finalX[-8:]
    out = finalX[4:-4]    # eliminate leading and trailing 0's
        
    print(out)   
    # output is approximately equal to input (0 is transformed into a low 
    # order quantity)

    # a priorBlock array had to be initialized to [0,0,0,0] for the first
    # block transform

    # the last block is obtained by zero-padding [0,0,0,0]. 
    # 4-sample delay is introduced, but can eliminated via post-processing
    """

    ### END 1.b ###
    
    """
    print("\n")
    print("Testing MDCT fast: \n")
    
    y0 = MDCT(np.concatenate([[0,0,0,0],x[0:4]]),4,4)
    print("First pass: ", y0)
    y1 = MDCT(x[0:8],4,4)
    print("Second pass: ", y1)
    y2 = MDCT(x[4:12],4,4)
    print("Third pass: ", y2)
    y3 = MDCT(np.concatenate([x[8:],[0,0,0,0]]),4,4)
    print("Fourth pass: ", y3)
    print("\n")
    x0 = MDCT(y0,4,4, True)
    print("First pass: ", x0)
    x1 = MDCT(y1,4,4, True)
    print("Second pass: ", x1)
    x2 = MDCT(y2,4,4, True)
    print("Third pass: ", x2)
    x3 = MDCT(y3,4,4,True)
    print("Fourth pass: ", x3)
    """

    """
    x = np.linspace(0,4096,4096)
    t1 = time.time()
    y = MDCTslow(x,2048,2048)
    t2 = time.time()
    print("TIME: %s\n" % (t2-t1))
    t1 = time.time()
    y = MDCT(x,2048,2048)
    t2 = time.time()
    print("TIME: %s\n" % (t2-t1))
    """
    
    ### YOUR TESTING CODE ENDS HERE ###

