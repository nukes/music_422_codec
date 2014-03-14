"""
codec.py -- The actual encode/decode functions for the perceptual audio codec

-----------------------------------------------------------------------
© 2009 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

import sys

import numpy as np  # used for arrays

# used by Encode and Decode
from codec.mdct import MDCT,IMDCT  # fast MDCT implementation (uses numpy FFT)
from codec.quantize import *  # using vectorized versions (to use normal versions, uncomment lines 18,67 below defining vMantissa and vDequantize)

# used only by Encode
#from smr import CalcSMRs  # calculates SMRs for each scale factor band
from codec.window import compose_kbd_window
from codec.bitalloc import BitAlloc as OurBitAlloc #allocates bits to scale factor bands given SMRs
from provided.bitalloc import BitAlloc as TheirBitAlloc
from codec.psychoac import *

def Decode(scaleFactor,bitAlloc,mantissa,overallScaleFactor,codingParams):
    """Reconstitutes a single-channel block of encoded data into a block of
    signed-fraction data based on the parameters in a PACFile object"""

    rescaleLevel = 1.*(1<<overallScaleFactor)
    halfN = codingParams.nMDCTLines
    N = 2*halfN
    # vectorizing the Dequantize function call
    vDequantize = np.vectorize(Dequantize)

    # reconstitute the first halfN MDCT lines of this channel from the stored data
    mdctLine = np.zeros(halfN,dtype=np.float64)
    iMant = 0
    for iBand in range(codingParams.sfBands.nBands):
        nLines =codingParams.sfBands.nLines[iBand]
        if bitAlloc[iBand]:
            mdctLine[iMant:(iMant+nLines)]=vDequantize(scaleFactor[iBand], mantissa[iMant:(iMant+nLines)],codingParams.nScaleBits, bitAlloc[iBand])
        iMant += nLines
    mdctLine /= rescaleLevel  # put overall gain back to original level


    # IMDCT and window the data for this channel
    #data = SineWindow( IMDCT(mdctLine, halfN, halfN) )  # takes in halfN MDCT coeffs
    half_long = 512
    half_short = 64
    window_state = codingParams.window_state
    if window_state == 0:
        half_n = half_long
        samples = IMDCT(mdctLine, half_long, half_long)
        win_samples = compose_kbd_window(samples, half_long, half_long, 4., 4.)
        #win_samples = SineWindow(samples)
    elif window_state == 1:
        half_n = (half_long + half_short) / 2
        samples = IMDCT(mdctLine, half_long, half_short)
        win_samples = compose_kbd_window(samples, half_long, half_short, 4., 6.)
    elif window_state == 2:
        half_n = half_short
        samples = IMDCT(mdctLine, half_short, half_short)
        win_samples = compose_kbd_window(samples, half_short, half_short, 6., 6.)
    elif window_state == 3:
        half_n = (half_long + half_short) / 2
        samples = IMDCT(mdctLine, half_short, half_long)
        win_samples = compose_kbd_window(samples, half_short, half_long, 6., 4.)
    else:
        raise ValueError('Unknown window state:' + str(window_state))

    # end loop over channels, return reconstituted time samples (pre-overlap-and-add)
    return win_samples


def Encode(data,codingParams):
    """Encodes a multi-channel block of signed-fraction data based on the parameters in a PACFile object"""
    scaleFactor = []
    bitAlloc = []
    mantissa = []
    overallScaleFactor = []

    # loop over channels and separately encode each one
    for iCh in range(codingParams.nChannels):
        (s,b,m,o) = EncodeSingleChannel(data[iCh],codingParams)
        scaleFactor.append(s)
        bitAlloc.append(b)
        mantissa.append(m)
        overallScaleFactor.append(o)
    # return results bundled over channels
    return (scaleFactor,bitAlloc,mantissa,overallScaleFactor)


def EncodeSingleChannel(data,codingParams):
    """Encodes a single-channel block of signed-fraction data based on the parameters in a PACFile object"""

    # prepare various constants
    halfN = codingParams.nMDCTLines
    N = 2*halfN
    nScaleBits = codingParams.nScaleBits
    maxMantBits = (1<<codingParams.nMantSizeBits)  # 1 isn't an allowed bit allocation so n size bits counts up to 2^n
    if maxMantBits>16: maxMantBits = 16  # to make sure we don't ever overflow mantissa holders
    sfBands = codingParams.sfBands
    # vectorizing the Mantissa function call
#    vMantissa = np.vectorize(Mantissa)
    window_state = codingParams.window_state

    half_long = 512
    half_short = 64
    samples = data
    if window_state == 0:
        halfN = half_long
        win_samples = compose_kbd_window(samples, half_long, half_long, 4., 4.)
        mdctLines = MDCT(win_samples, half_long, half_long)[:halfN]
    elif window_state == 1:
        halfN = (half_long + half_short) / 2
        win_samples = compose_kbd_window(samples, half_long, half_short, 4., 6.)
        mdctLines = MDCT(win_samples, half_long, half_short)[:halfN]
    elif window_state == 2:
        halfN = half_short
        win_samples = compose_kbd_window(samples, half_short, half_short, 6., 6.)
        mdctLines = MDCT(win_samples, half_short, half_short)[:halfN]
    elif window_state == 3:
        halfN = (half_long + half_short) / 2
        win_samples = compose_kbd_window(samples, half_short, half_long, 6., 4.)
        mdctLines = MDCT(win_samples, half_short, half_long)[:halfN]
    else:
        raise ValueError('Unknown window state:' + str(window_state))



    # compute target mantissa bit budget for this block of halfN MDCT mantissas
    bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate

    if window_state == 2:
        bitBudget *= 2

    bitBudget -=  nScaleBits*(sfBands.nBands +1)  # less scale factor bits (including overall scale factor)
    bitBudget -= codingParams.nMantSizeBits*sfBands.nBands  # less mantissa bit allocation bits

    bitBudget -= 2

    bitBudget = int(np.floor(bitBudget))

    # window data for side chain FFT and also window and compute MDCT
    timeSamples = data
    #mdctTimeSamples = SineWindow(data)
    #mdctLines = MDCT(mdctTimeSamples, halfN, halfN)[:halfN]

    # compute overall scale factor for this block and boost mdctLines using it
    maxLine = np.max( np.abs(mdctLines) )
    overallScale = ScaleFactor(maxLine,nScaleBits)  #leading zeroes don't depend on nMantBits
    mdctLines *= (1<<overallScale)

    # compute the mantissa bit allocations
    # compute SMRs in side chain FFT
    SMRs = CalcSMRs(timeSamples, mdctLines, overallScale, codingParams.sampleRate, sfBands)

    #print SMRs
    
    # perform bit allocation using SMR results

    bitAlloc = OurBitAlloc(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs)

    #print bitAlloc

    # given the bit allocations, quantize the mdct lines in each band
    scaleFactor = np.empty(sfBands.nBands,dtype=np.int32)
    nMant=halfN
    for iBand in range(sfBands.nBands):
        if not bitAlloc[iBand]: nMant-= sfBands.nLines[iBand]  # account for mantissas not being transmitted
    mantissa=np.empty(nMant,dtype=np.int32)
    iMant=0
    for iBand in range(sfBands.nBands):
        lowLine = sfBands.lowerLine[iBand]


        #highLine = sfBands.upperLine[iBand] + 1  # extra value is because slices don't include last value
        highLine = lowLine + sfBands.nLines[iBand]


        nLines= sfBands.nLines[iBand]
        if lowLine > highLine: #
            scaleLine = 1 #
            print 'empty'
        else: #
            scaleLine = np.max(np.abs( mdctLines[lowLine:highLine] ) )
        scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc[iBand])
        if bitAlloc[iBand]:
            mantissa[iMant:iMant+nLines] = vMantissa(mdctLines[lowLine:highLine],scaleFactor[iBand], nScaleBits, bitAlloc[iBand])
            iMant += nLines
    # end of loop over scale factor bands

    # return results
    return (scaleFactor, bitAlloc, mantissa, overallScale)



