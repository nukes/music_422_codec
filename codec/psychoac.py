import numpy as np
from mdct import *
from window import *

import matplotlib.pyplot as plt
import scipy.io.wavfile as wave

def SPL(intensity): 
    ''' Return SPL in dB for the given intensity vector '''
    intensity = np.maximum(np.finfo(float).eps, intensity)
    spl = 96. + 10 * np.log10(intensity)
    return np.maximum(-30, spl)


def Intensity(spl): 
    ''' Return intensity for the given SPL vector '''
    spl = np.clip(spl, -30, 96)
    return np.power(10, (spl - 96) / 10.0)


def Thresh(f): 
    ''' Returns the threshold in quiet measured in SPL at frequency f (in Hz) '''
    f = np.clip(f, 10, np.inf)
    khz = np.divide(f, 1000.)  # Convert frequencies to kHz to match the book

    a = np.power(khz, -0.8)
    a = np.multiply(a, 3.64)

    b = np.subtract(khz, 3.3)
    b = np.square(b)
    b = np.multiply(b, -0.6)
    b = np.exp(b)
    b = np.multiply(b, -6.5)

    c = np.power(khz, 4)
    c = np.multiply(c, 0.001)

    thresh = np.add(a, b)
    thresh = np.add(thresh, c)

    return thresh


def Bark(f): 
    ''' Returns the bark-scale frequency for input frequency f (in Hz) '''
    khz = np.divide(f, 1000.)

    a = np.multiply(khz, 0.76)
    a = np.arctan(a)
    a = np.multiply(a, 13.)

    b = np.divide(khz, 7.5)
    b = np.square(b)
    b = np.arctan(b)
    b = np.multiply(b, 3.5)

    return np.add(a, b)


class Masker: 
    """ 
    a Masker whose masking curve drops linearly in Bark beyond 0.5 Bark from the 
    masker frequency
    """
    
    def __init__(self,f,SPL,isTonal=True): 
        ''' Find the masker's location in Bark, from the given frequency.
            Store the SPL of the masker frequency, used in the masking curve.
            Determine the type of masker, which determines the drop.
        '''
        self.f = f
        self.z = Bark(f)
        self.spl = SPL
        self.isTonal = isTonal
        self.drop = 15. if isTonal else 15.5
        ## self.drop = 6.025 + 0.275*self.z if isTonal else 2.025+0.175*self.z

    def IntensityAtFreq(self,freq): 
        ''' The intensity of this masker at frequency freq '''
        return self.IntensityAtBark(Bark(freq))

    def IntensityAtBark(self,z): 
        ''' The intensity of this masker at Bark location z '''
        dz = z - self.z
        spread = (0.37*max(self.spl-40., 0))
        spread *= float(dz>=0)
        spread += -27.
        spread *= (abs(dz)-0.5)
        spread *= float(abs(dz) > 0.5)
        spl = self.spl + spread - self.drop
        return Intensity(spl)

    def vIntensityAtFreq(self,fVec):
        ''' This was convenient, so I defined it. '''
        return self.vIntensityAtBark(Bark(fVec))

    def vIntensityAtBark(self,zVec): 
        ''' The intensity of this masker at vector of Bark locations zVec '''
        dz = np.subtract(zVec, self.z)
        abs_dz = np.absolute(dz)

        slope_side = np.greater_equal(dz, 0)
        leveling = (0.37*max(self.spl-40., 0))

        spread = np.multiply(slope_side, leveling)
        spread = np.subtract(spread, 27.)
        spread = np.multiply(spread, np.subtract(abs_dz, 0.5))
        spread = np.multiply(spread, np.greater(abs_dz, 0.5))

        spls = np.add(self.spl, spread)
        spls = np.subtract(spls, self.drop)

        return Intensity(spls)


# Default data for 25 scale factor bands based on the traditional 25 critical bands
cbFreqLimits = np.array([100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270,
                         1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300,
                         6400, 7700, 9500, 12000, 15500, float('inf')])


# Trying summin'
cbFreqCenters = [50, 150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850,
                 2150, 2500, 2900, 3400, 4000, 4800, 5800, 7000, 8500, 10500, 13500, 20000]


def AssignMDCTLinesFromFreqLimits(nMDCTLines, sampleRate, flimit = cbFreqLimits): 
    """
    Assigns MDCT lines to scale factor bands for given sample rate and number 
    of MDCT lines using predefined frequency band cutoffs passed as an array 
    in flimit (in units of Hz). If flimit isn't passed it uses the traditional 
    25 Zwicker & Fastl critical bands as scale factor bands.
    """

    mdct_lines = np.arange(nMDCTLines)
    mdct_lines = np.add(mdct_lines, 0.5)
    mdct_lines = np.divide(mdct_lines, nMDCTLines)
    mdct_lines = np.multiply(mdct_lines, sampleRate/2)

    assignment = []

    previous_size = 0
    for limit in flimit:
        size = np.argmax(np.less(limit, mdct_lines))
        size = nMDCTLines if size == 0 else size  # Handle infinite freq center
        assignment.append(size - previous_size)
        previous_size = size

    return np.array(assignment)


def findPeaks(fftIntensity, thresh = 7.0):
    
    peaks = []
    dataSPL = SPL(SPL(fftIntensity))

    dataDiff = np.concatenate([[1],np.diff(dataSPL)])
    dataDiff = np.concatenate([dataDiff,[-1]])
    peakIndices = (dataDiff[0:-1] > 0) & (dataDiff[1:] < 0)

    return peakIndices


class ScaleFactorBands: 
    '''
    A set of scale factor bands (each of which will share a scale factor and a 
    mantissa bit allocation) and associated MDCT line mappings.
    
    Instances know the number of bands nBands; the upper and lower limits for 
    each band lowerLimit[i in range(nBands)], upperLimit[i in range(nBands)]; 
    and the number of lines in each band nLines[i in range(nBands)] 
    '''
    
    def __init__(self, nLines):
        ''' Assigns MDCT lines to scale factor bands based on a vector of 
            frequency line allocation 
        '''
        self.nBands = len(nLines)
        self.nLines = np.array(nLines)
        self.lowerLine = np.append(0, np.cumsum(nLines)[:-1])
        self.upperLine = np.add(self.nLines, np.subtract(self.lowerLine, 1))


def CalcSMRs(data, MDCTdata, MDCTscale, sampleRate, sfBands): 
    """
    Set SMR for each critical band in sfBands.
    
    Arguments:
                data:       is an array of N time domain samples 
                MDCTdata:   is an array of N/2 MDCT frequency lines for the data
                            in data which have been scaled up by a factor 
                            of 2^MDCTscale 
                MDCTscale:  is an overall scale factor for the set of MDCT 
                            frequency lines 
                sampleRate: is the sampling rate of the time domain samples 
                sfBands:    points to information about which MDCT frequency lines
                            are in which scale factor band 
                            
    Returns:
                SMR[sfBands.nBands] is the maximum signal-to-mask ratio in each 
                                    scale factor band
    
    Logic: 
                Performs an FFT of data[N] and identifies tonal and noise maskers. 
                Sums their masking curves with the hearing threshold at each MDCT 
                frequency location to the calculate absolute threshold at those 
                points. Then determines the maximum signal-to-mask ratio within 
                each critical band and returns that result in the SMR[] array.
    """

    # Convenience: this is used everywhere
    N = len(data)
    Nlines = N/2

    # Window the signal and perform an FFT
    dft = np.fft.fft(HanningWindow(data))[:Nlines]
    dft_intensity = (8./3. * 4./(N**2)) * np.abs(dft)**2
    dft_spl = SPL(dft_intensity)

    # Transform the MDCT data into the SPL
    # Scale down the MDCT's SPL by 6dB per bit
    mdct_scale_unwind = 6.02 * MDCTscale
    mdct_spl = SPL(4. * MDCTdata**2) - mdct_scale_unwind

    # Identify tonal and noise maskers
    # Get tonal maskers
    maskers = []
    spl_peaks = findPeaks(dft_intensity)

    for i, isPeak in enumerate(spl_peaks):
        if i+1 == Nlines: break
        if isPeak:
            if i == 0:
                intensity_sum = dft_intensity[i] + dft_intensity[i+1]
                f = .5*sampleRate/Nlines*(i*dft_intensity[i] + (i+1)*dft_intensity[i+1]) / intensity_sum
            else:
                intensity_sum = dft_intensity[i] + dft_intensity[i-1] + dft_intensity[i+1]
                f = .5*sampleRate/Nlines*(i*dft_intensity[i] + (i-1)*dft_intensity[i-1] + (i+1)*dft_intensity[i+1]) / intensity_sum
            spl = SPL(intensity_sum)
            if spl > Thresh(f):    # Eliminate if below quiet threshold
                maskers.append(Masker(f,spl))
            dft_intensity[i] = dft_intensity[i+1] = 0    # eliminate for noise masker calculation
            if i != 0:
                dft_intensity[i-1] = 0

    # Get noise maskers: sum over energy not in tonal peaks
    for i in range(sfBands.nBands):
        masker_intensity = 0.
        f = 0.
        for j in range(sfBands.lowerLine[i],sfBands.upperLine[i]+1):
            masker_intensity += dft_intensity[j]
            f += dft_intensity[j]*j    # intensity-weighted average frequency

        masker_intensity += 1**-12
        f = f*.5*sampleRate/Nlines / masker_intensity
        spl = SPL(masker_intensity)

        if spl > Thresh(f):    # Eliminate if below quiet threshold 
            maskers.append(Masker(f,spl,isTonal=False))


    fline = .5*sampleRate/Nlines * np.linspace(0.5, Nlines+0.5, Nlines)
    zline = Bark(fline)

    # Masking threshold with decimation
    masked_thresh = np.zeros(Nlines, dtype=np.float64)
    if len(maskers) != 0:
        max_masker = maskers[0].spl
        last_freq = maskers[0].f
        max_index = 0
        for i in range(1,len(maskers)):

            if abs(Bark(maskers[i].f) - Bark(last_freq)) < 0.5:
                # store max masker value
                if maskers[i].spl > max_masker:
                    max_index = i
                    max_masker = maskers[i].spl
            else:
                # add masker
                masked_thresh += maskers[max_index].vIntensityAtBark(zline)
                last_freq = maskers[i].f 
                max_masker = maskers[i].spl
                max_index = i
        
        if abs(Bark(maskers[-1].f) - Bark(last_freq)) >= 0.5:
            masked_thresh += maskers[-1].vIntensityAtBark(zline)


    # Global masked threshold
    masked_thresh += Intensity(Thresh(fline))
    masked_spl = SPL(masked_thresh)

    # Compute SMRs
    smr = np.empty(sfBands.nBands, dtype=np.float64)
    flag = sfBands.nBands / 2
    for i in range(sfBands.nBands):
        lower = sfBands.lowerLine[i]
        upper = sfBands.upperLine[i]+1
        if lower < upper:
            if i < flag:
                smr[i] = np.max(mdct_spl[lower:upper]-np.min(masked_spl[lower:upper]))
            else:
                smr[i] = np.max(mdct_spl[lower:upper]-np.mean(masked_spl[lower:upper]))
        else:
            smr[i] = 0.
    return smr

    #-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":
    pass

    """
    Fs = 48000.
    Ns = (512, 1024, 2048)
    components = ((0.47, 440.), (0.16, 550.), (0.12, 660.), (0.09, 880.), (0.04, 4400.), (0.03, 8800.))

    Fs, infile = wave.read("input.wav") #
    infile = infile * 1./2**16

    for i, N in enumerate(Ns):

        print("\nPeaks found using N = %d" % N)
        x = infile[:N]
        X = np.abs(np.fft.fft(HanningWindow(x)))[0:N/2]
        f = np.fft.fftfreq(N, 1./Fs)[0:N/2]

        Xintensity = 8./3. * 4./N**2 * np.abs(X)**2 
        Xspl = SPL(Xintensity)

        peaks = findPeaks(Xintensity, Fs)

        plt.figure(figsize=(14,6))
        plt.semilogx(f,Xspl)
        bLegend = True
        for index, val in enumerate(peaks):
            if val:
                if bLegend:
                    plt.plot(f[index], Xspl[index], 'b.', label="Estimated peaks")
                    bLegend = False
                else:
                    plt.plot(f[index], Xspl[index], 'b.')
                est_amp = np.sqrt(np.power(10.,(Xspl[index]-96)/10.))
                print("Peak found at %.3f (%.2f dB_SPL ==> amp = %.2f)" % (f[index], Xspl[index], est_amp))

    plt.title("N = %d" % (N))
    plt.grid(True) 
    plt.legend() 
    plt.xlabel('Freq. (Hz)') 
    plt.ylabel('SPL (dB)') 
    plt.xlim(100, 1e4) 
    plt.ylim(0, 96) 
    plt.savefig('findPeaks'+str(N)+'_2.png', bbox_inches='tight')

    # threshold in quiet comparison
    N = 1024
    n = np.arange(N, dtype=float)
    x = infile[:N]
    f = np.fft.fftfreq(N, 1./Fs)[0:N/2]
    dft_intensity = np.abs(np.fft.fft(HanningWindow(x)))[0:N/2]
    Xspl = 96. + 10.*np.log10( 8./3. * 4./N**2 * np.abs(dft_intensity)**2)

    threshold = Thresh(f)
    plt.figure(figsize=(14, 6)) 
    plt.semilogx(f, threshold) 
    plt.semilogx(f, Xspl) 
    plt.xlim(50, Fs/2) 
    plt.ylim(-20, 100) 
    plt.xlabel("Freq. (Hz)") 
    plt.ylabel("SPL (dB)") 
    plt.title("Test signal vs. Threshold in quiet") 
    plt.grid(True) 
    plt.savefig('Sig_vs_Thresh_W14.png', bbox_inches='tight')

    # Maskers
    plt.figure(figsize=(14, 6)) 
    plt.semilogx(f, threshold) 
    plt.semilogx(f, Xspl) 

    maskers = []
    sampleRate = Fs
    spl_peaks = findPeaks(dft_intensity, sampleRate)
    Nlines = N/2

    # scale factor bands
    nMDCTLines = N/2
    nLines = AssignMDCTLinesFromFreqLimits(nMDCTLines,Fs)
    mySFB = ScaleFactorBands(nLines)
    scaleplt = plt.vlines( (mySFB.lowerLine + 0.5)*Fs/(2.*nMDCTLines), -20, 150, linestyle='solid', colors='k', alpha=0.3 )
    textLocations = (mySFB.lowerLine + 0.5)*Fs/(2.*nMDCTLines)
    spacings = textLocations[1:] - textLocations[:-1]
    textLocations = textLocations[1:]
    mytext = 0
    for val in textLocations:
        mytext += 1
        scaletextplt = plt.text(val-0.5*spacings[mytext-1],-15,str(mytext))

    print(len(peaks))
    exit()

    for i, isPeak in enumerate(spl_peaks):
        if i+1 == Nlines: break
        if isPeak:
            intensity_sum = dft_intensity[i] + dft_intensity[i-1] + dft_intensity[i+1]
            f = .5*sampleRate/Nlines*(i*dft_intensity[i] + (i-1)*dft_intensity[i-1] + (i+1)*dft_intensity[i+1]) / intensity_sum
            spl = SPL(intensity_sum)
            if spl > Thresh(f):
                maskers.append(Masker(f,spl))
            dft_intensity[i] = dft_intensity[i-1] = dft_intensity[i+1] = 0    # eliminate for noise masker calculation
    
    for i in range(mySFB.nBands):
        spl = 0.
        freq = 0.
        for j in range(mySFB.lowerLine[i],mySFB.upperLine[i]+1):
            spl += dft_intensity[j]
            freq += dft_intensity[j]*j    # intensity-weighted average frequency

        freq = freq*.5*sampleRate/Nlines / spl
        spl = SPL(spl)

        if spl > Thresh(freq): 
            maskers.append(Masker(freq,spl,isTonal=False))
    
    f = np.fft.fftfreq(N, 1./Fs)[0:N/2]
    for m in maskers:
        print(m.spl)
        plt.semilogx(f, SPL(m.vIntensityAtBark(Bark(f)))) 

    plt.xlim(50, Fs/2) 
    plt.ylim(-20, 100) 
    plt.xlabel("Freq. (Hz)")
    plt.ylabel("SPL (dB)") 
    plt.title("Signal, Threshold in Quiet and Maskers") 
    plt.grid(True) 
    plt.savefig('maskers.png', bbox_inches='tight')
    
    exit()
    #############

    plt.figure(figsize=(14, 6))
    plt.semilogx(f, threshold , label='Threshold in quiet') 
    plt.semilogx(f, Xspl, label='FFT SPL') 
    for pair in components:
        masker = Masker(pair[1], SPL(pair[0]**2),True)
    plt.semilogx(f, SPL(masker.vIntensityAtBark(Bark(f))), alpha=0.5) 
    plt.xlim(50, Fs/2) 
    plt.ylim(-20, 100) 
    plt.xlabel("Freq. (Hz)")
    plt.ylabel("SPL (dB)") 
    plt.title("Signal, Threshold in Quiet and Maskers") 
    plt.grid(False)
    

    # Masking threshold
    maskThresh = np.zeros_like(f)
    intensity_squared_sum = np.zeros_like(maskThresh)
    for pair in components:
        masker = Masker(pair[1], SPL(pair[0]**2),True)
        intensity_squared_sum += masker.vIntensityAtBark(Bark(f))**2
    intensity_squared_sum += Intensity(Thresh(f))**2
    maskThresh = SPL( np.sqrt(intensity_squared_sum) )
    plt.plot(f,maskThresh,'r--',linewidth=2.0,label='Masked Threshold')

    plt.legend()
    plt.savefig('maskedThreshold.png', bbox_inches='tight')

    #############

    # SPL
    MDCTspl = 96. + 10.*np.log10( 8./3. * 4./N**2 * np.abs(X)**2)
    
    n = np.arange(N, dtype=float)
    x = np.zeros_like(n)
    for pair in components:
        x += pair[0]*np.cos(2*np.pi*pair[1]*n/Fs)
    MDCTdata = MDCT(SineWindow(x),N/2,N/2)
    
    CalcSMRs(x, MDCTdata, 1, 48000, mySFB)


    plt.figure(figsize=(14, 6)) 
    plt.semilogx(f, Xspl, label='FFT SPL')
    plt.semilogx(f, SPL( 8. * abs(MDCTdata)**2 ), label='MDCT SPL') 
    plt.plot(f, maskThresh ,'r',linewidth=1.0,  label='Masked Threshold') 
    scaleplt = plt.vlines((mySFB.lowerLine + 0.5)*Fs/(2.*nMDCTLines), -20, 150, linestyles='solid', colors='k', alpha=0.3) 
    plt.xlim(50, Fs/2)
    plt.ylim(-10, 100) 
    plt.xlabel("Freq. (Hz)") 
    plt.ylabel("SPL (dB)") 
    plt.legend()

    ax = plt.axes()
    ax.yaxis.grid(True) #horizontal lines
    mytext = 0
    for val in textLocations:
        mytext += 1
        scaletextplt = plt.text(val-0.5*spacings[mytext-1], -5, str(mytext))
    plt.savefig('spectraAndMaskingCurve.png', bbox_inches='tight')
    """

