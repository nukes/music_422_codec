import numpy as np
import matplotlib.pyplot as plt
from mdct import *
from window import *

FULLSCALESPL = 96.
def SPL(intensity): 
    """
    Returns the SPL corresponding to intensity (in units where 1 implies 96dB)
    """     
    return np.maximum(-30,FULLSCALESPL + 10.*np.log10(intensity))
    
def Intensity(spl): 
    """
    Returns the intensity (in units of the reference intensity level) for SPL spl
    """ 
    return np.power(10.,(spl-96)/10.)


def findPeaks(fftSPL, N, Fs, thresh = 7.0):
    peaks = []
    fftIntensity = np.power(10.,(fftSPL-96)/10.)

    for i in range(1,N-1):
        if fftSPL[i] > fftSPL[i-1] and fftSPL[i] > fftSPL[i+1]:

          intensity = fftIntensity[i] + fftIntensity[i-1] + fftIntensity[i+1]
          
          f = (0.5*Fs/N)*(i*fftIntensity[i] + (i-1)*fftIntensity[i-1] + (i+1)*fftIntensity[i+1])/intensity
          spl = np.maximum(-30, 96 + 10.*np.log10(intensity))

          peaks.append(dict(f=f, spl=spl))

    return peaks

    
def Thresh(f): 
    """Returns the threshold in quiet measured in SPL at frequency f (in Hz)""" 

    f = np.maximum(f,10.)
    return 3.64*np.power(f/1000.,-0.8) - 6.5*np.exp(-0.6*(f/1000.-3.3)**2) + 0.001*np.power(f/1000.,4)
    
def Bark(f): 
    """Returns the bark-scale frequency for input frequency f (in Hz) """     
    
    return 13.0*np.arctan(0.76*f/1000.)+3.5*np.arctan((f/7500.)*(f/7500.))

MASKTONALDROP = 15.
MASKNOISEDROP = 5.5
class Masker: 
    """ 
    a Masker whose masking curve drops linearly in Bark beyond 0.5 Bark from the 
    masker frequency
    """
    
    def __init__(self,f,SPL,isTonal=True): 
        """
        initialized with the frequency and SPL of a masker and whether or not 
        it is Tonal
        """
        self.f = f
        self.SPL = SPL
        self.z = Bark(f)
        self.drop = MASKTONALDROP
        if not isTonal: self.drop = MASKNOISEDROP

    def IntensityAtFreq(self,f): 
        """The intensity of this masker at frequency freq"""
        return self.IntensityAtBark(Bark(freq))

    def IntensityAtBark(self,z): 
        """The intensity of this masker at Bark location z""" 
        
        maskedDB = self.SPL - self.drop
        
        if abs(self.z - z) > 0.5:
            if self.z > z:
                maskedDB -= 27.*(self.z-z-0.5)
            else:
                iEffect = self.SPL - 40.
                if np.abs(iEffect) != iEffect:
                    iEffect = 0.
                maskedDB -= (27. -0.37*iEffect)*(z-self.z-0.5)
        
        return Intensity(maskedDB)

    def vIntensityAtBark(self,zVec): 
        """The intensity of this masker at vector of Bark locations zVec""" 
        maskedDB = np.empty(np.size(zVec), np.float)
        maskedDB.fill(self.SPL - self.drop)

        v = ((self.z-0.5) > zVec)
        maskedDB[v] -= 27.*(self.z-zVec[v]-0.5)
        iEffect = self.SPL-40.

        if iEffect < 0.:
            iEffect = 0.
        v = ((self.z+0.5) < zVec)
        maskedDB[v] -= (27.-0.37*iEffect)*(zVec[v]-(self.z-0.5))
        return Intensity(maskedDB)
    

# Default data for 25 scale factor bands based on the traditional 25 critical bands
cbFreqLimits = np.array([100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, float('inf')])
                 
def AssignMDCTLinesFromFreqLimits(nMDCTLines, sampleRate, flimit = cbFreqLimits): 
    """
    Assigns MDCT lines to scale factor bands for given sample rate and number 
    of MDCT lines using predefined frequency band cutoffs passed as an array 
    in flimit (in units of Hz). If flimit isn't passed it uses the traditional 
    25 Zwicker & Fastl critical bands as scale factor bands.
    """
    maxFreq = sampleRate*1.0/2    # Hz
    width = maxFreq/(nMDCTLines)

    linesInBands = np.zeros(len(flimit[ flimit < maxFreq ])+1)
    linesInBands = linesInBands.astype(int)

    pointer = 1.0*width/2
    for n in range(0,nMDCTLines):
        for i in range(0,len(flimit)):
            if pointer <= flimit[i]:
                linesInBands[i] = linesInBands[i] + 1
                break
        if pointer > flimit[len(flimit)-1]:
            linesInBands[len(flimit)-1] = linesInBands[len(flimit)-1] + 1
        pointer += 1.0*width

    return linesInBands
    
class ScaleFactorBands: 
    """
    A set of scale factor bands (each of which will share a scale factor and a 
    mantissa bit allocation) and associated MDCT line mappings.
    
    Instances know the number of bands nBands; the upper and lower limits for 
    each band lowerLimit[i in range(nBands)], upperLimit[i in range(nBands)]; 
    and the number of lines in each band nLines[i in range(nBands)] 
    """
    
    def __init__(self,nLines): 
        """
        Assigns MDCT lines to scale factor bands based on a vector of the number 
        of lines in each band
        """ 
        self.nBands = len(nLines)
        self.nLines = nLines
        self.lowerLine = np.zeros(len(nLines),dtype=np.int32)
        
        for i in range(1,len(nLines)):
            self.lowerLine[i] = self.lowerLine[i-1] + nLines[i-1]
            
        self.upperLine = self.lowerLine + nLines - 1
        

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
    
    dataFFT = np.fft.fft(HanningWindow(data))
    N = len(data)
    f = np.fft.fftfreq(N,1.0/sampleRate)
    f = f[0:N/2]
    dataFFT = dataFFT[0:N/2]
    
    dataIntensity = 4./(N**2 * 3./8) * np.abs(dataFFT)**2
    dataSPL = SPL(dataIntensity)     

    mdctSPL = SPL(4.0 * MDCTdata**2) - 6.02*MDCTscale

    maskers = []

    pressureLevels = [] # Find sound pressure levels
    dataDiff = np.concatenate([[1],np.diff(dataSPL)])
    dataDiff = np.concatenate([dataDiff,[-1]])
    peakIndices = (dataDiff[0:-1] > 0) & (dataDiff[1:] < 0)
    for i, val in enumerate(peakIndices):
        if val == True:
            newSpl = 96 + 10*np.log10( 4.0/(N**2 * 3.0/8) * np.abs( np.sum(dataFFT[i-2:i+2] * np.ma.conjugate(dataFFT[i-2:i+2]))) )
            pressureLevels.append(newSpl)
    
    i = 0
    for centerFreq in f[peakIndices]:
        maskers.append(Masker(centerFreq,pressureLevels[i]))
        i += 1

    for i in range(sfBands.nBands):
        intensitySum = 0.
        f = 0.
        for j in range(sfBands.lowerLine[i],sfBands.upperLine[i]+1):
            intensitySum += dataIntensity[j]
            f += dataIntensity[j]*j    # intensity-weighted average frequency

        f = f*1.0*sampleRate/N/intensitySum
        spl = SPL(intensitySum)

        if spl > Thresh(f): 
            maskers.append(Masker(f,spl,isTonal=False))
    
    f = 1.0*sampleRate/N * np.linspace(0.5, N/2+0.5, N/2)   # new frequency vector for mask threshold

    threshold = np.zeros(N/2, dtype=np.float64)

    for m in maskers: 
        threshold += m.vIntensityAtBark(Bark(f))

    threshold += Intensity(Thresh(f))   # sum mask intensities with hearing threshold
    threshold = SPL(threshold)  # convert to dB

    SMR = np.empty(sfBands.nBands, dtype=np.float64)
    for i in range(sfBands.nBands):
        SMR[i] = np.max(mdctSPL[sfBands.lowerLine[i]:sfBands.upperLine[i]+1]-threshold[sfBands.lowerLine[i]:sfBands.upperLine[i]+1])
    
    return SMR

#-----------------------------------------------------------------------------
