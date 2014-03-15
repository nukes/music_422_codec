import numpy as np
from window import HanningWindow

import matplotlib.pyplot as plt #

def SPL(intensity): 
    """ Return SPL in dB for the given intensity vector """
    intensity = np.maximum(np.finfo(float).eps, intensity)
    spl = 96. + 10 * np.log10(intensity)
    return np.maximum(-30, spl)


def Intensity(spl): 
    """ Return intensity for the given SPL vector """
    spl = np.clip(spl, -30, 96)
    return np.power(10, (spl - 96) / 10.0)


def Thresh(f): 
    """ Returns the threshold in quiet measured in SPL at frequency f (in Hz).
    This is composed of three primary terms. """
    f = np.clip(f, 10, np.inf)
    khz = f / 1000.  # Convert frequencies to kHz to match units in the book
    a = 3.64 * (khz ** -0.8)
    b = -6.5 * np.exp(-0.6 * ((khz - 3.3) ** 2))
    c = 0.001 * (khz ** 4)
    return a + b + c


def Bark(f): 
    """ Returns the bark-scale frequency for input frequency f (in Hz) """
    khz = f / 1000.
    a = 13. * np.arctan(khz * 0.76)
    b = 3.5 * np.arctan((khz / 7.5) ** 2)
    return np.add(a, b)


class Masker:
    """ A masker whose masking curve drops linearly in Bark beyond 0.5 Bark from the 
    masker frequency. """
    
    def __init__(self,f,SPL,isTonal=True): 
        """ Find the masker's location in Bark, from the given frequency.
            Store the SPL of the masker frequency, used in the masking curve.
            Determine the type of masker, which determines the drop.
        """
        self.f = f
        self.z = Bark(f)
        self.spl = SPL
        self.isTonal = isTonal
        #self.drop = 6.025 + 0.275*self.z if isTonal else 2.025+0.175*self.z
        self.drop = 15.
        if not isTonal: self.drop = 5.5

    def IntensityAtFreq(self,freq): 
        """ The intensity of this masker at frequency freq """
        return self.IntensityAtBark(Bark(freq))

    def IntensityAtBark(self,z): 
        """ The intensity of this masker at Bark location z """
        dz = z - self.z
        spread = (0.37*max(self.spl-40., 0)) * float(dz>=0) - 27.
        spread *= (abs(dz)-0.5) * float(abs(dz) > 0.5)
        spl = self.spl + spread - self.drop
        return Intensity(spl)

    def vIntensityAtFreq(self,fVec):
        """ This was convenient, so I defined it. Vectorize I -> Bark. """
        return self.vIntensityAtBark(Bark(fVec))

    def vIntensityAtBark(self,zVec): 
        """ The intensity of this masker at vector of Bark locations zVec """
        dz = np.subtract(zVec, self.z)
        abs_dz = np.absolute(dz)

        slope_side = np.greater_equal(dz, 0)
        leveling = (0.37*max(self.spl-40., 0))

        spread = (slope_side * leveling) - 27. 
        spread *= (abs_dz - 0.5) * np.greater(abs_dz, 0.5)

        spls = self.spl + spread - self.drop

        return Intensity(spls)


# Default data for 25 scale factor bands based on the Z&F 25 critical bands
cbFreqLimits = np.array([100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270,
                         1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300,
                         6400, 7700, 9500, 12000, 15500, float('inf')])


def AssignMDCTLinesFromFreqLimits(nMDCTLines, sampleRate, flimit = cbFreqLimits): 
    """ Assigns MDCT lines to scale factor bands for given sample rate and 
    number of MDCT lines using predefined frequency band cutoffs passed as an 
    array in flimit (in units of Hz). If flimit isn't passed it uses the
    traditional 25 Zwicker & Fastl critical bands as scale factor bands.
    """

    mdct_lines = (np.arange(nMDCTLines) + 0.5) / nMDCTLines * (sampleRate / 2)

    assignment = []
    previous_size = 0
    for limit in flimit:
        if limit == float('inf'):
            size = nMDCTLines
        else:
            size = np.argmax(np.less(limit, mdct_lines))
        assignment.append(size - previous_size)
        previous_size = size
    return np.array(assignment)


def findPeaks(fftIntensity):
    """ Simple method to detect peaks in the frequency space """
    peaks = []
    dataSPL = SPL(fftIntensity)
    dataDiff = np.concatenate([[1], np.diff(dataSPL), [-1]])
    peakIndices = (dataDiff[0:-1] > 0) & (dataDiff[1:] < 0)
    return peakIndices


class ScaleFactorBands: 
    """ A set of scale factor bands (each of which will share a scale factor
    and a mantissa bit allocation) and associated MDCT line mappings.
    
    Instances know the number of bands nBands; the upper and lower limits for 
    each band lowerLimit[i in range(nBands)], upperLimit[i in range(nBands)]; 
    and the number of lines in each band nLines[i in range(nBands)] 
    """
    
    def __init__(self, nLines):
        """ Assigns MDCT lines to scale factor bands based on a vector of 
            frequency line allocation 
        """
        self.nBands = len(nLines)
        self.nLines = np.array(nLines)
        self.lowerLine = np.append(0, np.cumsum(nLines)[:-1])
        self.upperLine = nLines + self.lowerLine
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

    # Create tonal maskers by finding tonal peaks in the freq spectrum
    maskers = []
    spl_peaks = findPeaks(SPL(dft_intensity))
    for i, isPeak in enumerate(spl_peaks):
        if i+1 == Nlines: break
        if isPeak:
            if i:
                intensity_sum = dft_intensity[i] + dft_intensity[i-1] + dft_intensity[i+1]
                f = .5*sampleRate/Nlines*(i*dft_intensity[i] + (i-1)*dft_intensity[i-1] + (i+1)*dft_intensity[i+1]) / intensity_sum
            else:
                intensity_sum = dft_intensity[i] + dft_intensity[i+1]
                f = .5*sampleRate/Nlines*(i*dft_intensity[i] + (i+1)*dft_intensity[i+1]) / intensity_sum
                
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

        masker_intensity += 1**-12    # Prevent division by zero
        f = f*.5*sampleRate/Nlines / masker_intensity
        spl = SPL(masker_intensity)

        if spl > Thresh(f):    # Eliminate if below quiet threshold 
            maskers.append(Masker(f,spl,isTonal=False))


    fline = .5*sampleRate/Nlines * np.linspace(0.5, Nlines+0.5, Nlines)
    zline = Bark(fline)

    # Masking threshold with decimation: eliminate clusters of maskers by taking only strongest one
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

    return np.array(smr)
