import numpy as np

from codec.window import KBDWindow, HanningWindow


def SPL(intensity): 
    ''' Return SPL in dB for the given intensity vector '''
    intensity = np.maximum(np.finfo(float).eps, intensity)
    spl = 96 + 10 * np.log10(intensity)
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
    dft = np.fft.fft(HanningWindow(data))[:N/2]
    dft_intensity = (8./3. * 4./(N**2)) * np.abs(dft)**2
    dft_spl = SPL(dft_intensity)

    # Transform the MDCT data into the SPL
    # Scale down the MDCT's SPL by 6dB per bit
    mdct_scale_unwind = 6.02 * MDCTscale
    mdct_spl = SPL(4. * MDCTdata**2) - mdct_scale_unwind

    ''' Masker Logic '''
    # Identify tonal and noise maskers
    # Do this by breaking the spectrum into blocks and estimating peaks and SPLs
    # The smaller peaks in each block should not contribute _much_ to the threshold
    # So actually, a better way to do this is to find local maxima
    # But! I'm just going to use the critical frequencies
    freq_lines = np.linspace(0.5, Nlines+0.5, Nlines) * 0.5 * sampleRate / Nlines
    bark_lines = Bark(freq_lines)

    combined_mask = np.zeros(N/2)

    # Try using just the midpoints of the critical frequencies
    for i in range(sfBands.nBands):
        lower = sfBands.lowerLine[i]
        upper = sfBands.upperLine[i] + 1
        crit_int = dft_intensity[lower:upper]

        # Compute the average SPL
        avg_spl = np.sum(crit_int)
        if avg_spl == 0.:
            avg_spl = -30.
        else:
            avg_spl = 96. + 10. * np.log10(avg_spl)

        # Take the mask frequency to be the center frequency
        avg_freq = cbFreqCenters[i]

        # Combine the masker into the total masking threshold
        if avg_spl > Thresh(avg_freq):
            masker = Masker(avg_freq, avg_spl, isTonal=False)
            combined_mask += masker.vIntensityAtBark(bark_lines)

    ''' End Masker Logic '''

    # Combine these masking curves with hearing threshold
    combined_mask += Intensity(Thresh(freq_lines))
    combined_mask = SPL(combined_mask)

    # Now, compute the SMR for each frequency
    smr = np.subtract(mdct_spl, np.real(combined_mask))

    # Take the maximum SMR inside each critical band
    critical_smr = []
    for i in range(sfBands.nBands):
        band_indices = np.linspace(0, sfBands.nLines[i]-1, sfBands.nLines[i])
        band_indices += sfBands.lowerLine[i]
        smrs_in_band = np.take(smr, band_indices.astype(int))
        max_smr = np.amax(smrs_in_band)
        critical_smr.append(max_smr)

    return np.array(critical_smr)