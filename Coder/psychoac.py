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
        

DBTOBITS = 6.02
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
    
    N = len(data)
    nLines = N/2
    lineToFreq = 0.5*sampleRate/nLines
    nBands = sfBands.nBands
    
    fftData = np.fft.fft(HanningWindow(data))[:nLines]
    fftIntensity = 32./3./N/N*np.abs(fftData)**2     # same as 4/N^2 for FFT Parserval's * 8/3 for Hanning Window
    fftSPL = SPL(fftIntensity)

    dtemp2 = DBTOBITS*MDCTscale

    mdctSPL = 4.*MDCTdata**2    # 8/N^2 for MDCT Parsevals * 2 for sine window, but 4/N^2 already in MDCT forward
    
    mdctSPL = SPL(mdctSPL) - dtemp2

    maskers = []

    for i in range(2,nLines-2):
        if fftIntensity[i] > fftIntensity[i-1] and fftIntensity[i] > fftIntensity[i+1] and fftSPL[i]-fftSPL[i-2] > 7 and fftSPL[i]-fftSPL[i+2] > 7 :
            
            spl = fftIntensity[i] + fftIntensity[i-1] + fftIntensity[i+1]

            f = lineToFreq*(i*fftIntensity[i] + (i-1)*fftIntensity[i-1] + (i+1)*fftIntensity[i+1]) / spl

            spl = SPL(spl)

            if spl > Thresh(f):
                maskers.append(Masker(f,spl))

            fftIntensity[i] = fftIntensity[i-1] = fftIntensity[i+1] = 0

    for i in range(nBands):
        spl = 0.
        f = 0.
        for j in range(sfBands.lowerLine[i],sfBands.upperLine[i]+1):
            spl += fftIntensity[j]
            f += fftIntensity[j]*j    # intensity-weighted average frequency

        f = f*lineToFreq/spl
        spl = SPL(spl)

        if spl > Thresh(f): maskers.append(Masker(f,spl,isTonal=False))

    fline = lineToFreq * np.linspace(0.5, nLines+0.5, nLines)
    zline = Bark(fline)

    maskedSPL = np.zeros(nLines, dtype=np.float64)

    for m in maskers: maskedSPL += m.vIntensityAtBark(zline)

    maskedSPL += Intensity(Thresh(fline))
    maskedSPL = SPL(maskedSPL)

    SMR = np.empty(nBands, dtype=np.float64)
    for i in range(nBands):
        lower = sfBands.lowerLine[i]
        upper = sfBands.upperLine[i]+1
        SMR[i] = np.max(mdctSPL[lower:upper]-maskedSPL[lower:upper])

    return SMR

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":
    
    """
    
    Fs = 48000.
    Ns = (512, 1024, 2048)
    components = ((0.47, 440.), (0.16, 550.), (0.12, 660.), (0.09, 880.), (0.04, 4400.), (0.03, 8800.))

    for i, N in enumerate(Ns):

        print("\nPeaks found using N = %d" % N)

        n = np.arange(N, dtype=float)
        x = np.zeros_like(n)
        for pair in components:
            x += pair[0]*np.cos(2*np.pi*pair[1]*n/Fs)
        X = np.abs(np.fft.fft(HanningWindow(x)))[0:N/2]
        f = np.fft.fftfreq(N, 1/Fs)[0:N/2]

        Xspl = 96. + 10.*np.log10( 8./3. * 4./N**2 * np.abs(X)**2 )
        peaks = findPeaks(Xspl, N/2, Fs)

        plt.figure(figsize=(14,6))
        plt.semilogx(f,Xspl)
        bLegend = True
        for pair in components:
            if bLegend:
                plt.plot(pair[1], 96 + 20.*np.log10(pair[0]), 'r.', label = "Expected peaks")
                bLegend = False
                plt.plot([pair[1], pair[1]], [0,96], 'k--', alpha=0.5)
            else:
                plt.plot(pair[1], 96 + 20.*np.log10(pair[0]), 'r.')
                plt.plot([pair[1], pair[1]], [0,96], 'k--', alpha=0.5)
        bLegend = True
        for peak in peaks:
            if bLegend:
                plt.plot(peak['f'], peak['spl'], 'b.', label="Estimated peaks")
                bLegend = False
            else:
                plt.plot(peak['f'], peak['spl'], 'b.')
            est_amp = np.sqrt(np.power(10.,(peak['spl']-96)/10.))
            print("Peak found at %.3f (%.2f dB_SPL ==> amp = %.2f)" % (peak['f'], peak['spl'], est_amp))

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
    x = np.zeros_like(n)
    for pair in components:
        x += pair[0]*np.cos(2*np.pi*pair[1]*n/Fs)
    X = np.abs(np.fft.fft(HanningWindow(x)))[0:N/2]
    f = np.fft.fftfreq(N, 1/Fs)[0:N/2]
    threshold = Thresh(f)

    # SPL
    Xspl = 96. + 10.*np.log10( 8./3. * 4./N**2 * np.abs(X)**2)

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
    for pair in components:
        masker = Masker(pair[1], SPL(pair[0]**2),True)
        plt.semilogx(f, SPL(masker.vIntensityAtBark(Bark(f)))) 
    plt.xlim(50, Fs/2) 
    plt.ylim(-20, 100) 
    plt.xlabel("Freq. (Hz)")
    plt.ylabel("SPL (dB)") 
    plt.title("Signal, Threshold in Quiet and Maskers") 
    plt.grid(True) 
    plt.savefig('maskers.png', bbox_inches='tight')
    
    #############

    plt.figure(figsize=(14, 6))
    plt.semilogx(f,	threshold ,	label='Threshold in quiet') 
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
    
    plt.figure(figsize=(14, 6)) 
    plt.semilogx(f, Xspl, label='FFT SPL')
    plt.semilogx(f, SPL( 8. * abs(MDCTdata)**2 ), label='MDCT SPL') 
    plt.plot(f,	maskThresh ,'r',linewidth=1.0,	label='Masked Threshold') 
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
