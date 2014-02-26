''' Test functions to help design the level converters '''

import unittest
import numpy as np

from codec.mdct import MDCT
from codec.window import SineWindow
import codec.psychoac as real
import provided.psychoac as test


class InvestigateSMRs(unittest.TestCase):

    def setUp(self):
        # Create a test oscillator signal
        self.Fs = 48000.
        freqs = [440, 550, 660, 880, 4400, 8800]
        amps = [0.47, 0.16, 0.12, 0.09, 0.04, 0.03]
        n = np.linspace(0, 1023, 1024)
        self.signal = np.zeros(n.size)
        for F, A in zip(freqs, amps):
            overtone = A * np.cos(2*np.pi*n*F/self.Fs)
            overtone = np.multiply(overtone, A)
            self.signal += overtone
        self.MDCTscale = 4
        self.MDCTdata = MDCT(SineWindow(self.signal), 512, 512)
        self.MDCTdata = self.MDCTdata * (2 ** self.MDCTscale)
        freq_lines = real.AssignMDCTLinesFromFreqLimits(512, 48000)
        self.sfbands = real.ScaleFactorBands(freq_lines)

    def test_compare_smrs(self):
        expected = test.CalcSMRs(self.signal, self.MDCTdata, self.MDCTscale, self.Fs, self.sfbands)
        computed = real.CalcSMRs(self.signal, self.MDCTdata, self.MDCTscale, self.Fs, self.sfbands)
        print expected
        for a, b in zip(expected, computed):
            print 'E: {:.5}, C: {:.5}, Diff: {:.5}'.format(a, b, abs(a-b))


class InvestigateMasker(unittest.TestCase):

    def test_noise_compare_masker(self):
        expected = test.Masker(f=880, SPL=60, isTonal=False)
        computed = real.Masker(f=880, SPL=60, isTonal=False)
        for i in range(25):
            e = expected.IntensityAtBark(i)
            c = computed.IntensityAtBark(i)
            self.assertTrue(np.allclose(e, c))

    def test_noise_compare_masker(self):
        expected = test.Masker(f=880, SPL=60, isTonal=True)
        computed = real.Masker(f=880, SPL=60, isTonal=True)
        for i in range(25):
            e = expected.IntensityAtBark(i)
            c = computed.IntensityAtBark(i)
            self.assertTrue(np.allclose(e, c))        



class TestScaleBands(unittest.TestCase):

    def test_confirm_not_insane(self):
        self.assertTrue(np.array_equal(real.cbFreqLimits, test.cbFreqLimits))

    def test_mdct_line_assignment(self):
        expected = test.AssignMDCTLinesFromFreqLimits(512, 48000, flimit=test.cbFreqLimits)
        computed = real.AssignMDCTLinesFromFreqLimits(512, 48000, flimit=real.cbFreqLimits)
        self.assertTrue(np.array_equal(expected, computed))

    def test_scale_factor_bands(self):
        nLines = real.AssignMDCTLinesFromFreqLimits(512, 48000, flimit=real.cbFreqLimits)
        expected = test.ScaleFactorBands(nLines)
        computed = real.ScaleFactorBands(nLines)
        self.assertEqual(expected.nBands, computed.nBands)
        self.assertEqual(type(expected.nLines), type(computed.nLines))
        self.assertTrue(np.array_equal(expected.nLines, computed.nLines))
        self.assertTrue(np.array_equal(expected.lowerLine, computed.lowerLine))
        self.assertTrue(np.array_equal(expected.upperLine, computed.upperLine))


class TestLevelConverters(unittest.TestCase):

    def setUp(self):
        self.intensities = np.linspace(0, 1, 100)
        self.spls = np.linspace(-45, 96, 200)
        self.thres_freq = np.linspace(1, 40000, 40000)
        self.crit_freq = np.array([0, 100, 200, 300, 400, 510, 630, 770, 920,
                                   1080, 1270, 1480, 1720, 2000, 2320, 2700,
                                   3150, 3700, 4400, 5300, 6400, 7700, 9500,
                                   12000, 15500])

    def test_intensity_to_spl(self):
        expected = test.SPL(self.intensities)
        computed = real.SPL(self.intensities)
        self.assertTrue(np.allclose(expected, computed))
  
    def test_spl_to_intensity(self):
        expected = test.Intensity(self.spls)
        computed = real.Intensity(self.spls)
        self.assertTrue(np.allclose(expected, computed))

    def test_threshold_in_quiet(self):
        expected = test.Thresh(self.thres_freq)
        computed = real.Thresh(self.thres_freq)
        self.assertTrue(np.allclose(expected, computed))

    def test_bark_scale_conversion(self):
        expected = test.Bark(self.crit_freq)
        computed = real.Bark(self.crit_freq)
        self.assertTrue(np.allclose(expected, computed))



if __name__ == '__main__':
    unittest.main()
