'''
Test functions for bit allocation. They do not exactly provide full
coverage, but it prevents an obvious base case. 
'''

import unittest
import numpy as np

from codec.mdct import MDCT
from codec.window import SineWindow
from codec.psychoac import AssignMDCTLinesFromFreqLimits
import codec.bitalloc as real
import tests.provided.bitalloc as test


class TestBitAllocation(unittest.TestCase):

    def setUp(self):
        ''' Some nice SMRs from the provided SMR solutionss '''
        self.smr = [7.25247249, 14.56170407, 7.10723327, 1.83829511, 16.03698066,
                    -0.33579172, 5.07776016, 15.8632976, 14.43906305,  17.14054399,
                    18.15409285, 16.08008443, 14.21182895, 12.12782107, 10.69488807,
                    9.79722076, 7.18667382, -5.69775959, 13.67881901, -5.89424813,
                    -10.49613988, 11.71039088, -22.9936503, -38.68104066, -80.88977674]

    def test_long_no_overalloc(self):
        budget = 1300
        lines = AssignMDCTLinesFromFreqLimits(512, 44100)
        computed = real.BitAlloc(budget, 16, 25, lines, np.array(self.smr))
        self.assertTrue(budget >= np.sum(np.multiply(computed[0], lines)))
        
    def test_compare_in_transition_window(self):
        budget = 655
        lines = AssignMDCTLinesFromFreqLimits(288, 44100)
        computed = real.BitAlloc(budget, 16, 25, lines, np.array(self.smr))
        computed_alloc = np.sum(np.multiply(computed[0], lines))
        self.assertTrue(budget >= np.sum(np.multiply(computed[0], lines)))

    def test_compare_in_short_window(self):
        budget = 5
        lines = AssignMDCTLinesFromFreqLimits(64, 44100)
        computed = real.BitAlloc(budget, 16, 25, lines, np.array(self.smr))
        computed_alloc = np.sum(np.multiply(computed[0], lines))
        self.assertTrue(budget >= computed_alloc)
        self.assertTrue(budget >= np.sum(np.multiply(computed[0], lines)))


if __name__ == '__main__':
    unittest.main()
