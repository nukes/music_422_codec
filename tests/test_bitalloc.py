''' Test functions for bit allocation '''

import unittest
import numpy as np

from codec.mdct import MDCT
from codec.window import SineWindow
from codec.psychoac import *
import codec.bitalloc as real
import provided.bitalloc as test


class TestBitTransfer(unittest.TestCase):

    def test_pairing_ones(self):
        previous = np.array([1, 0, 0, 0, 1, 1])
        expected = np.array([2, 0, 0, 0, 1, 0])
        computed = real._pair_ones(previous)
        print expected
        print computed
        self.assertTrue(np.array_equal(expected, computed))


"""
class TestBitAllocation(unittest.TestCase):

    def setUp(self):
        ''' Some nice SMRs from the provided SMR solutionss '''
        self.smr = [7.25247249, 14.56170407, 7.10723327, 1.83829511, 16.03698066,
                    -0.33579172, 5.07776016, 15.8632976, 14.43906305,  17.14054399,
                    18.15409285, 16.08008443, 14.21182895, 12.12782107, 10.69488807,
                    9.79722076, 7.18667382, -5.69775959, 13.67881901, -5.89424813,
                    -10.49613988, 11.71039088, -22.9936503, -38.68104066, -80.88977674]

    def test_compare_bit_allocation(self):
        budget = 1300
        lines = AssignMDCTLinesFromFreqLimits(512, 44100)
        print test.BitAlloc(budget, 16, 25, lines, np.array(self.smr))
        expected = test.BitAlloc(budget, 16, 25, lines, np.array(self.smr))
        computed = real.BitAlloc(budget, 16, 25, lines, np.array(self.smr))
        expected_alloc = np.sum(np.multiply(computed, lines))
        computed_alloc = np.sum(np.multiply(expected, lines))
        self.assertTrue(budget >= expected_alloc)
        self.assertTrue(budget >= computed_alloc)

    def test_compare_in_transition_window(self):
        budget = 655
        lines = AssignMDCTLinesFromFreqLimits(288, 44100)
        (expected, x) = test.BitAlloc(budget, 16, 25, lines, np.array(self.smr))
        (computed, x) = real.BitAlloc(budget, 16, 25, lines, np.array(self.smr))
        expected_alloc = np.sum(np.multiply(computed, lines))
        computed_alloc = np.sum(np.multiply(expected, lines))
        self.assertTrue(budget >= expected_alloc)
        self.assertTrue(budget >= computed_alloc)

    def test_compare_in_short_window(self):
        budget = 5
        lines = AssignMDCTLinesFromFreqLimits(64, 44100)
        (expected, x) = test.BitAlloc(budget, 16, 25, lines, np.array(self.smr))
        (computed, x) = real.BitAlloc(budget, 16, 25, lines, np.array(self.smr))
        expected_alloc = np.sum(np.multiply(expected, lines))
        computed_alloc = np.sum(np.multiply(computed, lines))
        self.assertTrue(budget >= computed_alloc)
"""

if __name__ == '__main__':
    unittest.main()
