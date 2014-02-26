''' Test functions for bit allocation '''

import unittest
import numpy as np

from codec.mdct import MDCT
from codec.window import SineWindow
from codec.psychoac import *
import codec.bitalloc as real
import provided.bitalloc as test


class TestBitAllocation(unittest.TestCase):

    def setUp(self):
        ''' Some nice SMRs from the provided SMR solutionss '''
        self.smr = [7.25247249, 14.56170407, 7.10723327, 1.83829511, 16.03698066,
                    -0.33579172, 5.07776016, 15.8632976, 14.43906305,  17.14054399,
                    18.15409285, 16.08008443, 14.21182895, 12.12782107, 10.69488807,
                    9.79722076, 7.18667382, -5.69775959, 13.67881901, -5.89424813,
                    -10.49613988, 11.71039088, -22.9936503, -38.68104066, -80.88977674]

    def test_compare_bit_allocation(self):
        lines = AssignMDCTLinesFromFreqLimits(512, 48000)
        expected = test.BitAlloc(1800, 16, 25, lines, np.array(self.smr))
        computed = real.BitAlloc(1800, 16, 25, lines, np.array(self.smr))
        print expected, np.sum(np.multiply(expected, lines))
        print list(computed), np.sum(np.multiply(computed, lines))
        print list(expected - computed)
        pass


if __name__ == '__main__':
    unittest.main()
