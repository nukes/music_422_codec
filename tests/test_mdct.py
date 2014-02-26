import unittest
import collections

import numpy as np

import codec.mdct as real
import provided.mdct as test


class TestMDCT(unittest.TestCase):

    def setUp(self):
        self.signal = [4, 4, 4, 4, 3, 1, -1, -3, 0, 1, 2, 3]
        self.X = [ [0, 0, 0, 0, 4, 4, 4, 4],
                   [4, 4, 4, 4, 3, 1, -1, -3],
                   [3, 1, -1, -3, 0, 1, 2, 3],
                   [0, 1, 2, 3, 0, 0, 0, 0] ]

    def test_forward_MDCT(self):
        for x in self.X:
            transform = real.MDCT(x, 4, 4)
            expected = test.MDCT(x, 4, 4)
            self.assertTrue(np.allclose(transform, expected))
    
    def test_inverse_MDCT(self):
        for x in self.X:
            transform = real.MDCT(x, 4, 4)
            inverse = real.IMDCT(transform, 4, 4)
            expected = test.IMDCT(transform, 4, 4)
            self.assertTrue(np.allclose(inverse, expected))


if __name__ == '__main__':
    unittest.main()
