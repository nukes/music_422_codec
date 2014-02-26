import unittest
import numpy as np

import codec.quantize as real_q
import provided.quantize as test_q


class TestQuantizerRoutines(unittest.TestCase):

    def setUp(self):
        self.sig = np.array([-1.0,-.98,-0.41,-0.02,0.0,0.01,0.35,0.78,0.99,1.0])

    def test_scale_factor(self):
        for x in self.sig:
            expected = test_q.ScaleFactor(x)
            computed = real_q.ScaleFactor(x)
            self.assertEqual(computed, expected)

    def test_vectorized_mantissa(self):
        scale_factor = real_q.ScaleFactor(self.sig[3])
        expected = test_q.vMantissa(self.sig, scale_factor)
        computed = real_q.vMantissa(self.sig, scale_factor)
        self.assertTrue(np.array_equal(expected, computed))

    def test_block_fp_dequantizer(self):
        scale_factor = real_q.ScaleFactor(self.sig[3])
        mantissas = real_q.vMantissa(self.sig, scale_factor)
        expected = test_q.vDequantize(scale_factor, mantissas)
        computed = real_q.vDequantize(scale_factor, mantissas)
        print expected
        print computed
        self.assertTrue(np.array_equal(expected, computed))


if __name__ == '__main__':
    unittest.main()
