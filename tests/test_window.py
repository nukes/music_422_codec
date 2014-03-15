import unittest
import numpy as np

import codec.window as real_w
import tests.provided.window as test_w


class WindowTests(unittest.TestCase):

    def setUp(self):
        self.signal = np.linspace(-1, 1, 8)

    def test_sine_window(self):
        computed = real_w.SineWindow(self.signal)
        expected = test_w.SineWindow(self.signal)
        self.assertTrue(np.allclose(computed, expected))

    def test_hanning_window(self):
        computed = real_w.HanningWindow(self.signal)
        expected = test_w.HanningWindow(self.signal)
        self.assertTrue(np.allclose(computed, expected))

    def test_kaiser_bessel_derived_window(self):
        pass

if __name__ == '__main__':
    unittest.main()
