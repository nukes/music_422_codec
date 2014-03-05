import unittest

from codec.onset import WindowState


class TestWindowStateMachine(unittest.TestCase):

    def setUp(self):
        self.machine = WindowState()

    def test_starts_at_long_state(self):
        self.assertEqual(self.machine.state, 0)

    def test_attack_forwards_through_transition(self):
        self.assertEqual(self.machine.state, 0)
        self.assertEqual(self.machine.transient(), 1)
        self.assertEqual(self.machine.transient(), 2)
        self.assertEqual(self.machine.transient(), 2)

    def test_window_state_can_reach_long_again_with_one_onset(self):
        self.assertEqual(self.machine.state, 0)
        self.assertEqual(self.machine.transient(), 1)
        self.assertEqual(self.machine.no_transient(), 2)
        self.assertEqual(self.machine.no_transient(), 3)
        self.assertEqual(self.machine.no_transient(), 4)

    def test_long_window_stays_long(self):
        self.assertEqual(self.machine.state, 0)
        self.assertEqual(self.machine.no_transient(), 0)



if __name__ == '__main__':
    unittest.main()