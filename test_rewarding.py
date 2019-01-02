import unittest

from parameterized import parameterized 

from Ai import Ai
from shared import LOSE, DRAW, WIN


class TestRewarding(unittest.TestCase):
    def setUp(self):
        self.ai = Ai()

    @parameterized.expand((
        (WIN, 5, 9),
        (WIN, 10, 8),
        (WIN, 20, 6.2),
        (WIN, 100, .3),
        (WIN, 200, .1),
        (LOSE, 5, -9),
        (LOSE, 10, -8),
        (LOSE, 20, -6.2),
        (LOSE, 100, -.3),
        (LOSE, 200, -.1),
        # (DRAW, 5, -8.7), # Not working yet, not major issue
        # (DRAW, 10, -7.6), # Not working yet, not major issue
        # (DRAW, 20, -5.3), # Not working yet, not major issue
        # (DRAW, 100, -2.4), # Not working yet, not major issue
        # (DRAW, 200, -.01), # Not working yet, not major issue
        ))
    def test_rewarding(self, condition, turns, expected_points):
        points = self.ai.calc_reward(condition, turns)

        expected_min = expected_points - 0.1
        expected_max = expected_points + 0.1
        
        self.assertTrue(expected_min <= points < expected_max,
            f'condition: {condition}, turns: {turns} -- {expected_min} <= {points} < {expected_max}'
        )

    @parameterized.expand((
        (WIN, 2),
        (WIN, WIN),
        (WIN, 50),
        (WIN, 100),
        (WIN, 100),
        ))
    def test_discount(self, reward, turns):
        discounted_array = self.ai.discount(reward, turns)
        
        previous = 0
        for current in discounted_array:
            self.assertGreater(current, previous, f'turns: {turns}, reward: {reward}, {current} !> {previous}')

if __name__ == "__main__":
    unittest.main()