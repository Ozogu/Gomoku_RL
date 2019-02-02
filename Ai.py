import numpy as np

class Ai():
    def __init__(self, verbose=False):
        assert(verbose in (True, False))
        self.__verbose = verbose
        self.batch_size = 200

    def __str__(self):
        return 'AI'

    def new_game(self):
        pass

    def reward(self, reward):
        pass

    def predict(self, data):
        input_board = self.__preprocess(data)
        coordinates = self.__action_from_board(input_board)

        return coordinates

    def __preprocess(self, data):
        input_board = np.vstack((data['attack_boards'], data['defense_boards']))

        return input_board

    def __action_from_board(self, board):
        # Find highest streak on defense or attack map
        max_indices = np.where(board == np.max(board))
        # Pick randomly from the best
        choice = np.random.randint(max_indices[0].size)

        return { 'x': max_indices[2][choice], 'y': max_indices[1][choice] }
