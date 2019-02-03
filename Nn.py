import tensorflow as tf
import numpy as np
import os
import string
import random
import time

from shared import LOSE, WIN, DRAW

class Nn():
    def __init__(self, keep_prob = 1, greedy=True, epsilon=0, verbose=False, learnign_rate=0.00001, batch_size=1, epochs=1):
        assert(greedy in (True, False))
        assert(verbose in (True, False))
        assert(0 <= keep_prob <= 1)
        assert(0 <= epsilon <= 1)

        # Discount factor
        self._gamma = 0.99
        self._learning_rate = learnign_rate
        self._keep_prob = keep_prob
        self._greedy = greedy
        self._epochs = epochs
        self._epsilon = epsilon
        self.batch_size = batch_size

        self._episodes = []
        self._episode = { "observations": [], "actions": [], "reward": 0 }

        self._model_path = "./model/"
        self._model_name = "model"
        self._verbose = verbose
        if verbose:
            np.set_printoptions(precision=3, suppress=True, threshold=np.nan, linewidth=np.nan)
        self._board_size = 25
        # self.__tensorboard()
        self._session = tf.Session()
        self._saver = None
        self.__load_model()

    def __str__(self):
        return 'NN'

    def new_game(self):
        self._episodes.append(self._episode)
        self._episode = { "observations": [], "actions": [], "reward": 0 }

    def train(self):
        # Load might have been changed by previous Ai
        self.__load_model()
        for _ in range(0, self._epochs):
            for batch in self._episodes:
                rewards = self.discount(batch["reward"], len(batch["actions"]))
                self._session.run([self._loss, self._training_optimizer],
                    {
                        self._input: np.vstack(batch["observations"]),
                        self._output: np.stack(batch["actions"]),
                        self._rewards: rewards,
                        self._dropout: self._keep_prob
                    }
                )

        self.__save_model()
        self._episodes = []

    def predict(self, data):
        input_board, output_board = self.__preprocess(data)
        prediction = self._session.run(self._sigmoid, { self._input: input_board })
        action_index = self.__action_from_prediction(output_board, prediction)
        index = self.__play_action(output_board, action_index)
        coordinates = self.__index_to_coordinates(index)

        action = np.zeros(self._board_size**2, dtype=np.uint8)
        action[action_index] = 1
        self._episode["actions"].append(action)
        self._episode["observations"].append(input_board)

        if (self._verbose):
            print('-'*80)
            print('-'*80)
            print('Board')
            print(np.reshape(output_board, (25,25)))
            print('Prediction')
            print(np.reshape(prediction, (25,25)))

        return coordinates

    def reward(self, reward):
        self._episode["reward"] = reward

    # TODO: This should be in completely another class
    def calc_reward(self, reward, turns):
        assert reward in (LOSE, DRAW, WIN)
        # use -tanh(x/100)+3 for scaled reward and covert to reward sign
        return np.sign(reward)*(-2*np.tanh(np.abs(reward*turns)/100)+3)

    def discount(self, reward, turns):
        rewards = np.zeros(turns)
        # Give more carrot or stick based on how fast won or lost.
        scaled_reward = self.calc_reward(reward, turns)
        rewards[0] = scaled_reward
        discounted_rewards = np.zeros(turns)
        cumulative = 0.0
        for index, reward in enumerate(rewards):
            cumulative = cumulative * self._gamma + reward
            discounted_rewards[index] = cumulative

        discounted_rewards = np.flip(discounted_rewards)

        return discounted_rewards

    def __index_to_coordinates(self, index):
        size = self._board_size
        c = {
            "x": index % size,
            "y": int(index / size)
        }
        # Make sure that index is transformed correctly
        assert c["y"] > -1 and c['y'] < size
        assert c["x"] > -1 and c['x'] < size
        assert c["y"]*size+c["x"] == index

        return c

    def __action_from_prediction(self, board, prediction):
        assert(board.shape == (1,625))
        assert(prediction.shape == (1,625))
        prediction[board != 0] = 0
        action = 0
        if (self._greedy):
            # Take first best
            # action = np.where(prediction == np.max(prediction))[1][0]
            # Take one of the best
            indexes = np.where(prediction == np.max(prediction))[1]
            action = np.random.choice(indexes)
        elif (self._epsilon <= random.random()):
            prediction_sum = prediction[0].sum()
            normalized = prediction[0]/prediction_sum
            # Boltzman approach
            action = np.random.choice(list(range(normalized.size)), p=normalized)
        else: # e-greedy approach
            action = np.random.choice(np.where(board == 0)[1])

        return action

    def __play_action(self, board, action):
        assert(board.shape == (1, 625))
        assert(int(action) == action)
        # If there is no mark at the predicted location, it is the best valid prediction
        if board[0][action] == 0:
            return action
        else:
            try:
                # If invalid spot, play random.
                return np.random.choice(np.where(board == 0)[1])
            except IndexError:
                np.savetxt("board.log", np.reshape(board, (self._board_size, self._board_size)), "%d")
                raise IndexError("Every slot has been played!")

    def __tensorboard(self):
        writer = tf.summary.FileWriter('.')
        writer.add_graph(tf.get_default_graph())
        writer.flush()

    def __preprocess(self, data):
        # data['board'] is a reference to gomoku board, we don't want to change
        # state of the game, so we take a copy.
        board = np.copy(data["board"])
        attack_boards = np.copy(data["attack_boards"])
        defense_boards = np.copy(data["defense_boards"])

        attack_boards[attack_boards < 0] = 0
        defense_boards[defense_boards < 0] = 0

        if data["player"] == 2:
            # AI plays as player 1. Inverse board
            board[board == 1] = -1
            board[board == 2] = 1
        elif data["player"] == 1:
            board[board == 2] = -1

        # Input board reshaped to 4d for tensorflow convolution
        input_board = np.vstack(([board[np.newaxis, ...], attack_boards, defense_boards]))
        # conv2d expects shape [batch_size, first_dim, second_dim, num_channels]
        # [num_channels, first_dim, second_dim] => [first_dim, second_dim, num_channels]
        input_board = np.moveaxis(input_board, 0, -1)
        input_board = input_board[np.newaxis, ...]
        output_board = np.reshape(board, (1, self._board_size**2))

        return input_board, output_board

    def __architechture(self):
        self._input = tf.placeholder(shape=(None, self._board_size, self._board_size, 9), dtype=tf.float32)
        self._output = tf.placeholder(shape=(None, self._board_size**2), dtype=tf.float32)
        self._rewards = tf.placeholder(shape=(None,), dtype=tf.float32)
        self._dropout = tf.placeholder(dtype=tf.float32)

        layer_1 = tf.layers.conv2d(self._input, filters=32, kernel_size=10, activation=tf.nn.relu, padding='valid')
        layer_1 = tf.layers.dropout(layer_1, self._dropout)

        layer_2 = tf.layers.conv2d(layer_1, filters=32, kernel_size=5, activation=tf.nn.relu, padding='valid')
        layer_2 = tf.layers.dropout(layer_2, self._dropout)

        layer_3 = tf.layers.conv2d(layer_2, filters=32, kernel_size=3, activation=tf.nn.relu, padding='valid')
        layer_3 = tf.layers.dropout(layer_3, self._dropout)

        layer_4 = tf.layers.flatten(layer_3)
        layer_4 = tf.layers.dense(layer_4, 1024, tf.nn.relu)
        layer_4 = tf.layers.dropout(layer_4, self._dropout)

        self._layer_out = tf.layers.dense(layer_4, self._board_size**2, activation=None)
        self._sigmoid = tf.nn.sigmoid(self._layer_out)

        x_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._layer_out, labels=self._output)
        self._loss = tf.reduce_mean(self._rewards * x_entropy)
        self._training_optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss)

        init = tf.global_variables_initializer()
        self._session.run(init)

    def __load_model(self):
        if (os.path.isfile(self._model_path + self._model_name + '.meta')):
            if self._saver is None:
                self._saver = tf.train.import_meta_graph(self._model_path + self._model_name + '.meta')
            self._saver.restore(self._session, tf.train.latest_checkpoint(self._model_path))

            # save variables
            self._input = tf.get_collection("input")[0]
            self._output = tf.get_collection("output")[0]
            self._rewards = tf.get_collection("rewards")[0]
            self._layer_out = tf.get_collection("output_layer")[0]
            self._sigmoid = tf.get_collection("scaled_output")[0]
            self._loss = tf.get_collection("loss")[0]
            self._training_optimizer = tf.get_collection('optimizer')[0]
            self._dropout =tf.get_collection('dropout')[0]

        else:
            self.__architechture()
            # Save meta if new architecture created
            self.__save_model(save_meta=True)

    def __save_model(self, save_meta=False):
        if self._saver is None:
            self._saver = tf.train.Saver()
        tf.add_to_collection("input", self._input)
        tf.add_to_collection("output", self._output)
        tf.add_to_collection("rewards", self._rewards)
        tf.add_to_collection("output_layer", self._layer_out)
        tf.add_to_collection("scaled_output", self._sigmoid)
        tf.add_to_collection("loss", self._loss)
        tf.add_to_collection('optimizer', self._training_optimizer)
        tf.add_to_collection('dropout', self._dropout)

        self._saver.save(self._session, self._model_path + self._model_name, write_meta_graph=save_meta)

if __name__ == "__main__":
    ai = Nn()
    ai2 = Nn()
    ai.predict({ "board": np.zeros((25, 25)), "player": 1 })
