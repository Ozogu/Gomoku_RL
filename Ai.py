import tensorflow as tf
import numpy as np
import os
import string
import random
import time

from shared import LOSE, WIN, DRAW

class Ai():
    def __init__(self, keep_prob = 1, greedy=True, epsilon=0, verbose=False):
        assert(greedy in (True, False))
        assert(verbose in (True, False))
        assert(0 <= keep_prob <= 1)
        assert(0 <= epsilon <= 1)

        # Discount factor
        self.__gamma = 0.99
        self.__learning_rate = 0.00001
        self.__keep_prob = keep_prob
        self.__greedy = greedy
        self.__epochs = 1
        self.__epsilon = epsilon
        self.batch_size = 128 # Public

        self.__episodes = []
        self.__episode = { "observations": [], "actions": [], "reward": 0 }

        self.__model_path = "./model/"
        self.__model_name = "model"
        self.__verbose = verbose
        if verbose:
            np.set_printoptions(precision=3, suppress=True, threshold=np.nan, linewidth=np.nan)
        self.__board_size = 25
        # self.__tensorboard()
        self.__session = tf.Session()
        self.__saver = None
        self.__load_model()

    def new_game(self):
        self.__episodes.append(self.__episode)
        self.__episode = { "observations": [], "actions": [], "reward": 0 }

    def train(self):
        # Load might have been changed by previous Ai
        self.__load_model()
        for _ in range(0, self.__epochs):
            for batch in self.__episodes:
                rewards = self.discount(batch["reward"], len(batch["actions"]))
                self.__session.run([self.__loss, self.__training_optimizer],
                    {
                        self.__input: np.vstack(batch["observations"]),
                        self.__output: np.stack(batch["actions"]),
                        self.__rewards: rewards,
                        self.__dropout: self.__keep_prob
                    }
                )
        
        self.__save_model()
        self.__episodes = []

    def predict(self, data):
        input_board, output_board = self.__preprocess(data)
        prediction = self.__session.run(self.__softmax, { self.__input: input_board })
        best_valid_index = self.__best_valid_prediction(prediction, output_board)
        coordinates = self.__index_to_coordinates(best_valid_index)        

        self.__episode["observations"].append(input_board)
        action = np.zeros(self.__board_size**2)
        action[best_valid_index] = 1
        self.__episode["actions"].append(action)

        if (self.__verbose):
            print('-'*80)
            print('-'*80)
            print('Board')
            print(np.reshape(output_board, (25,25)))
            print('Prediction')
            print(np.reshape(prediction, (25,25)))

        return coordinates

    def reward(self, reward):
        self.__episode["reward"] = reward

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
            cumulative = cumulative * self.__gamma + reward
            discounted_rewards[index] = cumulative
			
        discounted_rewards = np.flip(discounted_rewards)

        return discounted_rewards
    
    def __index_to_coordinates(self, index):
        size = self.__board_size
        c = {
            "x": index % size,
            "y": int(index / size)
        }
        # Make sure that index is transformed correctly
        assert c["y"] > -1 and c['y'] < size
        assert c["x"] > -1 and c['x'] < size
        assert c["y"]*size+c["x"] == index

        return c

    def __best_valid_prediction(self, prediction, board):
        # Remove invalid actions
        # prediction[board != 0] = 0

        index = 0
        if (self.__greedy):
            # Take first best
            # index = np.where(prediction == np.max(prediction))[1][0]
            # Take one of the best
            indexes = np.where(prediction == np.max(prediction))[1]
            index = np.random.choice(indexes)
        else: # Boltzman approach
            if (self.__epsilon <= random.random()):
                # normalized = prediction[0]/prediction[0].sum()
                index = np.random.choice(list(range(prediction.size)), p=prediction[0])
            else: # e-greedy approach
                index = np.random.choice(np.where(board == 0)[1])

        # If there is no mark at the predicted location, it is the best valid prediction
        if board[0][index] == 0:
            return index
        else:
            try:
                # If invalid spot, play random.
                return np.random.choice(np.where(board == 0)[1])
            except IndexError:
                np.savetxt("board.log", np.reshape(board, (self.__board_size, self.__board_size)), "%d")
                raise IndexError("Every slot has been played!")

    def __tensorboard(self):
        writer = tf.summary.FileWriter('.')
        writer.add_graph(tf.get_default_graph())
        writer.flush()

    def __preprocess(self, data):
        # data['board'] is a reference to gomoku board, we don't want to change
        # state of the game, so we take a copy.
        board = np.copy(data["board"])
        if data["player"] == 2:
            # AI plays as player 1. Inverse board
            board[board == 1] = -1
            board[board == 2] = 1
        elif data["player"] == 1:
            board[board == 2] = -1

        # Input board reshaped to 4d for tensorflow convolution
        input_board = board[np.newaxis, ..., np.newaxis]
        output_board = np.reshape(board, (1, self.__board_size**2))

        return input_board, output_board
        

    def __architechture(self):
        self.__input = tf.placeholder(shape=(None, self.__board_size, self.__board_size, 1), dtype=tf.float32)
        self.__output = tf.placeholder(shape=(None, self.__board_size**2), dtype=tf.int32)
        self.__rewards = tf.placeholder(shape=(None,), dtype=tf.float32)
        self.__dropout = tf.placeholder(dtype=tf.float32)

        layer_1 = tf.layers.conv2d(self.__input, filters=32, kernel_size=10, activation=tf.nn.relu, padding='valid')
        layer_1 = tf.layers.dropout(layer_1, self.__dropout)

        layer_2 = tf.layers.conv2d(layer_1, filters=16, kernel_size=5, activation=tf.nn.relu, padding='valid')
        layer_2 = tf.layers.dropout(layer_2, self.__dropout)

        layer_3 = tf.layers.flatten(layer_2)
        layer_3 = tf.layers.dense(layer_3, 1024, tf.nn.relu)
        layer_3 = tf.layers.dropout(layer_3, self.__dropout)

        self.__layer_out = tf.layers.dense(layer_3, self.__board_size**2, activation=None)
        self.__softmax = tf.nn.softmax(self.__layer_out)

        x_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.__layer_out, labels=self.__output)
        self.__loss = tf.reduce_mean(self.__rewards * x_entropy)
        self.__training_optimizer = tf.train.AdamOptimizer(self.__learning_rate).minimize(self.__loss)

        init = tf.global_variables_initializer()
        self.__session.run(init)


    def __load_model(self):
        if (os.path.isfile(self.__model_path + self.__model_name + '.meta')):
            if self.__saver is None:
                self.__saver = tf.train.import_meta_graph(self.__model_path + self.__model_name + '.meta')
            self.__saver.restore(self.__session, tf.train.latest_checkpoint(self.__model_path))
            
            # save variables
            self.__input = tf.get_collection("input")[0]
            self.__output = tf.get_collection("output")[0] 
            self.__rewards = tf.get_collection("rewards")[0] 
            self.__layer_out = tf.get_collection("output_layer")[0] 
            self.__softmax = tf.get_collection("scaled_output")[0] 
            self.__loss = tf.get_collection("loss")[0]
            self.__training_optimizer = tf.get_collection('optimizer')[0]
            self.__dropout =tf.get_collection('dropout')[0]

        else:
            self.__architechture()
            # Save meta if new architecture created
            self.__save_model(save_meta=True)

    def __save_model(self, save_meta=False):
        if self.__saver is None:
            self.__saver = tf.train.Saver()
        tf.add_to_collection("input", self.__input) 
        tf.add_to_collection("output", self.__output) 
        tf.add_to_collection("rewards", self.__rewards) 
        tf.add_to_collection("output_layer", self.__layer_out) 
        tf.add_to_collection("scaled_output", self.__softmax) 
        tf.add_to_collection("loss", self.__loss) 
        tf.add_to_collection('optimizer', self.__training_optimizer)
        tf.add_to_collection('dropout', self.__dropout)

        self.__saver.save(self.__session, self.__model_path + self.__model_name, write_meta_graph=save_meta)

if __name__ == "__main__":
    ai = Ai()
    ai2 = Ai()
    ai.predict({ "board": np.zeros((25, 25)), "player": 1 })
