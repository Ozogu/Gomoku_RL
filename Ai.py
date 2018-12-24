import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import numpy as np
import os
import string
import random

class Ai():
    def __init__(self):
        # Discount factor
        self.__gamma = 0.95
        self.__learning_rate = 0.0001
        self.__epochs = 1
        self.batch_size = 1 # Public

        self.__episodes = []
        self.__episode = { "observations": [], "actions": [], "reward": 0 }

        self.__model_path = "./model/"
        self.__model_name = "model"
        self.__board_size = 25
        self.__tensorboard()
        self.__session = None
        self.__load_model()

    def new_game(self):
        self.__episodes.append(self.__episode)
        self.__episode = { "observations": [], "actions": [], "reward": 0 }

    def train(self):
        # Load might have been changed by previous Ai
        self.__load_model()
        for _ in range(0, self.__epochs):
            for batch in self.__episodes:
                rewards = self.__discount_and_normalize_rewards(batch["reward"], len(batch["actions"]))
                self.__session.run([self.__loss, self.__training_optimizer],
                    {
                        self.__input: np.vstack(batch["observations"]),
                        self.__output: np.stack(batch["actions"]),
                        self.__rewards: rewards
                    }
                )
        
        self.__save_model()
        self.__episodes = []

    def predict(self, data):
        board = self.__preprocess(data)
        prediction = self.__session.run(self.__sigout, { self.__input: board })
        best_valid_index = self.__best_valid_prediction(prediction, board)
        coordinates = self.__index_to_coordinates(best_valid_index)

        self.__episode["observations"].append(board)
        action = np.zeros(self.__board_size**2)
        action[best_valid_index] = 1
        self.__episode["actions"].append(action)

        return coordinates

    def reward(self, reward):
        self.__episode["reward"] = reward

    def __discount_and_normalize_rewards(self, reward, episode_length):
        rewards = np.zeros(episode_length)
        rewards[0] = reward
        discounted_rewards = np.zeros(episode_length)
        cumulative = 0.0
        for index, reward in enumerate(rewards):
            cumulative = cumulative * self.__gamma + reward
            discounted_rewards[index] = cumulative

        mean = np.mean(discounted_rewards)
        std = np.std(discounted_rewards)
        discounted_rewards = (discounted_rewards-mean) / std

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
        prediction[board != 0] = 0

        # Choose randomly among the most confident choices
        indexes = np.where(prediction == np.max(prediction))[1]
        index = np.random.choice(indexes)

        # If there is no mark at the predicted location, it is the best valid prediction
        if board[0][index] == 0:
            return index
        else:
            try:
                # If for some reason, no valid points have been played, just play the first open slot.
                return np.where(board[0] == 0)[0][0]
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
            board[board == 2] = 3
            board[board == 1] = 2
            board[board == 3] = 1


        reshaped_board = np.reshape(board, (1,self.__board_size**2))

        return reshaped_board

    def __architechture(self):
        # self.__scope = "".join(random.choices(string.ascii_uppercase, k=8))
        layer_neurons = [self.__board_size**2, 256, 256, self.__board_size**2]

        # with tf.variable_scope(self.__scope):
        self.__input = tf.placeholder(shape=[None, layer_neurons[0]], dtype=tf.float32, name="input")
        self.__output = tf.placeholder(shape=[None, layer_neurons[3]], dtype=tf.int32, name="output")
        self.__rewards = tf.placeholder(shape=[None,], dtype=tf.float32, name="rewards")

        weights = [
            tf.Variable(tf.random_normal([layer_neurons[0], layer_neurons[1]])),
            tf.Variable(tf.random_normal([layer_neurons[1], layer_neurons[2]])),
            tf.Variable(tf.random_normal([layer_neurons[2], layer_neurons[3]]))
        ]
        biases = [
            tf.Variable(tf.random_normal([layer_neurons[1]])),
            tf.Variable(tf.random_normal([layer_neurons[2]])),
            tf.Variable(tf.random_normal([layer_neurons[3]]))
        ]

        layer_1 = tf.add(tf.matmul(self.__input, weights[0]), biases[0])
        layer_1 = tf.nn.relu(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, weights[1]), biases[1])
        layer_2 = tf.nn.relu(layer_2)

        self.__layer_out = tf.add(tf.matmul(layer_2, weights[2]), biases[2], name="output_layer")
        self.__sigout = tf.nn.sigmoid(self.__layer_out, name="scaled_output")

        x_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.__layer_out, labels=self.__output)
        self.__loss = tf.reduce_mean(self.__rewards * x_entropy, name="loss")
        self.__training_optimizer = tf.train.AdamOptimizer(self.__learning_rate).minimize(self.__loss, name="optimizer")

        init = tf.global_variables_initializer()
        self.__session.run(init)


    def __load_model(self):
        if (os.path.isfile(self.__model_path + self.__model_name + '.meta')):
            tf.reset_default_graph()
            if (self.__session is not None): self.__session.close()
            self.__session = tf.Session()

            # Load values
            saver = tf.train.import_meta_graph(self.__model_path + self.__model_name + '.meta')
            saver.restore(self.__session, tf.train.latest_checkpoint(self.__model_path))
            
            # save variables
            self.__input = tf.get_collection("input")[0]
            self.__output = tf.get_collection("output")[0] 
            self.__rewards = tf.get_collection("rewards")[0] 
            self.__layer_out = tf.get_collection("output_layer")[0] 
            self.__sigout = tf.get_collection("scaled_output")[0] 
            self.__loss = tf.get_collection("loss")[0]
            self.__training_optimizer = tf.get_collection('optimizer')[0]

            #tf.saved_model.loader.load(self.__session, tag_constants.SERVING, self.__model_path)
        else:
            self.__architechture()
            self.__session = tf.Session()
            init = tf.global_variables_initializer()
            self.__session.run(init)

            # Save meta if new architecture created
            self.__save_model(save_meta=True)

    def __save_model(self, save_meta=False):
        saver = tf.train.Saver()
        tf.add_to_collection("input", self.__input) 
        tf.add_to_collection("output", self.__output) 
        tf.add_to_collection("rewards", self.__rewards) 
        tf.add_to_collection("output_layer", self.__layer_out) 
        tf.add_to_collection("scaled_output", self.__sigout) 
        tf.add_to_collection("loss", self.__loss) 
        tf.add_to_collection('optimizer', self.__training_optimizer)
        saver.save(self.__session, self.__model_path + self.__model_name, write_meta_graph=save_meta)
        # tf.saved_model.simple_save(self.__session, self.__model_path, { "self.__input": self.__input }, { "self.__output": self.__output })

if __name__ == "__main__":
    ai = Ai()
    ai2 = Ai()
    ai.predict({ "board": np.zeros((25, 25)), "player": 1 })
