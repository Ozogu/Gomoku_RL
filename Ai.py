import tensorflow as tf
import numpy as np

class Ai():
    def __init__(self):
        self.__weights_dir = ""
        self.__board_size = 25
        self.__model = self.__architechture()
        # self.__tensorboard()
        self.__session = tf.Session()
        init = tf.global_variables_initializer()
        self.__session.run(init)

    def new_game(self):
        pass

    def train(self, winner):
        pass

    def predict(self, data):
        processed_data = self.__preprocess(data)
        prediction = self.__session.run(self.__model, { "Placeholder:0": processed_data })
        best_valid_index = self.__best_valid_prediction(prediction, processed_data)
        coordinates = self.__index_to_coordinates(best_valid_index)

        return coordinates
    
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
        index = self.__session.run(tf.argmax(prediction,1))[0]
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
        layer_neurons = [self.__board_size**2, 256, 256, self.__board_size**2]
        input = tf.placeholder(shape=[1, layer_neurons[0]], dtype=tf.float32)
        # output = tf.placeholder(shape=layer_neurons[3], dtype=tf.float32)
        weights = [
            tf.Variable(tf.random_normal(
                [layer_neurons[0], layer_neurons[1]])),
            tf.Variable(tf.random_normal(
                [layer_neurons[1], layer_neurons[2]])),
            tf.Variable(tf.random_normal([layer_neurons[2], layer_neurons[3]]))
        ]
        biases = [
            tf.Variable(tf.random_normal([layer_neurons[1]])),
            tf.Variable(tf.random_normal([layer_neurons[2]])),
            tf.Variable(tf.random_normal([layer_neurons[3]]))
        ]

        layer_1 = tf.add(tf.matmul(input, weights[0]), biases[0])
        layer_1 = tf.nn.relu(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, weights[1]), biases[1])
        layer_2 = tf.nn.relu(layer_2)
        layer_out = tf.add(tf.matmul(layer_2, weights[2]), biases[2])
        layer_out = tf.nn.sigmoid(layer_out)

        return layer_out

    def __load_weights(self):
        pass

    def __save_weights(self):
        pass

    def __train(self):
        pass

if __name__ == "__main__":
    ai = Ai()
    ai.predict({ "board": np.zeros((25, 25)), "player": 1 })
