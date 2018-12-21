import tensorflow as tf
import numpy as np

class Ai():
    def __init__(self):
        self.__weights_dir = ""
        self.__board_size = 25
        self.__model = self.__architechture()
        self.__tensorboard()
        self.__session = tf.Session()
        init = tf.global_variables_initializer()
        self.__session.run(init)

    def predict(self, data):
        processed_data = self.__preprocess(data)
        prediction = self.__session.run(self.__model, { "Placeholder:0": processed_data })
        best_valid_index = self.__best_valid_prediction(prediction, processed_data)
        coordinates = {
            "x": best_valid_index%self.__board_size,
            "y": self.__board_size
        }

        return coordinates
    
    def __index_to_coordinates(self, index):
        size = self.__board_size
        c = {
            "x": index % size,
            "y": int(index / size)
        }
        # Make sure that index is transformed correctly
        assert c["y"]*size+c["x"] == index

        return c

    def __best_valid_prediction(self, prediction, board):
        # Loop once at most to avoid infinite loop
        # for _ in prediction:
        #     # Find index of best prediction
        #     # prediction: (1,625)
        #     index = self.__session.run(tf.argmax(prediction,1))[0]
        #     # If there is no mark at the predicted location, it is the best valid prediction
        #     if board[0][index] == 0:
        #         return index
        #     # If there is mark, set the prediction to 0 so it's no longer the best
        #     else:
        #         prediction[1][index] = 0

        # If for some reason, no valid points have been played, just play the first open slot.
        try:
            return np.where(board == 0)[0][0]
        except IndexError:
            np.savetxt("board.log", board)
            raise IndexError("Every slot has been played!")


    def __tensorboard(self):
        writer = tf.summary.FileWriter('.')
        writer.add_graph(tf.get_default_graph())
        writer.flush()

    def __preprocess(self, data):
        return np.reshape(data, (1,25*25))

    def __architechture(self):
        layer_neurons = [25*25, 256, 256, 25*25]
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
    ai.predict(np.zeros((25, 25)))
