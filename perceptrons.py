import struct
import numpy as np
import sys
import matplotlib.pyplot as plt
import cProfile
import pstats


class perceptron_learning:
    def __init__(self):
        self.train_image_data = self.read_mnist_file("t10k-images-idx3-ubyte")
        self.train_label_data = self.read_mnist_file("t10k-labels-idx1-ubyte")
        self.test_image_data = self.read_mnist_file("train-images-idx3-ubyte")
        self.test_label_data = self.read_mnist_file("train-labels-idx1-ubyte")
        self.prep_input_data()

        self.lrate = 0.001
        self.epochs = 5
        self.weights = np.random.rand(785, 10) - 0.5

    def train_data(self):
        return zip(self.train_image_data, self.train_label_data)

    def test_data(self):
        return zip(self.test_image_data, self.test_label_data)

    def run(self):
        train_correct = []
        test_correct = []
        train_correct.append(self.process(self.train_data(), learn=False))
        test_correct.append(self.process(self.test_data(), learn=False))
        for _ in range(self.epochs):
            train_correct.append(self.process(self.train_data(), learn=True))
            test_correct.append(self.process(self.test_data(), learn=False))
        self.plot_epoch_pred_rate(test_correct, train_correct)

    def process(self, data, learn):
        guesses = 0
        wrong_guesses = 0
        for in_vect, label in data:
            guesses += 1
            out = in_vect @ self.weights
            if np.argmax(out) != label:
                wrong_guesses += 1
                if learn:
                    self.learn(in_vect, label, out)
        return 1 - wrong_guesses / guesses

    def learn(self, in_vect, label, out):
        error = self.one_hot_encoder(label) - self.activate(out)
        self.weights += np.outer(in_vect, error) * self.lrate

    def one_hot_encoder(self, hot, len=10):
        encoded = np.zeros(len)
        encoded[hot] = 1
        return encoded

    def activate(self, out):
        out[out > 0] = 1
        out[out < 0] = 0
        return out

    def plot_epoch_pred_rate(self, test_correct, train_correct):
        epochs_axis = list(range(self.epochs + 1))
        plt.plot(epochs_axis, test_correct, label="Test")
        plt.plot(epochs_axis, train_correct, label="Train")
        plt.title("Learning")
        plt.xlabel("Epochs")
        plt.ylabel("Correct pred. rate")
        plt.legend()
        plt.show()

    def read_mnist_file(self, path):
        with open(path, "rb") as f:
            _, _, d_type, num_dimensions = np.fromfile(f, dtype=np.dtype(">B"), count=4)
            dims = np.fromfile(f, dtype=np.dtype(">u4"), count=num_dimensions)
            data = np.fromfile(f, dtype=np.dtype(">B"))
            data = np.reshape(data, dims)
            return data

    def prep_input_data(self):
        self.train_image_data = self.flatten_normalize_add_bias(self.train_image_data)
        self.test_image_data = self.flatten_normalize_add_bias(self.test_image_data)

    def flatten_normalize_add_bias(self, data):
        data = np.reshape(data, (data.shape[0], 28 ** 2))
        data = data / 255
        data = np.insert(data, 28 ** 2, 1, axis=1)
        return data


if __name__ == "__main__":
    perceptrons = perceptron_learning()
    perceptrons.run()