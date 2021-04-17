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

        self.lrate = 0.1
        self.max_epochs = 5
        self.weights = (np.random.rand(785, 10) - 0.5) / 10

    def train_data(self):
        return zip(self.train_image_data, self.train_label_data)

    def test_data(self):
        return zip(self.test_image_data, self.test_label_data)

    def run(self):
        epoch = 0
        train_correct = [0]
        test_correct = [0]
        train_correct.append(self.process(self.train_data(), learn=False))
        test_correct.append(self.process(self.test_data(), learn=False))
        while train_correct[-1] - train_correct[-2] > 0.01 and epoch < self.max_epochs:
            epoch += 1
            train_correct.append(self.process(self.train_data(), learn=True))
            test_correct.append(self.process(self.test_data(), learn=False))
        train_correct.pop(0)
        test_correct.pop(0)
        self.plot_epoch_pred_rate(test_correct, train_correct)
        self.get_confusion_matrix(self.test_data())

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

    def get_confusion_matrix(self, data):
        conf_matrix = np.zeros((10, 10))
        for in_vect, label in data:
            out = in_vect @ self.weights
            guess = np.argmax(out)
            conf_matrix[label][guess] += 1
        conf_matrix = conf_matrix / np.linalg.norm(conf_matrix)
        fig, ax = plt.subplots()
        im = ax.imshow(conf_matrix)
        fig.tight_layout()

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
        epochs_axis = list(range(len(train_correct)))
        plt.plot(epochs_axis, train_correct, label="Train")
        plt.plot(epochs_axis, test_correct, label="Test")
        plt.title("Learning")
        plt.xlabel("Epochs")
        plt.ylabel("Correct pred. rate")
        plt.legend()

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
    plt.show()