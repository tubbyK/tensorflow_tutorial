# as learned on https://www.tensorflow.org/tutorials/quickstart/beginner
# MNIST (handwritten digits) is like the "hello world" of deep learning

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class tensorflow_tutorial_mnist():
    def __init__(self):
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None
        self.predictions = None
        self.history = None
        self.plt_index = 0

    def run(self):
        self.load_data()
        self.build_model()
        self.make_predictions()
        self.run_model(epochs=5)
        self.eval_model()
        self.plot_accuracy()

    def plot_accuracy(self):
        plt.figure(self.plt_index)
        self.plt_index += 1
        plt.plot(self.history.history['accuracy'])
        plt.show()

    def load_data(self):
        mnist = tf.keras.datasets.mnist
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        self.X_train, self.X_test = self.X_train/255.0, self.X_test/255.0

    def build_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28,28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(np.unique(self.y_train)))
        ])

    def make_predictions(self):
        self.predictions = self.model(self.X_train[:1]).numpy()
        tf.nn.softmax(self.predictions).numpy()

    def run_model(self, epochs=5):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss_fn(self.y_train[:1], self.predictions).numpy()
        self.model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        self.history = self.model.fit(self.X_train, self.y_train, batch_size=128, epochs=epochs)

    def eval_model(self):
        self.model.evaluate(self.X_test, self.y_test, verbose=2)
        probability_model = tf.keras.Sequential([
            self.model,
            tf.keras.layers.Softmax()
        ])
        probability_model(self.X_test[:5])

if __name__ == '__main__':
    t = tensorflow_tutorial_mnist()
    t.run()