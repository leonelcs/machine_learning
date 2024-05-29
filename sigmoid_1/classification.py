import numpy as np
import matplotlib.pyplot as plt
import sys

class Classification:
    def __init__(self, file="police.txt",learning_rate=0.001, n_iters=100000):
        self.lr = learning_rate
        print("lr: %4f" % self.lr)
        self.n_iters = n_iters
        x1, x2, x3, y = np.loadtxt(file, skiprows=1, unpack=True)

        self.X = np.column_stack((np.ones(x1.size), x1, x2, x3))
        self.Y = y.reshape(-1, 1)

    def plot(self):
        plt.scatter(self.X, self.Y)
        plt.axis([0, 50, 0, 50])                                 # scale axes (0 to 50)
        plt.xticks(fontsize=14)                                  # set x axis ticks
        plt.yticks(fontsize=14)
        plt.xlabel('Reservations', fontsize=14)
        plt.ylabel('Pizzas', fontsize=14)
        plt.title('Pizza Deliveries vs Reservations')
        plt.plot(self.X, self.Y, 'bo')
        plt.show()

    def train(self, precision=0.0001):
        w = np.zeros((self.X.shape[1], 1))
        for i in range(self.n_iters):
            if (i%2000==0 or i==9999):
                print("Iteration %4d => Loss: %.20f" % (i, self.loss(w)))
            w -= self.gradient(w) * self.lr
        return w
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X, w):
        weighted_sum =  np.matmul(X, w)
        return self.sigmoid(weighted_sum)
    
    def classify(self, X, w):
        return np.round(self.forward(X, w))
    
    def mse_loss(self, w):
        return np.average((self.forward(self.X, w) - self.Y) ** 2)
    
    def loss(self, w):
        y_hat = self.forward(self.X, w)
        first_term = self.Y * np.log(y_hat)
        second_term = (1 - self.Y) * np.log(1 - y_hat)
        return -np.average(first_term + second_term)
    
    def gradient(self, w):
        return np.matmul(self.X.T, (self.forward(self.X, w) - self.Y)) / self.X.shape[0]
    
    # Doing inference to test our model
    def test(self, w):
        total_examples = self.X.shape[0]
        correct_results = np.sum(self.classify(self.X, w) == self.Y)
        success_percent = correct_results * 100 / total_examples
        print("\nSuccess: %d/%d (%.2f%%)" % (correct_results, total_examples, success_percent))
