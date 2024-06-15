import numpy as np
import matplotlib.pyplot as plt
import sys

class Classification:
        
    def __init__(self, learning_rate=0.001, n_iters=200):
        self.lr = learning_rate
        print("lr: %4f" % self.lr)
        self.n_iters = n_iters

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

    def report(self, iteration, X_train, Y_train, X_test, Y_test, w):
        matches = np.count_nonzero(self.classify(X_test, w) == Y_test)
        n_test_examples = Y_test.shape[0]
        matches_percent = matches * 100 / n_test_examples
        training_loss = self.loss(X_train, Y_train, w)
        if (iteration%20 == 0)  or iteration == 199:
            print("Iteration %4d => Loss: %.20f, Test Accuracy: %.2f%%" % (iteration, training_loss, matches_percent))
    
    

    def train(self, train_X, train_Y, test_X, test_Y, lr):
        w = np.zeros((train_X.shape[1], train_Y.shape[1]))
        for i in range(self.n_iters):
            self.report(i, train_X, train_Y, test_X, test_Y, w)
            w -= self.gradient(train_X, train_Y, w) * self.lr
        self.report(self.n_iters, train_X, train_Y, test_X, test_Y, w)
        return w
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X, w):
        weighted_sum =  np.matmul(X, w)
        return self.sigmoid(weighted_sum)
    
    def classify(self, X, w):
        y_hat = self.forward(X, w)
        labels = np.argmax(y_hat, axis=1)
        return labels.reshape(-1, 1)
    
    def mse_loss(self, Y, w):
        return np.average((self.forward(self.X, w) - Y) ** 2)
    
    def loss(self, X, Y, w):
        y_hat = self.forward(X, w)
        first_term = Y * np.log(y_hat)
        second_term = (1 - Y) * np.log(1 - y_hat)
        return -np.average(first_term + second_term)
    
    def gradient(self, X, Y, w):
        return np.matmul(X.T, (self.forward(X, w) - Y)) / X.shape[0]
    
    # Doing inference to test our model
    def test(self, X, Y, w):
        total_examples = X.shape[0]
        correct_results = np.sum(self.classify(X, w) == Y)
        success_percent = correct_results * 100 / total_examples
        print("\nSuccess: %d/%d (%.2f%%)" % (correct_results, total_examples, success_percent))
