import numpy as np
import matplotlib.pyplot as plt
import sys

class LinearRegression:
    def __init__(self, file="pizza_3_vars.txt",learning_rate=0.001, n_iters=100000):
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
        current_loss = sys.float_info.max
        for i in range(self.n_iters):
            new_loss = self.loss(w)
            if abs(current_loss - new_loss) > precision:
                current_loss = new_loss
                print("Iteration %d, Loss=%.10f" % (i, new_loss))
                w -= self.gradient(w)*self.lr
            else:
                return w
        return w

        raise Exception("Couldn't converge within %d iterations" % self.n_iters)
    
    def gradient(self, w):
        return 2 * np.matmul(self.X.T, (self.predict(self.X, w) - self.Y)) / self.X.shape[0]


    def predict(self, X, w):
        return np.matmul(X, w)
    
    def loss(self, w):
        # copilot suggested mean instead of average
        return np.average((self.predict(self.X, w) - self.Y) ** 2)