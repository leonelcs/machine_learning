import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    
    def __init__(self, learning_rate=0.01, n_iters=10000):
        self.lr = learning_rate
        print("lr: %4f" % self.lr)
        self.n_iters = n_iters
        self.X, self.Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

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

    def train(self):
        w = b = 0
        print("iterations: %4d LR: %4f" % (self.n_iters, self.lr))
        for i in range(self.n_iters):
            current_loss = self.loss(w, b)
            print("Interaction %4d => Loss: %6f" % (i, current_loss))
            # copilot suggested using if instead of elif, but it changes the format
            if self.loss(w + self.lr, b) < current_loss:
                w += self.lr
            elif self.loss(w - self.lr, b) < current_loss:
                w -= self.lr
            elif self.loss(w, b + self.lr) < current_loss:
                b += self.lr
            elif self.loss(w, b - self.lr) < current_loss:
                b -= self.lr
            else:
                return w, b

        raise Exception("Couldn't converge within %d iterations" % self.n_iters)


    def predict(self, X, w, b):
        return X*w+b
    
    def loss(self, w, b):
        # copilot suggested mean instead of average
        return np.average((self.predict(self.X, w, b) - self.Y) ** 2)