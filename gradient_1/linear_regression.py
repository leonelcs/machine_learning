import numpy as np
import sys
import matplotlib.pyplot as plt

class LinearRegression:
    
    def __init__(self, learning_rate=0.001, n_iters=5000):
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

    def gradient(self, w, b):
        result_w = 2*np.average(self.X*(self.predict(self.X, w, b) - self.Y))
        result_b = 2*np.average(self.predict(self.X, w, b) - self.Y)
        return result_w, result_b

    def train(self):
        w = b = 0
        current_loss = sys.float_info.max
        for i in range(self.n_iters):
            new_loss = self.loss(w, b)
            if abs(current_loss - new_loss) > 0.001:
                current_loss = new_loss
                print("Iteration %d: w=%.3f, b=%.3f, Loss=%.10f" % (i, w, b, new_loss))
                w_gradient, b_gradient = self.gradient(w, b)
                w -= w_gradient*self.lr
                b -= b_gradient*self.lr
            else:
                return w, b
        return w, b


    def predict(self, X, w, b):
        return X*w+b
    
    def loss(self, w, b):
        result = np.average((self.predict(self.X, w, b) - self.Y) ** 2)
        print("w=%.3f and b=%.3f => Loss: %.10f" % (w, b, result))
        return result
    
