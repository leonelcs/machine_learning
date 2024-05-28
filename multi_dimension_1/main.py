from linear_regression import LinearRegression
import numpy as np

class Main:

    def __init__(self):
        self._name = 'Main'

    def start(self):
        #defaults to 0.01 and 2000
        self.linear_regression = LinearRegression(file="life_expectancy.txt", learning_rate=0.0001, n_iters=100000)

if __name__ == '__main__':
    main = Main()
    main.start()

    w = main.linear_regression.train()

print("\nWeights: %s" % w.T)
print("\nA few predictions:")
for i in range(5):
    print("X[%d] -> %.4f (label: %d)" % (i, main.linear_regression.predict(main.linear_regression.X[i], w), main.linear_regression.Y[i]))