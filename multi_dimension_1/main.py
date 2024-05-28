from linear_regression import LinearRegression
import numpy as np

class Main:

    def __init__(self):
        self._name = 'Main'

    def start(self):
        #defaults to 0.01 and 2000
        self.linear_regression = LinearRegression()

if __name__ == '__main__':
    main = Main()
    main.start()

    w = main.linear_regression.train()

    x1 = 20
    x2 = 25
    x3 = 10
    X = np.column_stack((x1, x2, x3))

    print("Prediction: x=",X, "prediction: ", main.linear_regression.predict(X, w))