from linear_regression import LinearRegression

class Main:

    def __init__(self):
        self._name = 'Main'

    def start(self):
        #defaults to 0.01 and 2000
        self.linear_regression = LinearRegression()

if __name__ == '__main__':
    main = Main()
    main.start()

    # main.linear_regression.plot()

    w, b = main.linear_regression.train()
    print("\nw=%.10f and b=%.3f" % (w,b))
    print("Prediction: x=%d => y=%.3f" % (20, main.linear_regression.predict(20, w, b)))