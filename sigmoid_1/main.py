from classification import Classification
import numpy as np

class Main:

    def __init__(self):
        self._name = 'Main'

    def start(self):
        #defaults to 0.01 and 2000
        self.classification = Classification(learning_rate=0.001, n_iters=10000)

if __name__ == '__main__':
    main = Main()
    main.start()

    w = main.classification.train()

    main.classification.test(w)