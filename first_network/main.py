from neuro_network import NeuroNetwork
from mnist import MNISTLoader

def main():
    # Load the MNIST data:
    mnist_loader = MNISTLoader()
    X_train = mnist_loader.load_image(filename="data/train-images-idx3-ubyte.gz")
    X_test = mnist_loader.load_image(filename="data/t10k-images-idx3-ubyte.gz")

    Y_train_unencoded = mnist_loader.load_labels("data/train-labels-idx1-ubyte.gz")
    Y_train = mnist_loader.one_hot_encode(Y_train_unencoded)

    

    Y_test_unencoded = mnist_loader.load_labels("data/t10k-labels-idx1-ubyte.gz")
    Y_test = mnist_loader.one_hot_encode(Y_test_unencoded)
    
    # # Train the model:
    nn = NeuroNetwork()

    w1, w2 = nn.train(X_train, Y_train,
               X_test, Y_test,
               n_hidden_nodes=200, iterations=20, lr=0.01)


    # Test the model:


if __name__ == "__main__":
    main()

