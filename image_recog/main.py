from classification import Classification
from mnist import MNISTLoader

def main():
    # Load the MNIST data:
    mnist_loader = MNISTLoader()
    image_loaded = mnist_loader.load_image(filename="data/train-images-idx3-ubyte.gz")
    X_train = mnist_loader.prepend_bias(image_loaded)
    test_loaded = mnist_loader.load_image(filename="data/t10k-images-idx3-ubyte.gz")
    X_test = mnist_loader.prepend_bias(test_loaded)
    Y_train = mnist_loader.encode_fives(mnist_loader.load_labels(filename="data/train-labels-idx1-ubyte.gz"))
    Y_test = mnist_loader.encode_fives(mnist_loader.load_labels(filename="data/t10k-labels-idx1-ubyte.gz"))

    # Train the model:
    classifier = Classification(learning_rate=1e-5, n_iters=100)
    classifier.X = X_train
    classifier.Y = Y_train
    w = classifier.train()

    # Test the model:
    classifier.test(w)

if __name__ == "__main__":
    main()

