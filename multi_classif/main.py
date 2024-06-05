from classification import Classification
from mnist import MNISTLoader

def main():
    # Load the MNIST data:
    mnist_loader = MNISTLoader()
    image_loaded = mnist_loader.load_image(filename="data/train-images-idx3-ubyte.gz")
    X_train = mnist_loader.prepend_bias(image_loaded)
    test_loaded = mnist_loader.load_image(filename="data/t10k-images-idx3-ubyte.gz")
    X_test = mnist_loader.prepend_bias(test_loaded)
    TRAINING_LABELS = mnist_loader.load_labels(filename="data/train-labels-idx1-ubyte.gz")
    Y_train = mnist_loader.one_hot_encode(TRAINING_LABELS)
    TEST_LABELS = mnist_loader.load_labels(filename="data/t10k-labels-idx1-ubyte.gz")

    Y_test = mnist_loader.load_labels("data/t10k-labels-idx1-ubyte.gz")

    # Train the model:
    classifier = Classification(learning_rate=1e-5, n_iters=200)

    w = classifier.train(X_train, Y_train, X_test, Y_test, 1e-5)

    # Test the model:


if __name__ == "__main__":
    main()

