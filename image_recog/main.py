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
    TEST_LABELS = mnist_loader.load_labels(filename="data/t10k-labels-idx1-ubyte.gz")
    Y_train = []
    Y_test = []

    for i in range(10):
        Y_train.append(mnist_loader.encode_numbers(TRAINING_LABELS, i))
        Y_test.append(mnist_loader.encode_numbers(TEST_LABELS, i))


    # Train the model:
    classifier = Classification(learning_rate=1e-5, n_iters=100)
    classifier.X = X_train
    classifier.Y = Y_train

    for i in range(10):
        print("\nTraining model for digit %d" % i)
        w = classifier.train(Y_train[i])
        classifier.test(X_test, Y_test[i], w)


    # Test the model:
    

if __name__ == "__main__":
    main()

