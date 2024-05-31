import numpy as np
import gzip
import struct

class MNISTLoader:

    all_pixels = None
    all_labels = None

    def __init__(self):
        pass

    def load_image(self, filename):
        print(filename)
        with gzip.open(filename, "rb") as f:
            
            _ignored, n_images, columns, rows = struct.unpack(">IIII", f.read(16))
            all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
            return all_pixels.reshape(n_images, columns * rows)
        
    def prepend_bias(self, X):
        return np.insert(X, 0, 1, axis=1)
    
    def load_labels(self, filename):
    # Open and unzip the file of images:
        with gzip.open(filename, 'rb') as f:
            # Skip the header bytes:
            f.read(8)
            # Read all the labels into a list:
            all_labels = f.read()
            # Reshape the list of labels into a one-column matrix:
            return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)

    def encode_fives(self, Y):
    # Convert all 5s to 1, and everything else to 0
        return (Y == 5).astype(int)





