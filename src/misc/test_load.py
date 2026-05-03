import pickle
import numpy as np

with open('data/raw/cifar-10-batches-py/test_batch', 'rb') as f:
    batch = pickle.load(f, encoding='bytes')

images=batch[b'data']
labels=batch[b'labels']

probs = np.load("data/raw/cifar10h-probs.npy")

print("Image Count:", len(images))
print("Soft label count:", len(probs))