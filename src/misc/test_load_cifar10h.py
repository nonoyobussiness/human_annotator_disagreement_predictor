import numpy as np

probs = np.load("data/raw/cifar10h-probs.npy")

print("Shape:", probs.shape)
print("First row:", probs[0])
print("Sum of first row:", probs[0].sum())