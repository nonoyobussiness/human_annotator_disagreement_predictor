from torchvision.datasets import CIFAR10

cifar10_train = CIFAR10(root='data/raw', train=True, download=True)
cifar10_test=CIFAR10(root='data/raw', train=False, download=True)

print("Download complete!")