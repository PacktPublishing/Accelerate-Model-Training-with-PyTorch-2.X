from torchvision import datasets

data_dir = '/tmp'

train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)
test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True)















