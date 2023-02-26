import os

from torch import utils
from torchvision import transforms, datasets


class MNISTDataset:
    """
    Create a data loader, that shuffles data from mnist dataset
    returns them in batches of size n
    """

    def __init__(self):
        self.t = transforms.ToTensor()
        self.n = 500

    def get_mnist_dataset(self, train):
        download_dataset = False if os.path.exists('data/processed') else True
        mnist = datasets.MNIST(root='data', train=train, download=download_dataset, transform=self.t)
        loader = utils.data.DataLoader(mnist, batch_size=self.n, shuffle=True)
        return loader


def save_array_as_txt(array):
    with open("data/output.txt", "w") as txt_file:
        for digit in array:
            txt_file.write(" ".join(str(digit)) + "\n")
