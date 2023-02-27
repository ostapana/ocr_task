import math
import matplotlib.pyplot as plt
import numpy as np
import cv2

from torch import utils
from torchvision import transforms, datasets
from scipy import ndimage


def save_array_as_txt(array):
    with open("data/output.txt", "w") as txt_file:
        for digit in array:
            txt_file.write(" ".join(str(digit)) + "\n")


def show_imgs_cv2(imgs: []):
    for im in imgs:
        cv2.imshow('image', im)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_progress(data, x_label, y_label):
    plt.plot(data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig('data/evaluation_figs/' + y_label + '.png')


class MNISTDataset:
    """
    Create a data loader, that shuffles data from mnist dataset
    returns them in batches of size n
    """

    def __init__(self):
        self.t = transforms.ToTensor()
        self.n = 500

    def get_mnist_dataset(self, train):
        mnist = datasets.MNIST(root='data', train=train, download=True, transform=self.t)
        loader = utils.data.DataLoader(mnist, batch_size=self.n, shuffle=True)
        return loader


class ImagePreprocessor:
    """
    Taken (mostly) from https://opensourc.es/blog/tensorflow-mnist/
    All images are size normalized to fit in a 20x20 pixel box
    and there are centered in a 28x28 image using the center of mass.
    """

    def __init__(self):
        pass

    def crop_image(self, img: np.ndarray):
        """
        Remove black rows and columns from the image
        https://opensourc.es/blog/tensorflow-mnist/
        """
        while np.sum(img[0]) == 0:
            img = img[1:]

        while np.sum(img[:, 0]) == 0:
            img = np.delete(img, 0, 1)

        while np.sum(img[-1]) == 0:
            img = img[:-1]

        while np.sum(img[:, -1]) == 0:
            img = np.delete(img, -1, 1)

        return img

    def apply_opening(self, img: np.ndarray, kernel_dil=None, kernel_erode=None, show=False):
        """
        Erosion followed by dilation
        """
        if kernel_dil is None:
            kernel_dil = np.ones((3, 3), np.uint8)
        if kernel_erode is None:
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        dilated_img = cv2.dilate(img, kernel_dil, iterations=1)
        opened_img = cv2.erode(dilated_img, kernel_erode, iterations=1)
        if show:
            show_imgs_cv2([dilated_img, opened_img])
        return opened_img

    def fit_and_resize_to_mnist_format(self, img: np.ndarray):
        """
        Resize img to 20x20 and then pad to 28x28
        https://opensourc.es/blog/tensorflow-mnist/
        """
        rows, cols = img.shape
        if rows > cols:
            factor = 20.0 / rows
            rows = 20
            cols = int(round(cols * factor))
            img = cv2.resize(img, (cols, rows))
        else:
            factor = 20.0 / cols
            cols = 20
            rows = int(round(rows * factor))
            img = cv2.resize(img, (cols, rows))

        colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
        rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
        img = np.lib.pad(img, (rowsPadding, colsPadding), 'constant')
        return img

    def getBestShift(self, img):
        """
        Get center of mass
        """
        cy, cx = ndimage.center_of_mass(img)

        rows, cols = img.shape
        shiftx = np.round(cols / 2.0 - cx).astype(int)
        shifty = np.round(rows / 2.0 - cy).astype(int)

        return shiftx, shifty

    def shift(self, img, sx, sy):
        rows, cols = img.shape
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted = cv2.warpAffine(img, M, (cols, rows))
        return shifted
