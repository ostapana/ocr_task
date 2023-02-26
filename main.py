import os
import torch
import cv2
import numpy as np

from ocr_nn import Model
from utils import *

from torchvision import transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = MNISTDataset()
ocr_model = Model(num_of_classes=10)  # 10 digits


def train():
    max_epochs = 1
    lrn_rate = 0.002
    torch.manual_seed(1)
    train_loader = dataset.get_mnist_dataset(train=True)

    ocr_model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ocr_model.parameters(), lr=lrn_rate)

    ocr_model.train()
    for epoch in range(max_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for mini_batch, (images, labels) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = ocr_model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if mini_batch % 10 == 0:
                print(f'[{epoch + 1}, {mini_batch + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0
        torch.save(ocr_model.state_dict(), 'data/weights/weight.pt')
    print('Finished Training')


def test():
    ocr_model.to(device)
    test_loader = dataset.get_mnist_dataset(train=False)
    correct = 0

    ocr_model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            output = ocr_model(images)
            _, predict = torch.max(output, 1)
            correct += (predict == labels).sum().item()

    print(f'Accuracy {correct / len(test_loader.dataset) * 100}%')


def make_prediction(path_to_data, show=False):
    """
    Use trained model and make prediction for an image
    also works for directory with images
    """
    results = []
    ocr_model.eval()
    with torch.no_grad():
        if os.path.isdir(path_to_data):
            for filename in os.listdir(path_to_data):
                image_path = os.path.join(path_to_data, filename)
                res = _load_img_and_predict(image_path, show)
                results.append(res)
        else:
            results = [_load_img_and_predict(path_to_data, show)]
    save_array_as_txt(results)


def _load_img_and_predict(image_path, show):
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gray_resized_img = cv2.resize(gray_img, (28, 28))
    np_img = np.asarray(gray_resized_img, dtype=np.float32)
    np_img = np_img[np.newaxis, ...]  # adding new axis as grayscale removes it
    t = transforms.ToTensor()
    tensor = t(np_img).unsqueeze(0)  # cs of batch processing
    output = ocr_model(tensor)
    _, pred = torch.max(output, 1)
    pred = pred.item()
    if show:
        cv2.imshow(str(pred), gray_img)
        cv2.waitKey(0)
    return pred


if __name__ == "__main__":
    train()
    # ocr_model.load_state_dict(torch.load('data/weights/weight.pt'))
    test()
    make_prediction('data/real')
