import os
import torch

from ocr_nn import Model
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = MNISTDataset()
ocr_model = Model(num_of_classes=10)  # 10 digits


def train():
    loss_progress_train = []

    max_epochs = 3
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
            loss_progress_train.append(loss.item())
            # accuracy_progress_train.append(acc_train)
            if mini_batch % 10 == 0:
                print(f'[{epoch}, {mini_batch}] loss: {running_loss / 10:.2f}')
                running_loss = 0.0
        plot_progress(loss_progress_train, 'mini_batch', 'loss_progress')
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

    print(f'Accuracy {correct / len(test_loader.dataset) * 100:.2f}%')


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
                res = predict_one_img(image_path, show)
                results.append(res)
        else:
            results = [predict_one_img(path_to_data, show)]
    save_array_as_txt(results)


def predict_one_img(path_to_data, show):
    ocr_model.to(device)
    ocr_model.eval()
    image_preproc = ImagePreprocessor()
    with torch.no_grad():
        if show:
            im = cv2.imread(path_to_data)
            show_imgs_cv2([im])  # for testing
        gray_img = cv2.imread(path_to_data, cv2.IMREAD_GRAYSCALE)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        gray_img = image_preproc.apply_opening(gray_img, kernel_dil=kernel)  # anti noise
        # invert img, in mnist there is black background & white digit
        _, inv_img = cv2.threshold(255 - gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img = image_preproc.crop_image(inv_img)
        img = image_preproc.fit_and_resize_to_mnist_format(img)
        # shift the inner box so that it is centered using the center of mass
        shiftx, shifty = image_preproc.getBestShift(img)
        shifted = image_preproc.shift(img, shiftx, shifty)
        # apply opening, after previous operations quality gets worse, holes appear
        img = image_preproc.apply_opening(shifted, show=show)
        # threshold again, gray pixels appeared
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite('data/img_results/' + os.path.basename(path_to_data), img)  # for testing purposes

        # evaluation
        img = img[np.newaxis, ...]  # adding new axis as grayscale removes it
        batch = torch.from_numpy(img).unsqueeze(0)
        batch = batch.to(torch.float32)
        output = ocr_model(batch)
        prediction = torch.argmax(output, 1).item()
        print(f'Prediction is {prediction}')
    return prediction


if __name__ == "__main__":
    # train()
    ocr_model.load_state_dict(torch.load('data/weights/weight.pt'))
    test()
    make_prediction('data/real', show=False)
