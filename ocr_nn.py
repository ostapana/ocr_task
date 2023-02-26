import torch


class Model(torch.nn.Module):
    def __init__(self, num_of_classes):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.hid1 = torch.nn.Linear(256, 120)
        self.hid2 = torch.nn.Linear(120, 84)
        self.output = torch.nn.Linear(84, num_of_classes)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.relu(self.hid1(x))
        x = self.relu(self.hid2(x))
        x = self.output(x)
        return x

