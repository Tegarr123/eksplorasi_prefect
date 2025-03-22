import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(1, 32, 3, 1)

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # Fully connected layer 1
        self.fc1 = nn.Linear(1600, 1200)

        # Fully connected layer 2
        self.fc2 = nn.Linear(1200, 10)

    def forward(self, x):
        # Pass the input image through the convolutional layers
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)

        # Flatten the output from the convolutional layers
        x = x.view(-1, 1600)

        # Pass the output from the convolutional layers through the fully connected layers
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)

        # Return the output of the fully connected layers
        return x