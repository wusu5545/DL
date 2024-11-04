import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = None
        if (stride != 1) or (in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), # 1x1 convolution
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        skip_connection = input_tensor
        output = self.block(input_tensor)
        if self.shortcut:
            skip_connection = self.shortcut(skip_connection)

        output += skip_connection
        output = self.relu(output)
        return output


class ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResBlock(64, 64, 1),
            #nn.Dropout(0.1),
            ResBlock(64, 128, 2),
            nn.Dropout(0.1),
            ResBlock(128, 256, 2),
            nn.Dropout(0.2),
            ResBlock(256, 512, 2),
            nn.AvgPool2d(kernel_size=10),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        return self.net(input_tensor)