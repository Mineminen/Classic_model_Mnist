import torch.nn as nn
class AlexNet8(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet8, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),          # Conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),         # Conv3
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),         # Conv4
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),         # Conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 4096),  # FC1
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),         # FC2
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)   # FC3
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) #x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x