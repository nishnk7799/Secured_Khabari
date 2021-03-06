import torch
import torch.nn as nn
'''
nn.Conv2d()
in_channels: the number of channels of the input signal
out_channels: the number of channels generated by convolution
kernel_size: Convolution kernel size
stride: step size
padding: complement 0
dilation: kernel spacing
'''


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
def conv4x4(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding=1)
def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=stride, padding=1)
# Hide image network
class Hide(nn.Module):
    def __init__(self):
        super(Hide, self).__init__()
        # As a container, the module will be added to the module in the order passed in the constructor
        self.prepare = nn.Sequential(
            # Input 3 dimensions, output 64 dimensions, convolution kernel size 3, step size 1, complement 1
            conv3x3(3, 64),
            # Activation function
            # inplace is True, the input data will be changed, otherwise the original input will not be changed, only new output will be generated
            nn.ReLU(True),
            # Input 64 dimensions, output 64 dimensions, step size is 2, complement 1
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.hidding_1 = nn.Sequential(
            # Input 128 dimensions, output 64 dimensions, convolution kernel size 1, step size 2, complement 0
            nn.Conv2d(128, 64, 1, 1, 0),
            # Normalization of data, which makes the data not cause unstable network performance due to excessive data before Relu
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Input 64 dimensions, output 64 dimensions, convolution kernel size 3, step size 1, complement 1
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Input 64 dimensions, output 32 dimensions, convolution kernel size 3, step size 1, complement 1
            conv3x3(64, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        #
        self.hidding_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Input 32 dimensions, output 16 dimensions, convolution kernel size 3, step size 1, complement 1
            conv3x3(32, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Input 16 dimensions, output 3 dimensions, convolution kernel size 3, step size 1, complement 1
            conv3x3(16, 3),
            # Activation function
            nn.Tanh()
        )
    def forward(self, secret, cover):
        # Process the secret image

        sec_feature = self.prepare(secret)
        # Process the carrier image
        cover_feature = self.prepare(cover)

        # Stitch the two trained images horizontally before training
        out = self.hidding_1(torch.cat([sec_feature, cover_feature], dim=1))

        out = self.hidding_2(out)
        return out
# Show secret image network
class Reveal(nn.Module):
    def __init__(self):
        super(Reveal, self).__init__()
        self.reveal = nn.Sequential(
            conv3x3(3, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            conv3x3(32, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            conv3x3(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            conv3x3(64, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            conv3x3(32, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            conv3x3(16, 3),
            nn.Tanh()
        )

    def forward(self, image):
        out = self.reveal(image)
        return out

