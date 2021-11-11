import torch
import torchvision
import torch.nn as nn


from UNet.models.base import BaseNet


class UNet(BaseNet):
    """
    Class for UNet architecture
    """
    def __init__(self):
        """
        Initializes UNet class
        Each encoder block consists of:
        - convolutional block
        - max-pooling
        - cropping
        - max-pooling
        """
        super().__init__()
        ##################
        # encoder layer 0
        self.e0_conv = UNetConvBlock(3, 64)
        self.e0_pool = nn.MaxPool2d(2, stride=2, return_indices=False)
        self.e0_crop = nn.Sequential(torchvision.transforms.CenterCrop(392))
        self.e0_pool_idx = nn.MaxPool2d(2, stride=2, return_indices=True)
        #
        #################
        # encoder layer1
        self.e1_conv = UNetConvBlock(64, 128)
        self.e1_pool = nn.MaxPool2d(2, stride=2, return_indices=False)
        self.e1_crop = nn.Sequential(torchvision.transforms.CenterCrop(200))
        self.e1_pool_idx = nn.MaxPool2d(2, stride=2, return_indices=True)
        #
        ###################
        # encoder layer 2
        self.e2_conv = UNetConvBlock(128, 256)
        self.e2_pool = nn.MaxPool2d(2, stride=2, return_indices=False)
        self.e2_crop = nn.Sequential(torchvision.transforms.CenterCrop(104))
        self.e2_pool_idx = nn.MaxPool2d(2, stride=2, return_indices=True)
        #
        ##################
        # encoder layer 3
        self.e3_conv = UNetConvBlock(256, 512)
        self.e3_pool = nn.MaxPool2d(2, stride=2, return_indices=False)
        self.e3_crop = nn.Sequential(torchvision.transforms.CenterCrop(56))
        self.e3_pool_idx = nn.MaxPool2d(2, stride=2, return_indices=True)
        #
        #################
        # bottleneck
        self.bottleneck_conv = UNetConvBlock(512, 1024)
        #
        ###################
        # decoder layer 3
        self.d3_upconv = UNetUpConvBlock(1024, 512)
        self.d3_upsample = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.d3_conv = UNetConvBlock(1024, 512)
        #
        ##################
        # decoder layer 2
        self.d2_upconv = UNetUpConvBlock(512, 256)
        self.d2_upsample = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.d2_conv = UNetConvBlock(512, 256)
        #
        ##################
        # decoder layer 1
        self.d1_upconv = UNetUpConvBlock(256, 128)
        self.d1_upsample = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.d1_conv = UNetConvBlock(256, 128)
        #
        ##################
        # decoder layer 0
        self.d0_upconv = UNetUpConvBlock(128, 64)
        self.d0_upsample = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.d0_conv = UNetConvBlock(128, 64)
        self.d0_conv_last = UNetConvBlockLast(64, 1)
        #
        #################
        # forward propagation variables
        self.e0_fm, self.e0_pool_fm, self.e0_crop_fm, self.idx0_crop_fm = (None,)*4
        self.e1_fm, self.e1_pool_fm, self.e1_crop_fm, self.idx1_crop_fm = (None,)*4
        self.e2_fm, self.e2_pool_fm, self.e2_crop_fm, self.idx2_crop_fm = (None,)*4
        self.e3_fm, self.e3_pool_fm, self.e3_crop_fm, self.idx3_crop_fm = (None,)*4
        self.bneck_fm = (None,)*1
        self.d3_upconv_fm, self.d3_upsample_fm,  self.d3_concat_fm, self.d3_fm = (None,)*4
        self.d2_upconv_fm, self.d2_upsample_fm, self.d2_concat_fm, self.d2_fm = (None,)*4
        self.d1_upconv_fm, self.d1_upsample_fm, self.d1_concat_fm, self.d1_fm = (None,)*4
        self.d0_upconv_fm, self.d0_upsample_fm, self.d0_concat_fm, self.d0_fm = (None,)*4
        self.d0_fm = (None,)*1

    def forward(self, x):
        """
        Implements the forward method of the the UNet class.
        There are encoder and decoder paths.
        Each encoder block consists of the following layers:
        - convolutions followed by ReLU,
        - max pooling to provide a feature map (f.m.) for the next encoder block,
        - cropping output of convolutions,
        - recording the max-pool indices of the cropped f.m. for the corresponding decoder block
        Each decoder block is built of:
        - upconvolution - decreases the number of channels, but keeps the feature map's size
        - upsampling - upsamples the upconvolved f.m. by using the max-pool indices from the corresponding encoder block
        - concatenation - concatenates the upsampled f.m. with the cropped f.m. from the corresponding encoder block
        - convolutions of the concatenated f.m. followed by ReLU
        """
        ###################
        # encoder layer 0
        self.e0_fm = self.e0_conv(x)
        self.e0_pool_fm = self.e0_pool(self.e0_fm)
        self.e0_crop_fm = self.e0_crop(self.e0_fm)
        _, self.idx0_crop_fm = self.e0_pool_idx(self.e0_crop_fm)
        #
        ##################
        # encoder layer 1
        self.e1_fm = self.e1_conv(self.e0_pool_fm)
        self.e1_pool_fm = self.e1_pool(self.e1_fm)
        self.e1_crop_fm = self.e1_crop(self.e1_fm)
        _, self.idx1_crop_fm = self.e1_pool_idx(self.e1_crop_fm)
        #
        ##################
        # encoder layer 2
        self.e2_fm = self.e2_conv(self.e1_pool_fm)
        self.e2_pool_fm = self.e2_pool(self.e2_fm)
        self.e2_crop_fm = self.e2_crop(self.e2_fm)
        _, self.idx2_crop_fm = self.e2_pool_idx(self.e2_crop_fm)
        #
        ##################
        # encoder layer 3
        self.e3_fm = self.e3_conv(self.e2_pool_fm)
        self.e3_pool_fm = self.e3_pool(self.e3_fm)
        self.e3_crop_fm = self.e3_crop(self.e3_fm)
        _, self.idx3_crop_fm = self.e3_pool_idx(self.e3_crop_fm)
        #
        ##################
        # bottleneck
        self.bneck_fm = self.bottleneck_conv(self.e3_pool_fm)
        #
        ##################
        # decoder layer 3 (reverse counting order)
        self.d3_upconv_fm = self.d3_upconv(self.bneck_fm)
        self.d3_upsample_fm = self.d3_upsample(self.d3_upconv_fm, self.idx3_crop_fm)
        self.d3_concat_fm = torch.cat((self.e3_crop_fm, self.d3_upsample_fm), dim=1)
        self.d3_fm = self.d3_conv(self.d3_concat_fm)
        #
        ##################
        # decoder layer 2
        self.d2_upconv_fm = self.d2_upconv(self.d3_fm)
        self.d2_upsample_fm = self.d2_upsample(self.d2_upconv_fm, self.idx2_crop_fm)
        self.d2_concat_fm = torch.cat((self.e2_crop_fm, self.d2_upsample_fm), dim=1)
        self.d2_fm = self.d2_conv(self.d2_concat_fm)
        #
        ##################
        # decoder layer 1
        self.d1_upconv_fm = self.d1_upconv(self.d2_fm)
        self.d1_upsample_fm = self.d1_upsample(self.d1_upconv_fm, self.idx1_crop_fm)
        self.d1_concat_fm = torch.cat((self.e1_crop_fm, self.d1_upsample_fm), dim=1)
        self.d1_fm = self.d1_conv(self.d1_concat_fm)
        #
        ##################
        # decoder layer 0
        self.d0_upconv_fm = self.d0_upconv(self.d1_fm)
        self.d0_upsample_fm = self.d0_upsample(self.d0_upconv_fm, self.idx0_crop_fm)
        self.d0_concat_fm = torch.cat((self.e0_crop_fm, self.d0_upsample_fm), dim=1)
        self.d0_conv_fm = self.d0_conv(self.d0_concat_fm)
        self.d0_fm = self.d0_conv_last(self.d0_conv_fm)

        return self.d0_fm


class UNetConvBlock(nn.Module):
    """
    Implements a convolutional block of the UNet
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        Initializes the class
        ---
        Parameters
        ---
        in_channels: int
            NUmber of channels in the input image
        out_channels: int
            Number of channels in the output image
        kernel_size:
            Size of he convolving kernel
        activation:
            Activation function to be used after the convolution and batch normalization
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))

        return out


class UNetConvBlockLast(nn.Module):
    """
    Implements the last convolutional block of the UNet
    """

    def __init__(self, in_channels, out_channels, kernel_size=1):
        """
        Initializes the class
        ---
        Parameters
        ---
        in_channels: int
            NUmber of channels in the input image
        out_channels: int
            Number of channels in the output image
        kernel_size:
            Size of he convolving kernel
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.batchnorm(out)
        
        return out


class UNetUpConvBlock(nn.Module):
    """
    Implements a up-convolutional block of the UNet
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        """
        Initializes the class
        ---
        Parameters
        ---
        in_channels: int
            NUmber of channels in the input image
        out_channels: int
            Number of channels in the output image
        kernel_size:
            Size of the convolving kernel
        activation:
            Activation function to be used after the convolution and batch normalization
        """
        super().__init__()

        self.convtr1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()


    def forward(self, x):

        out = self.relu(self.convtr1(x))
        out = self.relu(self.convtr1(x))

        return out
