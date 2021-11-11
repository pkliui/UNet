#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class for UNet network architecture

"""

class UNet(nn.Module):

    def __init__(self):
        super().__init__()
        """
        Initializes the UNet network architecture class
        Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI 2015.

        ---
        Class attributes
        ---
        -
        BN : batch normalization layer
        conv NxN: convolutional layer, kernel size NxN
        ,ax pool NxN: max pooling layer, kernel size NxN
        ReLU: rectified linear unit layer
        -
        input and output image sizes before and after net's layers are shown as comments in code
        --
        encoder layer 0
        --
        self.e0_conv:
            2x conv 3x3, BN, ReLU
        self.e0_pool:
            1x max pool 2x2, stride=2
        self.e0_crop:
            cropped output of self.e0_conv for concatenation in decoder
        self.e0_pool_idx:
            max-pool indices of self.e0_crop for upsampling (max-unpooling) in decoder
        --
        encoder layer 1
        --
        self.e1_conv:
            2x conv 3x3, BN, ReLU
        self.e1_pool:
            1x max pool 2x2, stride=2
        self.e1_crop:
            cropped output of self.e1_conv for concatenation in decoder
        self.e1_pool_idx:
            max-pool indices of self.e1_crop for upsampling (max-unpooling) in decoder
        --
        encoder layer 2
        --
        self.e2_conv:
            2x conv 3x3, BN, ReLU
        self.e2_pool:
            1x max pool 2x2, stride=2
        self.e2_crop:
            cropped output of self.e2_conv for concatenation in decoder
        self.e2_pool_idx:
            max-pool indices of self.e2_crop for upsampling (max-unpooling) in decoder
        --
        encoder layer 3
        --
        self.e3_conv:
            2x conv 3x3, BN, ReLU
        self.e3_pool:
            1x max pool 2x2, stride=2
        self.e3_crop:
            cropped output of self.e3_conv for concatenation in decoder
        self.e3_pool_idx:
            max-pool indices of self.e3_crop for upsampling (max-unpooling) in decoder
        --
        bottleneck
        --
        self.bottleneck_conv:
            2x conv 3x3, BN, ReLU
        --
        decoder layer 3 (numbering in reverse order to match it with encoder)
        --
        self.d3_upsample:
            1x maxunpool 2x2, stride 2
            uses indices provided by self.e3_pool_idx
        self.d3_upconv:
            1x upconv 3x3, padding 1
        self.d3_conv:
            2x conv 3x3, BN, ReLU
        --
        decoder layer 2
        --
        self.d2_upsample:
            1x maxunpool 2x2, stride 2
            uses indices provided by self.e2_pool_idx
        self.d2_upconv:
            1x upconv 3x3, padding 1
        self.d2_conv:
            2x conv 3x3, BN, ReLU
        --
        decoder layer 1
        --
        self.d1_upsample:
            1x maxunpool 2x2, stride 2
            uses indices provided by self.e1_pool_idx
        self.d1_upconv:
            1x upconv 3x3, padding 1
        self.d1_conv:
            2x conv 3x3, BN, ReLU
        --
        decoder layer 0
        --
        self.d0_upsample:
            1x maxunpool 2x2, stride 2
            uses indices provided by self.e0_pool_idx
        self.d0_upconv:
            1x upconv 3x3, padding 1
        self.d0_conv:
            2x conv 3x3, 1x conv 1x1, BN
        """

        # encoder (downsampling)
        ##################
        # encoder layer 0
        #################
        # 3, 572, 572
        self.e0_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            # 64, 570, 570
            nn.Conv2d(64, 64, kernel_size=3),
            # 64, 568, 568
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 64, 568, 568
        self.e0_pool = nn.MaxPool2d(2, stride=2, return_indices=False)
        # 64, 284, 284
        #
        ##################
        # encoder layer 0 - cropping for decoder layer and generating indices for maxunpool in decoder
        #################
        self.e0_crop = nn.Sequential(
            # 64, 568, 568
            torchvision.transforms.CenterCrop(392)
            # 64, 392,392
        )
        self.e0_pool_idx = nn.MaxPool2d(2, stride=2, return_indices=True)
        # 64, 196,196
        #
        #################
        # encoder layer1
        ################
        self.e1_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            # 128, 282, 282
            nn.Conv2d(128, 128, kernel_size=3),
            # 128, 280, 280
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # 128, 280, 280
        self.e1_pool = nn.MaxPool2d(2, stride=2, return_indices=False)
        # 128, 140, 140
        #
        ###################
        # encoder layer 1 - cropping for decoder layer and generating a fake maxpool image tensor to get indices for maxunpool
        ##################
        self.e1_crop = nn.Sequential(
            # 128, 280, 280
            torchvision.transforms.CenterCrop(200)
            # 128, 200, 200
        )
        self.e1_pool_idx = nn.MaxPool2d(2, stride=2, return_indices=True)
        # 128, 100, 100
        #
        ###################
        # encoder layer 2
        ###################
        self.e2_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),
            # 256, 138, 138
            nn.Conv2d(256, 256, kernel_size=3),
            # 256, 136, 136
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # 256, 136, 136
        self.e2_pool = nn.MaxPool2d(2, stride=2, return_indices=False)
        # 256, 68, 68
        #
        #################
        # encoder layer 2 - cropping for decoder layer and generating a fake maxpool image tensor to get indices for maxunpool
        #################
        self.e2_crop = nn.Sequential(
            # 256, 136, 136
            torchvision.transforms.CenterCrop(104)
            # 256, 104, 104
        )
        self.e2_pool_idx = nn.MaxPool2d(2, stride=2, return_indices=True)
        # 256, 52, 52
        #
        ##################
        # encoder layer 3
        #################
        self.e3_conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3),
            # 512, 66, 66
            nn.Conv2d(512, 512, kernel_size=3),
            # 512, 64, 64
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.e3_pool = nn.MaxPool2d(2, stride=2, return_indices=False)
        # 512, 32, 32
        #
        #################
        # encoder layer 3 - cropping for decoder layer and generating a fake maxpool image tensor to get indices for maxunpool
        #################
        self.e3_crop = nn.Sequential(
            # 512, 64, 64
            torchvision.transforms.CenterCrop(56)
            # 512, 56, 56
        )
        self.e3_pool_idx = nn.MaxPool2d(2, stride=2, return_indices=True)
        # 512, 28, 28
        #
        ###
        # bottleneck
        ###
        # 512, 32, 32
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3),
            # 1024, 30, 30
            nn.Conv2d(1024, 1024, kernel_size=3),
            # 1024, 28, 28
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        # 1024, 28, 28

        # decoder (upsampling)
        ###################
        # decoder layer 3
        ###################
        # 1024, 28, 28--> 255
        self.d3_upsample = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.d3_upsample =  nn.Upsample(scale_factor=2)
        # 1024, 56, 56
        self.d3_upconv = nn.Sequential(
            # H_out=stride*(H_in−1)−2×padding+kernel_size+output_padding
            # 56 = 1*(56-1)-2*1+2+1
            # nn.ConvTranspose2d(1024, 512, kernel_size=2, padding=1, output_padding = 1),
            # 56 = 1*(56-1)-2*1+3
            nn.ConvTranspose2d(1024, 512, kernel_size=3, padding=1),
            # 512, 56, 56
            nn.BatchNorm2d(512),
            nn.ReLU()
            # 512, 56, 56
            # 1024, 56, 56 after concatenation w/ corresponding cropped encoder map
        )
        # 1024, 56, 56
        #
        self.d3_conv = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3),
            ## 56- 3 + 1
            # 512, 54 ,54
            nn.Conv2d(512, 512, kernel_size=3),
            # 512, 52, 52
            nn.BatchNorm2d(512),
            nn.ReLU()
            # 512, 52, 52
        )
        #
        ##################
        # decoder layer 2
        ##################
        # 512, 52, 52
        self.d2_upsample = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.d2_upsample =  nn.Upsample(scale_factor=2)
        # 512, 104, 104
        self.d2_upconv = nn.Sequential(
            # nn.ConvTranspose2d(512, 256, kernel_size=2, padding=1, output_padding = 1),
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),
            # 256, 104, 104
            nn.BatchNorm2d(256),
            nn.ReLU()
            # 256, 104, 104
            # 512, 104, 104 after concatenation w/ corresponding cropped encoder map
        )
        #
        self.d2_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3),
            # 256, 102 ,102
            nn.Conv2d(256, 256, kernel_size=3),
            # 256, 100, 100
            nn.BatchNorm2d(256),
            nn.ReLU()
            # 256, 100, 100
        )
        ##################
        # decoder layer 1
        ##################
        # 256, 100, 100
        self.d1_upsample = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.d1_upsample =  nn.Upsample(scale_factor=2)
        # 256, 200, 200
        self.d1_upconv = nn.Sequential(
            # nn.ConvTranspose2d(256, 128, kernel_size=2, padding=1, output_padding = 1),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            # 128, 200, 200
            nn.BatchNorm2d(128),
            nn.ReLU()
            # 128, 200, 200
            # 256, 200, 200 after concatenation w/ corresponding cropped encoder map
        )
        #
        self.d1_conv = nn.Sequential(
            # 256, 200, 200
            nn.Conv2d(256, 128, kernel_size=3),
            # 128, 198, 198
            nn.Conv2d(128, 128, kernel_size=3),
            # 128, 196, 196
            nn.BatchNorm2d(128),
            nn.ReLU()
            # 128, 196, 196
        )
        ###
        # decoder layer 0
        ###
        # 128, 196, 196
        self.d0_upsample = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.d0_upsample =  nn.Upsample(scale_factor=2)
        # 128, 392, 392
        self.d0_upconv = nn.Sequential(
            # nn.ConvTranspose2d(128, 64, kernel_size=2, padding=1, output_padding = 1),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            # 64, 392, 392
            nn.BatchNorm2d(64),
            nn.ReLU()
            # 64, 392, 392
            # 128, 392, 392 after concatenation w/ corresponding cropped encoder map
        )
        #
        self.d0_conv = nn.Sequential(
            # 128, 392, 392
            nn.Conv2d(128, 64, kernel_size=3),
            # 64, 390, 390
            nn.Conv2d(64, 64, kernel_size=3),
            # 64, 388, 388
            nn.Conv2d(64, 1, kernel_size=1),
            # 1, 388, 388
            nn.BatchNorm2d(1),
            # 1, 388, 388
        )

    def forward(self, x):
        """
        implementation of forward propagation in UNet
        ---
        Parameters
        ---
        see parameters of __init__ method
        ---
        Return
        ---
        d0: tensor of floats, size (1,388,388)
        """
        # encoder
        ###################
        # encoder layer 0
        ##################
        #
        # convolutions
        # --> # 1, 572, 572
        e0 = self.e0_conv(x)
        # --> # 64, 568, 568
        assert (e0.shape[1], e0.shape[2], e0.shape[3]) == (
        64, 568, 568), "encoder layer e0 expected shape {}, got{}".format("(64, 568, 568)",
                                                                          (e0.shape[1], e0.shape[2], e0.shape[3]))
        #
        # pooling
        e0_pool = self.e0_pool(e0)
        # --> # 64, 284, 284
        assert (e0_pool.shape[1], e0_pool.shape[2], e0_pool.shape[3]) == (
        64, 284, 284), "encoder layer e0 after pooling expected shape{}".format("(64, 284, 284)")
        #
        # cropping and cropped indices for skip connections
        e0_crop = self.e0_crop(e0)
        # --> # 64, 392,392
        _, idx0_crop = self.e0_pool_idx(e0_crop)  # --> pooling indices from 64, 392,392 cropped map
        # --> # 64, 196,196
        assert (e0_crop.shape[1], e0_crop.shape[2], e0_crop.shape[3]) == (
        64, 392, 392), "encoder layer e0 after cropping expected shape{}".format("(64, 392, 392)")
        assert (idx0_crop.shape[1], idx0_crop.shape[2], idx0_crop.shape[3]) == (
        64, 196, 196), "encoder indices idx0 after cropping expected shape {}, got {}".format("(64, 196, 196)", (
        idx0_crop.shape[1], idx0_crop.shape[2], idx0_crop.shape[3]))
        #
        ##################
        # encoder layer 1
        ##################
        #  convolutions
        # --> # 64, 284, 284
        e1 = self.e1_conv(e0_pool)
        # --> # 128, 280, 280
        assert (e1.shape[1], e1.shape[2], e1.shape[3]) == (128, 280, 280), "encoder layer e1 expected shape {}".format(
            "(128, 280, 280)")
        #
        # pooling
        e1_pool = self.e1_pool(e1)
        # --> # 128, 140, 140
        assert (e1_pool.shape[1], e1_pool.shape[2], e1_pool.shape[3]) == (
        128, 140, 140), "encoder layer e1 after pooling expected shape{}".format("(128, 140, 140)")
        #
        # cropping and cropped indices for skip connections
        e1_crop = self.e1_crop(e1)
        # --> # 128, 200, 200
        _, idx1_crop = self.e1_pool_idx(e1_crop)  # --> pooling indices from 128, 200, 200 cropped map
        # --> # 128, 100, 100
        assert (e1_crop.shape[1], e1_crop.shape[2], e1_crop.shape[3]) == (
        128, 200, 200), "encoder layer e1 after cropping expected shape{}".format("(128, 200, 200)")
        assert (idx1_crop.shape[1], idx1_crop.shape[2], idx1_crop.shape[3]) == (
        128, 100, 100), "encoder indices idx1 after cropping expected shape{}".format("(128, 100, 100)")
        #
        ##################
        # encoder layer 2
        ##################
        #  convolutions
        # --> # 128, 140, 140
        e2 = self.e2_conv(e1_pool)
        # --> # 256, 136, 136
        assert (e2.shape[1], e2.shape[2], e2.shape[3]) == (256, 136, 136), "encoder layer e2 expected shape {}".format(
            "(256, 136, 136)")
        #
        # pooling
        e2_pool = self.e2_pool(e2)
        # --> # 256, 68, 68
        assert (e2_pool.shape[1], e2_pool.shape[2], e2_pool.shape[3]) == (
        256, 68, 68), "encoder layer e2 after pooling expected shape{}".format("(256, 68, 68)")
        #
        # cropping and cropped indices for skip connections
        e2_crop = self.e2_crop(e2)
        # --> # 256, 104, 104
        _, idx2_crop = self.e2_pool_idx(e2_crop)  # --> pooling indices from 256, 104, 104 cropped map
        # --> # 256, 52, 52
        assert (e2_crop.shape[1], e2_crop.shape[2], e2_crop.shape[3]) == (
        256, 104, 104), "encoder layer e2 after cropping expected shape{}".format("(256, 104, 104)")
        assert (idx2_crop.shape[1], idx2_crop.shape[2], idx2_crop.shape[3]) == (
        256, 52, 52), "encoder indices idx2 after cropping expected shape{}".format("(256, 52, 52)")
        #
        ##################
        # encoder layer 3
        ##################
        #  convolutions
        # --> # 256, 68, 68
        e3 = self.e3_conv(e2_pool)
        # --> # 512, 64, 64
        assert (e3.shape[1], e3.shape[2], e3.shape[3]) == (512, 64, 64), "encoder layer e3 expected shape {}".format(
            "(512, 64, 64)")
        #
        # pooling
        e3_pool = self.e3_pool(e3)
        # --> # 512, 32, 32
        assert (e3_pool.shape[1], e3_pool.shape[2], e3_pool.shape[3]) == (
        512, 32, 32), "encoder layer e3 after pooling expected shape{}".format("(512, 32, 32)")
        #
        # cropping and cropped indices for skip connections
        e3_crop = self.e3_crop(e3)
        # --> # 512, 56, 56
        _, idx3_crop = self.e3_pool_idx(e3_crop)  # --> pooling indices from 512, 56, 56 cropped map
        # --> # 512, 28, 28
        assert (e3_crop.shape[1], e3_crop.shape[2], e3_crop.shape[3]) == (
        512, 56, 56), "encoder layer e3 after cropping expected shape{}".format("(512, 56, 56)")
        assert (idx3_crop.shape[1], idx3_crop.shape[2], idx3_crop.shape[3]) == (
        512, 28, 28), "encoder indices idx3 after cropping expected shape{}".format("(512, 28, 28)")
        #
        # bottleneck
        # --> # 512, 32, 32
        b = self.bottleneck_conv(e3_pool)
        # --> # 1024, 28, 28
        assert (b.shape[1], b.shape[2], b.shape[3]) == (1024, 28, 28), "bottleneck expected shape{}".format(
            "(1024, 28, 28)")
        #
        # decoder
        ##################
        # decoder layer 3 (reverse counting order)
        ##################
        #
        # upconvolution
        d3_upconv = self.d3_upconv(b)
        # --> # 512, 28, 28
        assert (d3_upconv.shape[1], d3_upconv.shape[2], d3_upconv.shape[3]) == (
        512, 28, 28), "decoder layer d3 after upconvolution expected shape{}".format("(512, 28, 28)")
        #
        # upsampling  idx3 - 512, 28, 28
        # --> # 512, 28, 28
        d3_upsample = self.d3_upsample(d3_upconv, idx3_crop)
        # d3_upsample = self.d3_upsample(b)
        # --> # 512, 56, 56
        assert (d3_upsample.shape[1], d3_upsample.shape[2], d3_upsample.shape[3]) == (
        512, 56, 56), "decoder layer d3 after upsampling expected shape{}".format("(512, 56, 56)")
        #
        # concatenation
        d3_concat = torch.cat((e3_crop, d3_upsample), dim=1)
        # -->  512,56,56 + 512,56,56 = 1024,56,56
        assert (d3_concat.shape[1], d3_concat.shape[2], d3_concat.shape[3]) == (
        1024, 56, 56), "decoder layer d3 after concatenation expected shape{}".format("(1024,56,56)")
        #
        # convolution
        d3 = self.d3_conv(d3_concat)
        # -->    # 512, 52, 52
        assert (d3.shape[1], d3.shape[2], d3.shape[3]) == (
        512, 52, 52), "decoder layer d3 final expected shape{}".format("(512, 52, 52)")
        #
        ##################
        # decoder layer 2
        ##################
        #
        # upconvolution
        d2_upconv = self.d2_upconv(d3)
        # --> # 256, 52, 52
        assert (d2_upconv.shape[1], d2_upconv.shape[2], d2_upconv.shape[3]) == (
        256, 52, 52), "decoder layer d2 after upconvolution expected shape{}".format("(256, 52, 52)")
        #
        # upsampling - idx2 - # 256, 52, 52
        d2_upsample = self.d2_upsample(d2_upconv, idx2_crop)
        # d2_upsample = self.d2_upsample(d3)
        # --> 256, 104, 104
        assert (d2_upsample.shape[1], d2_upsample.shape[2], d2_upsample.shape[3]) == (
        256, 104, 104), "decoder layer d2 after upsampling expected shape{}".format("(256, 104, 104)")
        #
        # concatenation
        d2_concat = torch.cat((e2_crop, d2_upsample), dim=1)
        # -->  256, 104, 104 + 256, 104, 104 = 512, 104, 104
        assert (d2_concat.shape[1], d2_concat.shape[2], d2_concat.shape[3]) == (
        512, 104, 104), "decoder layer d2 after concatenation expected shape{}".format("(512, 104, 104)")
        #
        # convolution
        d2 = self.d2_conv(d2_concat)
        # 256, 100, 100
        assert (d2.shape[1], d2.shape[2], d2.shape[3]) == (
        256, 100, 100), "decoder layer d2 final expected shape{}".format("(256, 100, 100 )")
        #
        ##################
        # decoder layer 1
        ##################
        #
        # upconvolution
        d1_upconv = self.d1_upconv(d2)
        # --> # 128, 100, 100
        assert (d1_upconv.shape[1], d1_upconv.shape[2], d1_upconv.shape[3]) == (
        128, 100, 100), "decoder layer d1 after upconvolution expected shape{}".format("(128,100,100)")
        #
        # upsampling
        d1_upsample = self.d1_upsample(d1_upconv, idx1_crop)
        # d1_upsample = self.d1_upsample(d2)
        # --> 128, 200, 200
        assert (d1_upsample.shape[1], d1_upsample.shape[2], d1_upsample.shape[3]) == (
        128, 200, 200), "decoder layer d1 after upsampling expected shape{}".format("(128, 200, 200)")
        #
        # concatenation
        d1_concat = torch.cat((e1_crop, d1_upsample), dim=1)
        # -->  128, 200, 200 + 128, 200, 200 = 256, 200, 200
        assert (d1_concat.shape[1], d1_concat.shape[2], d1_concat.shape[3]) == (
        256, 200, 200), "decoder layer d1 after concatenation expected shape{}".format("(256, 200, 200)")
        #
        # convolution
        d1 = self.d1_conv(d1_concat)
        # -->    # 128, 196, 196
        assert (d1.shape[1], d1.shape[2], d1.shape[3]) == (
        128, 196, 196), "decoder layer d1 final expected shape{}".format("(128, 196, 196)")
        #
        ##################
        # decoder layer 0
        ##################
        #
        # upconvolution
        d0_upconv = self.d0_upconv(d1)
        # --> # 64, 196, 196
        assert (d0_upconv.shape[1], d0_upconv.shape[2], d0_upconv.shape[3]) == (
        64, 196, 196), "decoder layer d0 after upconvolution expected shape{}".format("(64, 196, 196)")
        #
        # upsampling
        d0_upsample = self.d0_upsample(d0_upconv, idx0_crop)
        # d0_upsample = self.d0_upsample(d1)
        # --> 64, 392, 392
        assert (d0_upsample.shape[1], d0_upsample.shape[2], d0_upsample.shape[3]) == (
        64, 392, 392), "decoder layer d0 after upsampling expected shape{}".format("(64, 392, 392)")
        #
        # concatenation
        d0_concat = torch.cat((e0_crop, d0_upsample), dim=1)
        # -->  64, 392, 392 + 64, 392, 392 = 128, 392, 392
        assert (d0_concat.shape[1], d0_concat.shape[2], d0_concat.shape[3]) == (
        128, 392, 392), "decoder layer d0 after concatenation expected shape{}".format("(128, 392, 392)")
        #
        # convolution
        d0 = self.d0_conv(d0_concat)
        # -->    # 1,388,388
        assert (d0.shape[1], d0.shape[2], d0.shape[3]) == (
        1, 388, 388), "decoder layer d0 final expected shape{}".format("(1,388,388)")

        # return d0 output
        return d0

    def encoder_layer(self, pow = None):

        # 3, 572, 572
        self.e0_conv = nn.Sequential(
            nn.Conv2d(3, 64 * pow, kernel_size=3),
            # 64, 570, 570
            nn.Conv2d(64 * pow, 64 * pow, kernel_size=3),
            # 64, 568, 568
            nn.BatchNorm2d(64 * pow),
            nn.ReLU()
        )
        # 64, 568, 568
        self.e0_pool = nn.MaxPool2d(2, stride=2, return_indices=False)
        # 64, 284, 284
        #
        ##################
        # encoder layer 0 - cropping for decoder layer and generating indices for maxunpool in decoder
        #################
        self.e0_crop = nn.Sequential(
            # 64, 568, 568
            torchvision.transforms.CenterCrop(392)
            # 64, 392,392
        )
        self.e0_pool_idx = nn.MaxPool2d(2, stride=2, return_indices=True)
        # 64, 196,196
        #