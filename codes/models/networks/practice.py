import functools
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

#
class BicubicUpsample(nn.Module):
    """ A bicubic upsampling class with similar behavior to that in TecoGAN-Tensorflow

        Note that it's different from torch.nn.functional.interpolate and
        matlab's imresize in terms of bicubic kernel and sampling scheme

        Theoretically it can support any scale_factor >= 1, but currently only
        scale_factor = 4 is tested

        References:
            The original paper: http://verona.fi-p.unam.mx/boris/practicas/CubConvInterp.pdf
            https://stackoverflow.com/questions/26823140/imresize-trying-to-understand-the-bicubic-interpolation
    """

    def __init__(self, scale_factor, a=-0.75):
        super(BicubicUpsample, self).__init__()

        # calculate weights
        cubic = torch.FloatTensor([
            [0, a, -2 * a, a],
            [1, 0, -(a + 3), a + 2],
            [0, -a, (2 * a + 3), -(a + 2)],
            [0, 0, a, -a]
        ])  # accord to Eq.(6) in the reference paper
        cubic = cubic.cuda()
        kernels = [
            torch.matmul(cubic, torch.FloatTensor([1, s, s ** 2, s ** 3]))
            for s in [1.0*d/scale_factor for d in range(scale_factor)]
        ]  # s = x - floor(x)

        # register parameters
        self.scale_factor = scale_factor
        self.register_buffer('kernels', torch.stack(kernels))

    def forward(self, input):
        n, c, h, w = input.size()
        s = self.scale_factor

        # pad input (left, right, top, bottom)
        input = F.pad(input, (1, 2, 1, 2), mode='replicate')

        # calculate output (height)
        kernel_h = self.kernels.repeat(c, 1).view(-1, 1, s, 1)
        output = F.conv2d(input, kernel_h, stride=1, padding=0, groups=c)
        output = output.reshape(
            n, c, s, -1, w + 3).permute(0, 1, 3, 2, 4).reshape(n, c, -1, w + 3)

        # calculate output (width)
        kernel_w = self.kernels.repeat(c, 1).view(-1, 1, 1, s)
        output = F.conv2d(output, kernel_w, stride=1, padding=0, groups=c)
        output = output.reshape(
            n, c, s, h * s, -1).permute(0, 1, 3, 4, 2).reshape(n, c, h * s, -1)

        return output

import torch
input = torch.randn([20, 3, 32, 32]).cuda()
scale_factor = 4
a = - 0.75
cubic = torch.FloatTensor([
    [0, a, -2 * a, a],
    [1, 0, -(a + 3), a + 2],
    [0, -a, (2 * a + 3), -(a + 2)],
    [0, 0, a, -a]
])
print(cubic)
s = 4
b = torch.FloatTensor([1, s, s ** 2, s ** 3])
print(b)
# c = torch.matmul(cubic, torch.FloatTensor([1, s, s ** 2, s ** 3]))
# print(c)
#
kernels = [
            torch.matmul(cubic, torch.FloatTensor([1, s, s ** 2, s ** 3]))
            for s in [1.0*d/scale_factor for d in range(scale_factor)]
        ]
print(kernels)
# input = F.pad(input, (1, 2, 1, 2), mode='replicate')
# 20 3 35 35  20 3 67 67
# 20 12  32 35  20 12 66 67
# kernels=torch.Tensor(kernels)
# a = torch.kernels.repeat(3, 1).view(-1, 1, s, 1)
import numpy as np
kernels =  np.array(([0., 1., 0., 0.],[-0.1055,  0.8789,  0.2617, -0.0352],[-0.0938,  0.5938,  0.5938, -0.0938],[-0.0352,  0.2617,  0.8789, -0.1055]))
print(kernels)
kernel_h = np.repeat(kernels, 1.5, axis=0)
print(kernel_h)
# kernel_h = kernel_h.view(-1, 1, s, 1)
# print(kernel_h.shape)
# output = F.conv2d(input, kernel_h, stride=1, padding=0, groups=c)
# a = torch.randn([5, 3, 32, 32]).cuda()
# b = torch.randn([12, 1, 4, 1]).cuda()
# output = F.conv2d(a, b, stride=1, padding=0, groups=3)
# print(output.shape)
# import torch.fft
# def fftshift2d(img, size_psc=128):
#     bs,ch, h, w = img.shape
#     fs11 = img[:,:, h//2:, w//2:]
#     fs12 = img[:,:, h//2:, :w//2]
#     fs21 = img[:,:, :h//2, w//2:]
#     fs22 = img[:,:, :h//2, :w//2]
#     output = torch.cat([torch.cat([fs11, fs21], axis=2), torch.cat([fs12, fs22], axis=2)], axis=3)
#     # output = tf.image.resize_images(output, (size_psc, size_psc), 0)
#     return output
# x = torch.randn([1, 64, 32, 32]).cuda()
# print(x[0][0][0])
# x = torch.fft.fftn(x, dim=(2, 3))
# print(x[0][0][0])
# x = torch.pow(torch.abs(x) + 1e-8, 0.1)  # abs
# print(x[0][0][0])
# x = fftshift2d(x)
# print(x[0][0][0])
# 读取图像
# import cv2 as cv
#
# import numpy as np
#
# from matplotlib import pyplot as plt
# img = cv.imread('/home/yang/project/EGVSR/codes/data/VimeoTecoGAN/GT/000/000.png', 0)
# # 傅里叶变换
# f = torch.fft.fftn(img)
# fshift = torch.fft.fftshift(f)
# res = np.log(np.abs(fshift))
# # 傅里叶逆变换
# ishift = np.fft.ifftshift(fshift)
# iimg = np.fft.ifft2(ishift)
# iimg = np.abs(iimg)
# # 展示结果
# plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
# plt.axis('off')
# plt.subplot(132), plt.imshow(res, 'gray'), plt.title('Fourier Image')
# plt.axis('off')
# plt.subplot(133), plt.imshow(iimg, 'gray'), plt.title('Inverse Fourier Image')
# plt.axis('off')
# plt.show()
