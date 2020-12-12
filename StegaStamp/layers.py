from collections import OrderedDict
import torch
import torch.nn as nn


def conv(nin, nout, kernel_size=3, stride=1, padding=1, layer=nn.Conv2d,
         ws=False, bn=False, pn=False, activ=None, gainWS=2):
    conv = layer(nin, nout, kernel_size, stride=stride, padding=padding, bias=False if bn else True)
    layers = OrderedDict()

    if ws:
        layers['ws'] = WScaleLayer(conv, gain=gainWS)

    layers['conv'] = conv

    if bn:
        layers['bn'] = nn.BatchNorm2d(nout)
    if activ:
        if activ == nn.PReLU:
            # to avoid sharing the same parameter, activ must be set to nn.PReLU (without '()') and initialized here
            layers['activ'] = activ(num_parameters=1)
        else:
            layers['activ'] = activ
    if pn:
        layers['pn'] = PixelNormLayer()
    return nn.Sequential(layers)


class PixelNormLayer(nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

    def __repr__(self):
        return self.__class__.__name__


class WScaleLayer(nn.Module):
    def __init__(self, incoming, gain=2):
        super(WScaleLayer, self).__init__()

        self.gain = gain
        self.scale = (self.gain / incoming.weight[0].numel()) ** 0.5

    def forward(self, input):
        return input * self.scale

    def __repr__(self):
        return '{}(gain={})'.format(self.__class__.__name__, self.gain)
