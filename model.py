from math import ceil

import torch.nn.functional as F

from layers import *


class Generator(nn.Module):
    def __init__(
        self,
        max_res=8,
        nch=16,
        nc=3,
        bn=False,
        ws=False,
        pn=False,
        activ=nn.LeakyReLU(0.2),
    ):
        super(Generator, self).__init__()
        # resolution of output as 4 * 2^max_res: 0 -> 4x4, 1 -> 8x8, ..., 8 -> 1024x1024
        self.max_res = max_res

        # output convolutions
        self.toRGBs = nn.ModuleList()
        for i in range(self.max_res + 1):
            # max of nch * 32 feature maps as in the original article (with nch=16, 512 feature maps at max)
            self.toRGBs.append(
                conv(
                    int(nch * 2 ** (8 - max(3, i))),
                    nc,
                    kernel_size=1,
                    padding=0,
                    ws=ws,
                    activ=None,
                    gainWS=1,
                )
            )

        # convolutional blocks
        self.blocks = nn.ModuleList()
        # first block, always present
        self.blocks.append(
            nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv0",
                            conv(
                                nch * 32,
                                nch * 32,
                                kernel_size=4,
                                padding=3,
                                bn=bn,
                                ws=ws,
                                pn=pn,
                                activ=activ,
                            ),
                        ),
                        (
                            "conv1",
                            conv(nch * 32, nch * 32, bn=bn, ws=ws, pn=pn, activ=activ),
                        ),
                    ]
                )
            )
        )
        for i in range(self.max_res):
            nin = int(nch * 2 ** (8 - max(3, i)))
            nout = int(nch * 2 ** (8 - max(3, i + 1)))
            self.blocks.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv0",
                                conv(nin, nout, bn=bn, ws=ws, pn=pn, activ=activ),
                            ),
                            (
                                "conv1",
                                conv(nout, nout, bn=bn, ws=ws, pn=pn, activ=activ),
                            ),
                        ]
                    )
                )
            )

        self.pn = None
        if pn:
            self.pn = PixelNormLayer()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 1) if ws else nn.init.kaiming_normal_(
                    m.weight
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input, x=None):
        # value driving the number of layers used in generation
        if x is None:
            progress = self.max_res
        else:
            progress = min(x, self.max_res)

        alpha = progress - int(progress)

        norm_input = self.pn(input) if self.pn else input

        # generating image of size corresponding to progress
        # Example : for progress going from 0 + epsilon to 1 excluded :
        # the output will be of size 8x8 as sum of 4x4 upsampled and output of convolution
        y1 = self.blocks[0](norm_input)
        y0 = y1

        for i in range(1, int(ceil(progress) + 1)):
            y1 = F.upsample(y1, scale_factor=2)
            y0 = y1
            y1 = self.blocks[i](y0)

        # converting to RGB
        y = self.toRGBs[int(ceil(progress))](y1)

        # adding upsampled image from previous layer if transitioning, i.e. when progress is not int
        if progress % 1 != 0:
            y0 = self.toRGBs[int(progress)](y0)
            y = alpha * y + (1 - alpha) * y0

        return y


class Discriminator(nn.Module):
    def __init__(
        self, max_res=8, nch=16, nc=3, bn=False, ws=False, activ=nn.LeakyReLU(0.2)
    ):
        super(Discriminator, self).__init__()
        # resolution of output as 4 * 2^maxRes: 0 -> 4x4, 1 -> 8x8, ..., 8 -> 1024x1024
        self.max_res = max_res

        # input convolutions
        self.fromRGBs = nn.ModuleList()
        for i in range(self.max_res + 1):
            self.fromRGBs.append(
                conv(
                    nc,
                    int(nch * 2 ** (8 - max(3, i))),
                    kernel_size=1,
                    padding=0,
                    bn=bn,
                    ws=ws,
                    activ=activ,
                )
            )

        # convolutional blocks
        self.blocks = nn.ModuleList()
        # last block, always present
        self.blocks.append(
            nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv_std",
                            conv(nch * 32 + 1, nch * 32, bn=bn, ws=ws, activ=activ),
                        ),
                        (
                            "conv_pool",
                            conv(
                                nch * 32,
                                nch * 32,
                                kernel_size=4,
                                padding=0,
                                bn=bn,
                                ws=ws,
                                activ=activ,
                            ),
                        ),
                        (
                            "conv_class",
                            conv(
                                nch * 32,
                                1,
                                kernel_size=1,
                                padding=0,
                                ws=ws,
                                gainWS=1,
                                activ=None,
                            ),
                        ),
                    ]
                )
            )
        )
        for i in range(self.max_res):
            nin = int(nch * 2 ** (8 - max(3, i + 1)))
            nout = int(nch * 2 ** (8 - max(3, i)))
            self.blocks.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("conv0", conv(nin, nin, bn=bn, ws=ws, activ=activ)),
                            ("conv1", conv(nin, nout, bn=bn, ws=ws, activ=activ)),
                        ]
                    )
                )
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 1) if ws else nn.init.kaiming_normal_(
                    m.weight
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def minibatchstd(self, input):
        # must add 1e-8 in std for stability
        return (input.var(dim=0) + 1e-8).sqrt().mean().view(1, 1, 1, 1)

    def forward(self, input, x=None):
        if x is None:
            progress = self.max_res
        else:
            progress = min(x, self.max_res)

        alpha = progress - int(progress)

        y0 = self.fromRGBs[int(ceil(progress))](input)

        if progress % 1 != 0:
            y1 = F.avg_pool2d(input, kernel_size=2, stride=2)
            y1 = self.fromRGBs[int(progress)](y1)
            y0 = self.blocks[int(ceil(progress))](y0)
            y0 = alpha * F.avg_pool2d(y0, kernel_size=2, stride=2) + (1 - alpha) * y1

        for i in range(int(progress), 0, -1):
            y0 = self.blocks[i](y0)
            y0 = F.avg_pool2d(y0, kernel_size=2, stride=2)

        y = self.blocks[0](
            torch.cat(
                (y0, self.minibatchstd(y0).expand_as(y0[:, 0].unsqueeze(1))), dim=1
            )
        )

        return y.squeeze()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def param_number(net):
        n = 0
        for par in net.parameters():
            n += par.numel()
        return n

    # test in original configuration
    nch = 16
    G = Generator(nch=nch, ws=True, pn=True).to(device)
    print(G)
    D = Discriminator(nch=nch, ws=True).to(device)
    print(D)
    z = torch.randn(4, nch * 32, 1, 1, device=device)

    with torch.no_grad():
        print("##### Testing Generator #####")
        print("Generator has {} parameters".format(param_number(G)))
        for i in range((G.max_res + 1) * 2):
            print(i / 2, " -> ", G(z, i / 2).size())
        print("##### Testing Discriminator #####")
        print("Generator has {} parameters".format(param_number(D)))
        for i in range((G.max_res + 1) * 2):
            print(i / 2, " -> ", D(G(z, i / 2), i / 2).size())
