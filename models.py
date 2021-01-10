import torch
from torch import nn
from torch.nn.functional import relu, sigmoid


class StegaStampEncoder(nn.Module):
    def __init__(
        self,
        height=32,
        width=32,
        IMAGE_CHANNELS=1,
        secret_size=100,
        return_residual=False,
    ):
        super(StegaStampEncoder, self).__init__()
        self.secret_size = secret_size
        self.IMAGE_CHANNELS = IMAGE_CHANNELS
        self.return_residual = return_residual
        self.secret_dense = nn.Linear(self.secret_size, 16 * 16 * IMAGE_CHANNELS)
        if height == width == 32:
            self.secret_upsample = nn.Upsample(scale_factor=(2, 2))
        elif height == width == 128:
            self.secret_upsample = nn.Upsample(scale_factor=(8, 8))
        elif height == width == 512:
            self.secret_upsample = nn.Upsample(scale_factor=(32, 32))
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(2 * IMAGE_CHANNELS, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)
        self.pad6 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up6 = nn.Conv2d(256, 128, 2, 1)
        self.upsample6 = nn.Upsample(scale_factor=(2, 2))
        self.conv6 = nn.Conv2d(128 + 128, 128, 3, 1, 1)
        self.pad7 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up7 = nn.Conv2d(128, 64, 2, 1)
        self.upsample7 = nn.Upsample(scale_factor=(2, 2))
        self.conv7 = nn.Conv2d(64 + 64, 64, 3, 1, 1)
        self.pad8 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up8 = nn.Conv2d(64, 32, 2, 1)
        self.upsample8 = nn.Upsample(scale_factor=(2, 2))
        self.conv8 = nn.Conv2d(32 + 32, 32, 3, 1, 1)
        self.pad9 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up9 = nn.Conv2d(32, 32, 2, 1)
        self.upsample9 = nn.Upsample(scale_factor=(2, 2))
        self.conv9 = nn.Conv2d(32 + 32 + 2 * IMAGE_CHANNELS, 32, 3, 1, 1)
        self.conv10 = nn.Conv2d(32, 32, 3, 1, 1)
        self.residual = nn.Conv2d(32, IMAGE_CHANNELS, 1)

    def forward(self, secret, image):
        secret = relu(self.secret_dense(secret))
        secret = secret.view((-1, self.IMAGE_CHANNELS, 16, 16))
        secret_enlarged = self.secret_upsample(secret)
        inputs = torch.cat([secret_enlarged, image], dim=1)
        conv1 = relu(self.conv1(inputs))
        conv2 = relu(self.conv2(conv1))
        conv3 = relu(self.conv3(conv2))
        conv4 = relu(self.conv4(conv3))
        conv5 = relu(self.conv5(conv4))
        up6 = relu(self.up6(self.pad6(self.upsample6(conv5))))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = relu(self.conv6(merge6))
        up7 = relu(self.up7(self.pad7(self.upsample7(conv6))))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = relu(self.conv7(merge7))
        up8 = relu(self.up8(self.pad8(self.upsample8(conv7))))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = relu(self.conv8(merge8))
        up9 = relu(self.up9(self.pad9(self.upsample9(conv8))))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = relu(self.conv9(merge9))
        conv10 = relu(self.conv10(conv9))
        residual = self.residual(conv10)
        if not self.return_residual:
            residual = sigmoid(residual)
        return residual


class StegaStampDecoder(nn.Module):
    def __init__(self, height=32, width=32, IMAGE_CHANNELS=1, secret_size=1):
        super(StegaStampDecoder, self).__init__()
        self.height = height
        self.width = width
        self.IMAGE_CHANNELS = IMAGE_CHANNELS
        self.decoder = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, (3, 3), 2, 1),  # 16
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 8
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),  # 4
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 2
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), 2, 1),
            nn.ReLU(),
        )
        self.dense = nn.Sequential(
            nn.Linear(height * width * 128 // 32 // 32, 512),
            nn.ReLU(),
            nn.Linear(512, secret_size),
        )

    def forward(self, image):
        x = self.decoder(image)
        x = x.view(-1, self.height * self.width * 128 // 32 // 32)
        return self.dense(x)


class Discriminator(nn.Module):
    def __init__(self, height, width, IMAGE_CHANNELS):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 8, (3, 3), 2, 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3), 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3), 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, (2, 2), 1),
        )

    def forward(self, image):
        x = self.model(image)
        output = torch.mean(x, dim=[1, 2, 3]).view(-1, 1)
        return output, x
