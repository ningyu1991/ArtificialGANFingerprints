import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--fingerprint_size",
    type=int,
    required=True,
    default=100,
    help="Number of bits in the fingerprint.",
)
parser.add_argument(
    "--decoder_path",
    type=str,
    required=True,
    help="Path to trained StegaStamp decoder.",
)
parser.add_argument(
    "--gan_path", type=str, required=True, help="Path to trained GAN generator."
)
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--ganmodel", type=str, default="ProGAN")
parser.add_argument("--scale", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--w_n_digits", type=int, default=5)
parser.add_argument("--noise_weights", action="store_true")
parser.add_argument("--w_std", type=float, default=0.0)
parser.add_argument("--rewatermark", action="store_true")
parser.add_argument("--rewatermark_weight", type=float, default=0.0)
parser.add_argument("--rewatermark_encoderpath", type=str)
parser.add_argument("--rewatermark_seed", type=int)


args = parser.parse_args()

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision import transforms

from tqdm import tqdm


def hypersphere(z, radius=1):
    return z * radius / z.norm(p=2, dim=1, keepdim=True)


def generate_random_fingerprints(fingerprint_size, batch_size=4):
    z = torch.zeros((batch_size, fingerprint_size), dtype=torch.float).random_(0, 2)
    return z


Z_DIM = 100

SECRET_SIZE = args.fingerprint_size

if args.cuda != -1:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def load_encoder_decoder():
    global RevealNet, EncoderNet

    from models import StegaStampDecoder, StegaStampEncoder

    RevealNet = StegaStampDecoder(128, 128, 3, SECRET_SIZE)
    kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
    RevealNet.load_state_dict(torch.load(args.decoder_path, **kwargs))
    RevealNet = RevealNet.to(device)

    if args.rewatermark:
        EncoderNet = StegaStampEncoder(128, 128, 3, SECRET_SIZE)
        kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
        EncoderNet.load_state_dict(torch.load(args.rewatermark_encoderpath, **kwargs))
        EncoderNet = EncoderNet.to(device)


def save_samples():
    global Gs

    Gs = torch.load(args.gan_path)
    batch_size = 64
    z_save = hypersphere(torch.randn(batch_size, 4 * 32, 1, 1))
    z_save = z_save.to(device)

    fake_images = Gs(z_save, args.scale)
    fake_images = fake_images.to(device)
    save_image(fake_images[:1], "original_fake.png", normalize=True)

    args.w_n_digits = 0

    def round_weights(m):
        if type(m) == nn.Conv2d:
            m.weight.data = torch.round(m.weight * 10 ** args.w_n_digits) / (
                10 ** args.w_n_digits
            )
            m.bias.data = torch.round(m.bias * 10 ** args.w_n_digits) / (
                10 ** args.w_n_digits
            )

    Gs = Gs.apply(round_weights)

    fake_images = Gs(z_save, args.scale)
    fake_images = fake_images.to(device)
    fake_images = fake_images.clamp(-1, 1)
    save_image(fake_images[:1], "quantization_fake.png", normalize=True)

    args.w_std = 0.16
    Gs = torch.load(args.gan_path)

    def noise_weights(m):
        if type(m) == nn.Conv2d:
            m.weight.data += torch.zeros_like(m.weight.data).normal_() * args.w_std
            m.bias.data += torch.zeros_like(m.bias.data).normal_() * args.w_std

    Gs = Gs.apply(noise_weights)

    fake_images = Gs(z_save, args.scale)
    fake_images = fake_images.to(device)
    fake_images = fake_images.clamp(-1, 1)
    save_image(fake_images[:1], "noise_fake.png", normalize=True)


def load_G():
    global Gs

    if args.ganmodel == "ProGAN":

        kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
        Gs = torch.load(args.gan_path, **kwargs)
        if args.quantize:

            def round_weights(m):
                if type(m) == nn.Conv2d:
                    m.weight.data = torch.round(m.weight * 10 ** args.w_n_digits) / (
                        10 ** args.w_n_digits
                    )
                    m.bias.data = torch.round(m.bias * 10 ** args.w_n_digits) / (
                        10 ** args.w_n_digits
                    )

            Gs = Gs.apply(round_weights)
        if args.noise_weights:

            def noise_weights(m):
                if type(m) == nn.Conv2d:
                    m.weight.data += (
                        torch.zeros_like(m.weight.data).normal_() * args.w_std
                    )
                    m.bias.data += torch.zeros_like(m.bias.data).normal_() * args.w_std

            Gs = Gs.apply(noise_weights)


def extract_fingerprint():

    batch_size = 64
    if args.ganmodel == "ProGAN":
        batch_size = 64
        z_save = hypersphere(torch.randn(batch_size, 4 * 32, 1, 1))
        z_save = z_save.to(device)

        fake_images = Gs(z_save, args.scale)
        fake_images = fake_images.to(device)
        save_image(fake_images, "fake.png", normalize=True)

        fake_images = (fake_images + 1) / 2.0

    if args.rewatermark:
        torch.manual_seed(args.rewatermark_seed)
        fingerprint = generate_random_fingerprints(SECRET_SIZE, 1)
        fingerprint_batch = fingerprint.view(1, SECRET_SIZE).expand(
            batch_size, SECRET_SIZE
        )
        fingerprint_batch = fingerprint_batch.to(device)

        normalized_fake = fake_images.clamp(-1, 1)
        normalized_fake = (normalized_fake + 1) / 2.0

        fake_images = fake_images * (
            1 - args.rewatermark_weight
        ) + args.rewatermark_weight * (
            2 * EncoderNet(fingerprint_batch, normalized_fake) - 1
        )

    revealed = RevealNet(fake_images)
    rev_fingerprint = (revealed > 0).long()

    batch_size = 64

    if len(revealed.size()) == 2:
        torch.manual_seed(args.seed)
        fingerprint = generate_random_fingerprints(SECRET_SIZE, 1)
        fingerprint_batch = fingerprint.view(1, SECRET_SIZE).expand(
            batch_size, SECRET_SIZE
        )
        fingerprint_batch = fingerprint_batch.to(device)

        fingerprint_batch = fingerprint_batch.long()
        print(
            f"Bitwise accuracy on fingerprinted images: {(fingerprint_batch.detach() == rev_fingerprint).float().mean().item()}"
        )
        print(
            f"Bitwise accuracy on fingerprinted images per example: {(fingerprint_batch.detach() == rev_fingerprint).float().mean(dim=1)}"
        )

        rev_fingerprint_maj_vote = rev_fingerprint.float().mean(dim=0)
        rev_fingerprint_maj_vote = torch.round(rev_fingerprint_maj_vote).long()
        rev_fingerprint_maj_vote = rev_fingerprint_maj_vote.view(1, -1).expand_as(
            rev_fingerprint
        )
        print(
            f"Bitwise accuracy on fingerprinted images correcting errors with majority vote: {(fingerprint_batch.detach() == rev_fingerprint_maj_vote).float().mean().item()}"
        )


if __name__ == "__main__":
    load_encoder_decoder()
    save_samples()
    load_G()
    extract_fingerprint()
