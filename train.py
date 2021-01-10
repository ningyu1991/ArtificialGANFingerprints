import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, required=True, help="Directory with image dataset."
)
parser.add_argument(
    "--output_dir", type=str, required=True, help="Directory to save results to."
)
parser.add_argument(
    "--fingerprint_size",
    type=int,
    default=100,
    required=True,
    help="Number of bits in the fingerprint.",
)
parser.add_argument("--dataset", type=str, required=True, help="CelebA or LSUN.")
parser.add_argument(
    "--num_epochs", type=int, default=20, help="Number of training epochs."
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
parser.add_argument("--cuda", type=str, default=0)

parser.add_argument(
    "--l2_loss_await",
    help="Train without L2 loss for the first x iterations",
    type=int,
    default=1000,
)
parser.add_argument("--l2_loss_scale", type=float, default=10, help="L2 loss weight.")
parser.add_argument(
    "--l2_loss_ramp",
    type=int,
    default=3000,
    help="Linearly increase L2 loss weight over x iterations.",
)

parser.add_argument("--BCE_loss_scale", type=float, default=1, help="BCE loss weight.")

args = parser.parse_args()


import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
from datetime import datetime

from tqdm import tqdm
from time import time
import os
import models
from os.path import join

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from torch.optim import Adam

writer = SummaryWriter()

LOGS_PATH = os.path.join(args.output_dir, "logs")
CHECKPOINTS_PATH = os.path.join(args.output_dir, "checkpoints")
SAVED_IMAGES = os.path.join(args.output_dir, "./saved_images")

if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)
if not os.path.exists(CHECKPOINTS_PATH):
    os.makedirs(CHECKPOINTS_PATH)
if not os.path.exists(SAVED_IMAGES):
    os.makedirs(SAVED_IMAGES)


def generate_random_fingerprints(fingerprint_size, batch_size=4, size=(400, 400)):
    z = torch.zeros((batch_size, fingerprint_size), dtype=torch.float).random_(0, 2)
    return z


plot_points = (
    list(range(0, 1000, 100))
    + list(range(1000, 3000, 200))
    + list(range(3000, 100000, 1000))
)


def main():
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H:%M:%S")
    EXP_NAME = f"stegastamp_{args.dataset}_{args.fingerprint_size}_{dt_string}"

    device = torch.device("cuda")

    if args.dataset == "CelebA":
        from celeba_dataset import CelebA

        # https://github.com/andersbll/autoencoding_beyond_pixels/blob/24aa0f20f1a73a3886551e065bbda818ad139ac2/dataset/celeba.py#L40
        transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )

        print("Reading data...")
        image_folder = ImageFolder(args.data_dir, transform=transform)
        data = CelebA(train=True, ImageFolderObject=image_folder)
        print("Finished reading.")
        IMAGE_HEIGHT = 128
        IMAGE_WIDTH = 128
        IMAGE_CHANNELS = 3

    elif args.dataset == "LSUN":
        transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )

        s = time()
        print("loading data...")
        data = ImageFolder(args.data_dir, transform=transform)
        print("loading took {}".format(time() - s))

        IMAGE_HEIGHT = args.resolution
        IMAGE_WIDTH = args.resolution
        IMAGE_CHANNELS = 3
    else:
        raise ValueError(
            f"Unrecognized dataset option {args.dataset}. Expected CelebA or LSUN."
        )

    encoder = models.StegaStampEncoder(
        secret_size=args.fingerprint_size,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        IMAGE_CHANNELS=IMAGE_CHANNELS,
        return_residual=False,
    )
    decoder = models.StegaStampDecoder(
        secret_size=args.fingerprint_size,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        IMAGE_CHANNELS=IMAGE_CHANNELS,
    )
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    decoder_encoder_optim = Adam(
        params=list(decoder.parameters()) + list(encoder.parameters()), lr=args.lr
    )

    global_step = 0

    for i_epoch in range(args.num_epochs):
        dataloader = DataLoader(
            data, batch_size=args.batch_size, shuffle=True, num_workers=16
        )
        for images, _ in tqdm(dataloader):
            global_step += 1

            batch_size = min(args.batch_size, images.size(0))
            fingerprints = generate_random_fingerprints(
                args.fingerprint_size, batch_size, (IMAGE_HEIGHT, IMAGE_WIDTH)
            )

            l2_loss_scale = min(
                max(
                    0,
                    args.l2_loss_scale
                    * (global_step - args.l2_loss_await)
                    / args.l2_loss_ramp,
                ),
                args.l2_loss_scale,
            )
            BCE_loss_scale = args.BCE_loss_scale

            clean_images = images.to(device)
            fingerprints = fingerprints.to(device)

            fingerprinted_images = encoder(fingerprints, clean_images)
            residual = fingerprinted_images - clean_images

            decoder_output = decoder(fingerprinted_images)

            criterion = nn.MSELoss()
            l2_loss = criterion(fingerprinted_images, clean_images)

            criterion = nn.BCEWithLogitsLoss()
            BCE_loss = criterion(decoder_output.view(-1), fingerprints.view(-1))

            loss = l2_loss_scale * l2_loss + BCE_loss_scale * BCE_loss

            encoder.zero_grad()
            decoder.zero_grad()

            loss.backward()
            decoder_encoder_optim.step()

            fingerprints_predicted = (decoder_output > 0).float()
            bitwise_accuracy = 1.0 - torch.mean(
                torch.abs(fingerprints - fingerprints_predicted)
            )

            # Logging
            if global_step in plot_points:
                writer.add_scalar("bitwise_accuracy", bitwise_accuracy, global_step),
                print("Bitwise accuracy {}".format(bitwise_accuracy))
                writer.add_scalar("loss", loss, global_step),
                writer.add_scalar("BCE_loss", BCE_loss, global_step),
                writer.add_scalars(
                    "clean_statistics",
                    {"min": clean_images.min(), "max": clean_images.max()},
                    global_step,
                ),
                writer.add_scalars(
                    "with_fingerprint_statistics",
                    {
                        "min": fingerprinted_images.min(),
                        "max": fingerprinted_images.max(),
                    },
                    global_step,
                ),
                writer.add_scalars(
                    "residual_statistics",
                    {
                        "min": residual.min(),
                        "max": residual.max(),
                        "mean_abs": residual.abs().mean(),
                    },
                    global_step,
                ),
                print(
                    "residual_statistics: {}".format(
                        {
                            "min": residual.min(),
                            "max": residual.max(),
                            "mean_abs": residual.abs().mean(),
                        }
                    )
                )
                writer.add_image(
                    "clean_image", make_grid(clean_images, normalize=True), global_step
                )
                writer.add_image(
                    "residual",
                    make_grid(residual, normalize=True, scale_each=True),
                    global_step,
                )
                writer.add_image(
                    "image_with_fingerprint",
                    make_grid(fingerprinted_images, normalize=True),
                    global_step,
                )
                save_image(
                    fingerprinted_images,
                    SAVED_IMAGES + "/{}.png".format(global_step),
                    normalize=True,
                )

                writer.add_scalar(
                    "loss_scales/l2_loss_scale", l2_loss_scale, global_step
                )
                writer.add_scalar(
                    "loss_scales/BCE_loss_scale",
                    BCE_loss_scale,
                    global_step,
                )

            # checkpointing
            if global_step % 5000 == 0:
                torch.save(
                    decoder_encoder_optim.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_optim.pth"),
                )
                torch.save(
                    encoder.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_encoder.pth"),
                )
                torch.save(
                    decoder.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_decoder.pth"),
                )
                torch.save(
                    decoder.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_decoder.pth"),
                )
                f = open(join(CHECKPOINTS_PATH, EXP_NAME + "_variables.txt"), "w")
                f.write(str(global_step))
                f.close()

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


if __name__ == "__main__":
    main()
