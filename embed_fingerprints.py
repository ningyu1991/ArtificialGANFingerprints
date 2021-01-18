import argparse
import os
import glob
import PIL

parser = argparse.ArgumentParser()
parser.add_argument(
    "--use_celeba_preprocessing",
    action="store_true",
    help="Use CelebA specific preprocessing when loading the images.",
)
parser.add_argument(
    "--encoder_path", type=str, help="Path to trained StegaStamp encoder."
)
parser.add_argument("--data_dir", type=str, help="Directory with images.")
parser.add_argument(
    "--output_dir", type=str, help="Path to save watermarked images to."
)
parser.add_argument(
    "--fingerprint_size",
    type=int,
    default=100,
    required=True,
    help="Number of bits in the fingerprint.",
)
parser.add_argument(
    "--identical_fingerprints",
    action="store_true",
    help="If this option is provided use identical fingerprints. Otherwise sample arbitrary fingerprints.",
)
parser.add_argument(
    "--check", action="store_true", help="Validate fingerprint detection accuracy."
)
parser.add_argument(
    "--decoder_path",
    type=str,
    help="Provide trained StegaStamp decoder to verify fingerprint detection accuracy.",
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed to sample fingerprints."
)
parser.add_argument("--cuda", type=int, default=0)


args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

BATCH_SIZE = 50


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

from time import time
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image


def generate_random_fingerprints(fingerprint_size, batch_size=4):
    z = torch.zeros((batch_size, fingerprint_size), dtype=torch.float).random_(0, 2)
    return z


uniform_rv = torch.distributions.uniform.Uniform(
    torch.tensor([0.0]), torch.tensor([1.0])
)

if int(args.cuda) == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename)
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.filenames)


def load_data():
    global dataset, dataloader
    global IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, SECRET_SIZE

    SECRET_SIZE = args.fingerprint_size

    if args.use_celeba_preprocessing:
        transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )
    else:

        transform = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
            ]
        )

    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")

    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    IMAGE_CHANNELS = 3


def load_models():
    global HideNet, RevealNet

    from models import StegaStampEncoder, StegaStampDecoder

    HideNet = StegaStampEncoder(
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_CHANNELS,
        secret_size=SECRET_SIZE,
        return_residual=False,
    )
    RevealNet = StegaStampDecoder(
        IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, secret_size=SECRET_SIZE
    )

    kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
    if args.check:
        RevealNet.load_state_dict(torch.load(args.decoder_path), **kwargs)
    HideNet.load_state_dict(torch.load(args.encoder_path, **kwargs))

    HideNet = HideNet.to(device)
    RevealNet = RevealNet.to(device)


def check():
    torch.manual_seed(args.seed)
    fingerprint = generate_random_fingerprints(SECRET_SIZE, 1)
    fingerprint_batch = fingerprint.view(1, SECRET_SIZE).expand(BATCH_SIZE, SECRET_SIZE)
    fingerprint_batch = fingerprint_batch.to(device)

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16
    )
    data, _ = next(iter(dataloader))
    data = data.to(device)

    container_img = HideNet(fingerprint_batch, data)

    rev_fingerprint_img = RevealNet(container_img)
    rev_without_fingerprint_img = RevealNet(data)

    save_image(data, os.path.join(args.output_dir, "test_clean_image.png"))
    save_image(
        container_img, os.path.join(args.output_dir, "test_fingerprinted_image.png")
    )
    save_image(
        torch.abs(container_img - data),
        os.path.join(args.output_dir, "test_residual.png"),
        normalize=True,
    )

    rev_fingerprint = (rev_fingerprint_img > 0).long()
    rev_without_fingerprint = (rev_without_fingerprint_img > 0).long()
    fingerprint_batch = fingerprint_batch.long()
    print(
        f"Bitwise accuracy on fingerprinted images per example: {(fingerprint_batch.detach() == rev_fingerprint).float().mean(dim=1)}"
    )
    print(
        f"Bitwise accuracy on fingerprinted images: {(fingerprint_batch.detach() == rev_fingerprint).float().mean().item()}"
    )
    print(
        f"Bitwise accuracy on non-fingerprinted images: {(fingerprint_batch.detach() == rev_without_fingerprint).float().mean().item()}"
    )
    # save_image(fingerprint_batch.detach(), 'fingerprint.png')
    # save_image(rev_fingerprint, 'revealed.png')


def putmark():
    all_fingerprinted_images = []
    all_fingerprints = []

    print("Fingerprinting the images...")
    torch.manual_seed(args.seed)

    # generate identical fingerprints
    fingerprints = generate_random_fingerprints(SECRET_SIZE, 1)
    fingerprints = fingerprints.view(1, SECRET_SIZE).expand(BATCH_SIZE, SECRET_SIZE)
    fingerprints = fingerprints.to(device)

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16
    )

    torch.manual_seed(args.seed)

    for images, _ in tqdm(dataloader):

        # generate arbitrary fingerprints
        if not args.identical_fingerprints:
            fingerprints = generate_random_fingerprints(SECRET_SIZE, BATCH_SIZE)
            fingerprints = fingerprints.view(BATCH_SIZE, SECRET_SIZE)
            fingerprints = fingerprints.to(device)

        images = images.to(device)

        fingerprinted_images = HideNet(fingerprints[: images.size(0)], images)
        all_fingerprinted_images.append(fingerprinted_images.detach().cpu())
        all_fingerprints.append(fingerprints[: images.size(0)].detach().cpu())

    dirname = args.output_dir
    if not os.path.exists(os.path.join(dirname, "fingerprinted_images")):
        os.makedirs(os.path.join(dirname, "fingerprinted_images"))

    all_fingerprinted_images = torch.cat(all_fingerprinted_images, dim=0).cpu()
    all_fingerprints = torch.cat(all_fingerprints, dim=0).cpu()
    f = open(os.path.join(args.output_dir, "fingerprints.txt"), "w")
    for idx in range(len(all_fingerprinted_images)):
        image = all_fingerprinted_images[idx]
        fingerprint = all_fingerprints[idx]
        _, filename = os.path.split(dataset.filenames[idx])
        save_image(
            image,
            os.path.join(args.output_dir, "fingerprinted_images", f"{filename}"),
            padding=0,
        )
        fingerprint_str = "".join(map(str, fingerprint.cpu().long().numpy().tolist()))
        f.write(f"{filename} {fingerprint_str}\n")
    f.close()


#     torch.save(
#         torch.cat(data_with_fingerprint, dim=0).cpu(), os.path.join(dirname, "data.pth")
#     )


def main():

    load_data()
    load_models()

    if args.check:
        check()
    else:
        putmark()


if __name__ == "__main__":
    main()
