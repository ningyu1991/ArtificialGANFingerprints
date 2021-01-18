import argparse
import glob
import PIL

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="Directory with images.")
parser.add_argument(
    "--output_dir", type=str, help="Path to save watermarked images to."
)
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
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)


args = parser.parse_args()

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

from time import time
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision import transforms


SECRET_SIZE = args.fingerprint_size

if args.cuda != -1:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.filenames = sorted(self.filenames)
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename)
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.filenames)


def load_decoder():
    global RevealNet

    from models import StegaStampDecoder

    RevealNet = StegaStampDecoder(128, 128, 3, SECRET_SIZE)
    kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
    RevealNet.load_state_dict(torch.load(args.decoder_path, **kwargs))
    RevealNet = RevealNet.to(device)


def load_data():
    global dataset, dataloader
    global IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, SECRET_SIZE

    SECRET_SIZE = args.fingerprint_size

    transform = transforms.Compose(
        [
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


def extract_fingerprints():
    all_fingerprinted_images = []
    all_fingerprints = []

    BATCH_SIZE = 50
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    for images, _ in tqdm(dataloader):
        images = images.to(device)

        fingerprints = RevealNet(images)
        fingerprints = (fingerprints > 0).long()

        all_fingerprinted_images.append(images.detach().cpu())
        all_fingerprints.append(fingerprints.detach().cpu())

    dirname = args.output_dir
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    all_fingerprints = torch.cat(all_fingerprints, dim=0).cpu()
    f = open(os.path.join(args.output_dir, "fingerprints.txt"), "w")
    for idx in range(len(all_fingerprints)):
        fingerprint = all_fingerprints[idx]
        fingerprint_str = "".join(map(str, fingerprint.cpu().long().numpy().tolist()))
        _, filename = os.path.split(dataset.filenames[idx])
        filename = filename.split('.')[0] + ".png"
        f.write(f"{filename} {fingerprint_str}\n")
    f.close()


if __name__ == "__main__":
    load_decoder()
    load_data()
    extract_fingerprints()
