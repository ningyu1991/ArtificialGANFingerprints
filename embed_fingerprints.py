import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="LSUN or CelebA.")
parser.add_argument(
    "--encoder_path", type=str, help="Path to trained StegaStamp encoder."
)
parser.add_argument(
    "--decoder_path", type=str, help="Path to trained StegaStamp decoder."
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
parser.add_argument("--batchsize", type=int, default=50)
parser.add_argument("--check", action="store_true")
parser.add_argument("--cuda", type=str, default="0")
parser.add_argument("--changepercent", type=float, default=100.0)
parser.add_argument("--cleanandmarked", type=bool, default=False)
parser.add_argument("--mark_n_images", type=int, default=0)
parser.add_argument("--mark_only_class", type=int, default=-1)
parser.add_argument("--mark_testset", type=bool, default=False)
parser.add_argument("--mark_only_male", type=bool, default=False)
parser.add_argument("--drop_female", type=bool, default=False)
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_size", type=int, default=50000)

args = parser.parse_args()

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

from time import time
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder


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


def load_data():
    global train_data, test_data, train_dataloader, test_dataloader, val_data, val_dataloader
    global IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, SECRET_SIZE
    global attributes

    SECRET_SIZE = args.fingerprint_size

    if args.dataset == "LSUN":
        from torchvision.datasets import ImageFolder

        transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )

        s = time()
        print("loading data...")
        train_data = ImageFolder(
            os.path.join(args.data_dir, "train"), transform=transform
        )
        test_data = ImageFolder(
            os.path.join(args.data_dir, "test"), transform=transform
        )
        print("loading took {}".format(time() - s))
        train_dataloader = DataLoader(
            train_data, batch_size=args.batchsize, shuffle=False, num_workers=16
        )
        test_dataloader = DataLoader(
            test_data, batch_size=args.batchsize, shuffle=False, num_workers=16
        )

        IMAGE_HEIGHT = 128
        IMAGE_WIDTH = 128
        IMAGE_CHANNELS = 3
    elif args.dataset == "CelebA":

        from celeba_dataset import CelebA
        from torchvision.datasets import ImageFolder

        # https://github.com/andersbll/autoencoding_beyond_pixels/blob/24aa0f20f1a73a3886551e065bbda818ad139ac2/dataset/celeba.py#L40
        transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )
        print(f"image_folder = ImageFolder('{args.data_dir}', transform=transform)...")
        image_folder = ImageFolder(args.data_dir, transform=transform)
        print("finished")

        print("train_data = CelebA(train=True)...")
        train_data = CelebA(train=True, ImageFolderObject=image_folder)
        print("finished")
        print("train_data = CelebA(train=False)...")
        test_data = CelebA(train=False, ImageFolderObject=image_folder, test_size=50000)
        print("finished")
        train_dataloader = DataLoader(
            train_data, batch_size=args.batchsize, shuffle=False, num_workers=16
        )
        test_dataloader = DataLoader(
            test_data, batch_size=args.batchsize, shuffle=False, num_workers=16
        )

        IMAGE_CHANNELS = 3
        IMAGE_HEIGHT = 128
        IMAGE_WIDTH = 128
    else:
        raise ValueError(
            f"Unrecognized dataset option {args.dataset}. Expected CelebA or LSUN."
        )


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
    RevealNet.load_state_dict(torch.load(args.decoder_path), **kwargs)
    HideNet.load_state_dict(torch.load(args.encoder_path, **kwargs))

    HideNet = HideNet.to(device)
    RevealNet = RevealNet.to(device)


def check():
    global attributes
    torch.manual_seed(args.seed)
    fingerprint = generate_random_fingerprints(SECRET_SIZE, 1)
    fingerprint_batch = fingerprint.view(1, SECRET_SIZE).expand(
        args.batchsize, SECRET_SIZE
    )
    fingerprint_batch = fingerprint_batch.to(device)

    data, _ = next(iter(test_dataloader))
    data = data.to(device)

    container_img = HideNet(fingerprint_batch, data)

    if args.mark_only_male:
        for j in range(data.size(0)):
            if attributes["Male"].iloc[j] != 1:
                container_img[j] = data[j]

    rev_fingerprint_img = RevealNet(container_img)
    rev_without_fingerprint_img = RevealNet(data)

    save_image(data, "clean.png")
    save_image(
        container_img, "with_fingerprint_{}_bits.png".format(args.fingerprint_size)
    )
    save_image(torch.abs(container_img - data), "diff.png", normalize=True)

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
    data_with_fingerprint = []

    print("started one epoch")
    torch.manual_seed(args.seed)
    fingerprint = generate_random_fingerprints(SECRET_SIZE, 1)
    fingerprint_batch = fingerprint.view(1, SECRET_SIZE).expand(
        args.batchsize, SECRET_SIZE
    )
    fingerprint_batch = fingerprint_batch.to(device)

    if not args.mark_testset:
        dataloader = train_dataloader
        dataset = train_data
    else:
        dataloader = test_dataloader
        dataset = test_data

    for i, (data, y) in tqdm(enumerate(dataloader, 0)):
        data = data.to(device)
        if i * args.batchsize > args.output_size:
            break
        if (
            not args.mark_n_images
            and i * args.batchsize * 100.0 / len(dataset) < args.changepercent
        ):
            if data.size(0) < args.batchsize:
                fingerprint_batch = fingerprint_batch[: data.size(0)]

            container_img = HideNet(fingerprint_batch, data)

            for j in range(data.size(0)):
                if args.mark_only_class != -1 and y[j].item() != args.mark_only_class:
                    container_img[j] = data[j]

            if args.mark_only_male:
                for j in range(data.size(0)):
                    if attributes["Male"].iloc[i * args.batchsize + j] != 1:
                        container_img[j] = data[j]

            if args.drop_female:
                male_indices = []
                for j in range(data.size(0)):
                    if attributes["Male"].iloc[i * args.batchsize + j] == 1:
                        male_indices.append(j)
                # from pdb import set_trace; set_trace()
                container_img = container_img[male_indices]
                save_image(container_img, "female_dropped.png")
                alkdjf

            data_with_fingerprint.append(container_img.detach().cpu())
            if args.cleanandmarked:
                data_with_fingerprint.append(data.cpu())
        else:
            if i == 0 and args.mark_n_images:
                data = data[args.mark_n_images :]
            data_with_fingerprint.append(data.cpu())

    subset = "train" if not args.mark_testset else "test"
    dirname = os.path.join(args.output_dir, subset)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    torch.save(
        torch.cat(data_with_fingerprint, dim=0).cpu(), os.path.join(dirname, "data.pth")
    )


def main():

    load_data()
    load_models()

    if args.check:
        check()
    else:
        putmark()


if __name__ == "__main__":
    main()
