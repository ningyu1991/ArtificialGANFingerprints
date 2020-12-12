from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms

class CelebA(Dataset):
    def __init__(self, train=True, ImageFolderObject=None, test_size=None, convert_to_greyscale=False, female_only=False, male_only=False):
        super(CelebA, self).__init__()
        self.train = train
        if ImageFolderObject is None:
            if convert_to_greyscale:
                transform = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.CenterCrop(148),#https://github.com/andersbll/autoencoding_beyond_pixels/blob/24aa0f20f1a73a3886551e065bbda818ad139ac2/dataset/celeba.py#L40
                    transforms.Resize(128),
                    transforms.ToTensor(),
                    ])
            else: 
                transform = transforms.Compose([
                    transforms.CenterCrop(148),#https://github.com/andersbll/autoencoding_beyond_pixels/blob/24aa0f20f1a73a3886551e065bbda818ad139ac2/dataset/celeba.py#L40
                    transforms.Resize(128),
                    transforms.ToTensor(),
                    ])
            raise NotImplementedError("Define CelebA path in celeba_dataset.py line 24.")
            #self.data = ImageFolder('CelebA/path', transform=transform)
        else:
            self.data = ImageFolderObject

        self.female_only = female_only
        self.male_only = male_only
        assert (not male_only or not female_only)
        if female_only or male_only:
            import pandas
            attributes = pandas.read_csv('~/list_attr_celeba.txt',delim_whitespace=True) 
            self.female_indices = attributes.index[attributes['Male'] == -1].tolist()
            self.male_indices = attributes.index[attributes['Male'] == 1].tolist()
     
        if female_only:
            pass
            self.train_size = len(self.female_indices)
            self.test_size = 0
        elif male_only:
            pass
            self.train_size = len(self.male_indices)
            self.test_size = 0
        else:
            if test_size is not None:
                self.train_size = len(self.data) - test_size
                self.test_size = test_size
            else:
                self.train_size = int(len(self.data)*0.8)
                self.test_size = len(self.data) - self.train_size

    def __getitem__(self, idx):
#        return self.data[idx]
        if self.female_only:
            return self.data[self.female_indices[idx]]
        if self.male_only:
            return self.data[self.male_indices[idx]]
        if self.train:
            return self.data[idx]
        else:
            return self.data[-idx]

    def __len__(self):
        if self.train:
            return self.train_size
        else:
            return self.test_size
#        return len(self.data)
