import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .medmnist_info import INFO


class MedMNIST(Dataset):

    flag = ...

    def __init__(self,
                 root,
                 split='train',
                 transform=None,
                 target_transform=None,
                 download=False):
        ''' dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation
    
        '''

        self.info = INFO[self.flag]
        self.root = root

        if download:
            self.download()

        if not os.path.exists(
                os.path.join(self.root, "{}.npz".format(self.flag))):
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        npz_file = np.load(os.path.join(self.root, "{}.npz".format(self.flag)))

        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        if self.split == 'train':
            self.data = npz_file['train_images']
            self.targets = npz_file['train_labels']
            self.targets = np.squeeze(self.targets)
        elif self.split == 'val':
            self.data = npz_file['val_images']
            self.targets = npz_file['val_labels']
            self.targets = np.squeeze(self.targets)
        elif self.split == 'test':
            self.data = npz_file['test_images']
            self.targets = npz_file['test_labels']
            self.targets = np.squeeze(self.targets)
            
        if self.flag == 'octmnist' or self.flag == 'pneumoniamnist':
            new_data = []
            for i in self.data:
                i = np.stack((i,)*3,axis=-1)
                new_data.append(i)
            self.data = np.array(new_data)

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index].astype(int)
        data = Image.fromarray(np.uint8(data))

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        '''Adapted from torchvision.
        '''
        _repr_indent = 4
        head = "Dataset " + self.__class__.__name__

        body = ["Number of datapoints: {}".format(self.__len__())]
        body.append("Root location: {}".format(self.root))
        body.append("Split: {}".format(self.split))
        body.append("Task: {}".format(self.info["task"]))
        body.append("Number of channels: {}".format(self.info["n_channels"]))
        body.append("Meaning of labels: {}".format(self.info["label"]))
        body.append("Number of samples: {}".format(self.info["n_samples"]))
        body.append("Description: {}".format(self.info["description"]))
        body.append("License: {}".format(self.info["license"]))

        lines = [head] + [" " * _repr_indent + line for line in body]
        return '\n'.join(lines)

    def download(self):
        try:
            from torchvision.datasets.utils import download_url
            download_url(url=self.info["url"],
                         root=self.root,
                         filename="{}.npz".format(self.flag),
                         md5=self.info["MD5"])
        except:
            raise RuntimeError('Something went wrong when downloading! ' +
                               'Go to the homepage to download manually. ' +
                               'https://github.com/MedMNIST/MedMNIST')


class PathMNIST(MedMNIST):
    flag = "pathmnist"


class OCTMNIST(MedMNIST):
    flag = "octmnist"


class PneumoniaMNIST(MedMNIST):
    flag = "pneumoniamnist"


class ChestMNIST(MedMNIST):
    flag = "chestmnist"


class DermaMNIST(MedMNIST):
    flag = "dermamnist"


class RetinaMNIST(MedMNIST):
    flag = "retinamnist"


class BreastMNIST(MedMNIST):
    flag = "breastmnist"


class OrganMNISTAxial(MedMNIST):
    flag = "organmnist_axial"


class OrganMNISTCoronal(MedMNIST):
    flag = "organmnist_coronal"


class OrganMNISTSagittal(MedMNIST):
    flag = "organmnist_sagittal"

class TissueMNIST(MedMNIST):
    flag = "tissuemnist"