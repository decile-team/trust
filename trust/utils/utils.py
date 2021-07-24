import torch
from torch.utils.data import Dataset

class SubsetWithTargets(Dataset):
    """
    Provides a convenience torch.utils.data.Dataset subclass that allows one to 
    access a targets field while creating subsets of the dataset.

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        The dataset from which to pull a subset
    indices: sequence
        A sequence of indices of the passed dataset from which to select a subset
    labels: sequence
        A sequence of labels for each of the elements drawn for the subset. This 
        sequence should be the same length as indices.
    """
    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels.type(torch.long)
        
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)

class SubsetWithTargetsSingleChannel(Dataset):
    """
    Provides a convenience torch.utils.data.Dataset subclass that allows one to 
    access a targets field while creating subsets of the dataset. Additionally, 
    the single-channel images from the wrapped dataset are expanded to 
    three channels for compatibility with three-chanel model input. 

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        The dataset from which to pull a subset
    indices: sequence
        A sequence of indices of the passed dataset from which to select a subset
    labels: sequence
        A sequence of labels for each of the elements drawn for the subset. This 
        sequence should be the same length as indices.
    """
    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels.type(torch.long)
        
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        image = torch.repeat_interleave(image, 3, 0)
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)


class ConcatWithTargets(Dataset):
    """
    Provides a convenience torch.utils.data.Dataset subclass that allows one to 
    access a targets field while creating concatenations of two datasets. 

    Parameters
    ----------
    dataset1: torch.utils.data.Dataset
        The first dataset to concatenate. Must have a targets field.
    dataset2: torch.utils.data.Dataset
        The second dataset to concatenate. Must have a targets field.
    """
    def __init__(self, dataset1, dataset2):
        self.dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
        self.targets = torch.Tensor(list(dataset1.targets) + list(dataset2.targets)).type(torch.long)
        
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)


class LabeledToUnlabeledDataset(Dataset):
    """
    Provides a convenience torch.utils.data.Dataset subclass that allows one to 
    ignore the labels in a labeled dataset, thereby making it unlabeled.

    Parameters
    ----------
    wrapped_dataset: torch.utils.data.Dataset
        The labeled dataset in which only the data will be returned.
    """
    def __init__(self, wrapped_dataset):
        self.wrapped_dataset = wrapped_dataset

    def __getitem__(self, index):
        data, label = self.wrapped_dataset[index]
        return data

    def __len__(self):
        return len(self.wrapped_dataset)