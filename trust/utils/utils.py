import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, roc_auc_score

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

def get_roc_auc(target, output, n_classes):
    """
    Function to compute false positive rate(fpr), true positive rate(tpr), and area under ROC(Reciever Operator Characteristics) curve for 
    a list of predicted outputs and ground truth targets. The output is in the form of three dictionaries with class numbers as keys and the values
    of fpr, tpr, area under ROC curve. 

    Parameters
    ----------
    target: numpy.ndarray
        The ground truth label of the set
    output: sequence
        Predicted output of the set
    n_classes: int
        The number of classes in the dataset
    """
    target = label_binarize(target, classes=list(range(n_classes)))
    output = np.array(output)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(target[:,i],output[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        fpr[i] = fpr[i].tolist()
        tpr[i] = tpr[i].tolist()
        roc_auc[i] = roc_auc[i].tolist()
    return fpr,tpr,roc_auc

def get_pr_auc(target, output, n_classes):
    """
    Function to compute precision, recall, and area under Precision Recall curve for a list of predicted outputs and ground truth targets. 
    The output is in the form of three dictionaries with class numbers as keys and the values of precision, recall, area under precision recall curve. 

    Parameters
    ----------
    target: numpy.ndarray
        The ground truth label of the set
    output: sequence
        Predicted output of the set
    n_classes: int
        The number of classes in the dataset
    """
    target = label_binarize(target, classes=list(range(n_classes)))
    output = np.array(output)
    precision = dict()
    recall = dict()
    aupr = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(target[:,i],output[:,i])
        aupr[i] = auc(recall[i], precision[i])
        precision[i] = precision[i].tolist()
        recall[i] = recall[i].tolist()
        aupr[i] = aupr[i].tolist()
    return precision, recall, aupr

def get_macro_roc_auc(target, output, n_classes):
    """
    Function to compute macro average ROC(Reciever Operater Characteristics) for a list of predicted outputs and ground truth targets. 
    The output is in the form of three dictionaries with class numbers as keys and the values
    of precision, recall, area under precision recall curve. 

    Parameters
    ----------
    target: numpy.ndarray
        The ground truth label of the set
    output: sequence
        Predicted output of the set
    n_classes: int
        The number of classes in the dataset
    """
    target = label_binarize(target,classes=list(range(n_classes)))
    output = np.array(output)
    return roc_auc_score(target,output,average="macro")