'''
Customized dataset loading code for medical datasets at http://medmnist.com/
'''

import numpy as np
import os
import torch
import torchvision
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
import PIL.Image as Image
from .utils import *
from .medmnist import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal, TissueMNIST
np.random.seed(42)
torch.manual_seed(42)

from torch.utils.data import Dataset

name_to_class = {
        "pathmnist": (PathMNIST,9),
        "chestmnist": (ChestMNIST,14),
        "dermamnist": (DermaMNIST,7),
        "octmnist": (OCTMNIST,4),
        "pneumoniamnist": (PneumoniaMNIST,2),
        "retinamnist": (RetinaMNIST,5),
        "breastmnist": (BreastMNIST,2),
        "axial_organmnist": (OrganMNISTAxial,11),
        "coronal_organmnist": (OrganMNISTCoronal,11),
        "sagittal_organmnist": (OrganMNISTSagittal,11),
        "tissuemnist":(TissueMNIST,8),
    }


def create_class_imb(dset_name, fullset, split_cfg, num_cls, augVal):
    np.random.seed(42)
    train_idx = []
    val_idx = []
    lake_idx = []
    selected_classes = np.random.choice(np.arange(num_cls), size=split_cfg['num_cls_imbalance'], replace=False) #classes to imbalance
    for i in range(num_cls): #all_classes
        full_idx_class = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
        if(i in selected_classes):
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_imbclass_train'], replace=False))
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_imbclass_val'], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_imbclass_lake'], replace=False))
        else:
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_class_train'], replace=False))
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_class_val'], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_class_lake'], replace=False))
    
        train_idx += class_train_idx
        if(augVal and (i in selected_classes)): #augment with samples only from the imbalanced classes
            train_idx += class_val_idx
        val_idx += class_val_idx
        lake_idx += class_lake_idx
    train_set = SubsetWithTargets(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx])
    val_set = SubsetWithTargets(fullset, val_idx, torch.Tensor(fullset.targets)[val_idx])
    lake_set = SubsetWithTargets(fullset, lake_idx, torch.Tensor(fullset.targets)[lake_idx])
    return train_set, val_set, lake_set, selected_classes


def create_class_imb_bio(dset_name, fullset, split_cfg, num_cls, augVal):
    np.random.seed(42)
    train_idx = []
    val_idx = []
    lake_idx = []
    selected_classes=split_cfg['sel_cls_idx']
    for i in range(num_cls): #all_classes
        full_idx_class = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
        if(i in selected_classes):
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_imbclass_train'][i], replace=False))
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_imbclass_val'][i], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_imbclass_lake'][i], replace=False))
        else:
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_class_train'][i], replace=False))
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_class_val'][i], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_class_lake'][i], replace=False))
    
        train_idx += class_train_idx
        if(augVal and (i in selected_classes)): #augment with samples only from the imbalanced classes
            train_idx += class_val_idx
        val_idx += class_val_idx
        lake_idx += class_lake_idx
    train_set = SubsetWithTargets(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx])
    val_set = SubsetWithTargets(fullset, val_idx, torch.Tensor(fullset.targets)[val_idx])
    lake_set = SubsetWithTargets(fullset, lake_idx, torch.Tensor(fullset.targets)[lake_idx])
    return train_set, val_set, lake_set, selected_classes

def create_class_imb_bio_with_testset(dset_name, fullset, testset, split_cfg, num_cls, augVal):
    np.random.seed(42)
    train_idx = []
    val_idx = []
    lake_idx = []
    test_idx = []
    selected_classes=split_cfg['sel_cls_idx']
    for i in range(num_cls): #all_classes
        full_idx_class = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
        test_idx_class = list(torch.where(torch.Tensor(testset.targets) == i)[0].cpu().numpy())
        if(i in selected_classes):
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_imbclass_train'][i], replace=False))
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_imbclass_val'][i], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_imbclass_lake'][i], replace=False))
            class_test_idx = list(np.random.choice(np.array(test_idx_class), size=split_cfg['per_imbclass_test'][i], replace=False)) 
        else:
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_class_train'][i], replace=False))
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_class_val'][i], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_class_lake'][i], replace=False))
            class_test_idx = list(np.random.choice(np.array(test_idx_class), size=split_cfg['per_class_test'][i], replace=False))
    
        train_idx += class_train_idx
        test_idx += class_test_idx
        if(augVal and (i in selected_classes)): #augment with samples only from the imbalanced classes
            train_idx += class_val_idx
        val_idx += class_val_idx
        lake_idx += class_lake_idx
    train_set = SubsetWithTargets(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx])
    val_set = SubsetWithTargets(fullset, val_idx, torch.Tensor(fullset.targets)[val_idx])
    lake_set = SubsetWithTargets(fullset, lake_idx, torch.Tensor(fullset.targets)[lake_idx])
    test_set = SubsetWithTargets(testset, test_idx, torch.Tensor(testset.targets)[test_idx])
    return train_set, val_set, lake_set, test_set, selected_classes    

def create_longtail(dset_name, fullset, split_cfg, num_cls, augVal):
    np.random.seed(42)
    train_idx = []
    val_idx = []
    lake_idx = []
    selected_classes=split_cfg['sel_cls_idx']
    for i in range(num_cls): #all_classes
        full_idx_class = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
        if(i in selected_classes):
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_imbclass_train'][i], replace=False))
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_imbclass_val'][i], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_imbclass_lake'][i], replace=False))
        else:
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_class_train'][i], replace=False))
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_class_val'][i], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_class_lake'][i], replace=False))
    
        train_idx += class_train_idx
        if(augVal and (i in selected_classes)): #augment with samples only from the imbalanced classes
            train_idx += class_val_idx
        val_idx += class_val_idx
        lake_idx += class_lake_idx
    train_set = SubsetWithTargets(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx])
    val_set = SubsetWithTargets(fullset, val_idx, torch.Tensor(fullset.targets)[val_idx])
    lake_set = SubsetWithTargets(fullset, lake_idx, torch.Tensor(fullset.targets)[lake_idx])
    return train_set, val_set, lake_set, selected_classes
    
def load_biodataset_custom(datadir, dset_name, feature, split_cfg, augVal=False, dataAug=True):
    """
    Loads a biomedical dataset with additional options to create class imbalances, out-of-distribution classes, and redundancies.

    Parameters
    ----------
    datadir : string
        The root directory in which the data is stored (or should be downloaded)
    dset_name : string
        The name of the dataset. This should be one of 'cifar10', 'mnist', 'svhn', 'cifar100', 'breast-density'.
    feature : string
        The modification that should be applied to the dataset. This should be one of 'classimb', 'ood', 'duplicate', 'vanilla'
    split_cfg : dict
        Contains information relating to the dataset splits that should be created. Some of the keys for this dictionary are as follows:
            'per_imbclass_train': int
                The number of examples in the train set for each imbalanced class (classimb)
            'per_imbclass_val': int
                The number of examples in the validation set for each imbalanced class (classimb)
            'per_imbclass_lake': int
                The number of examples in the lake set for each imbalanced class (classimb)
            'per_class_train': int
                The number of examples in the train set for each balanced class (classimb
            'per_class_val': int
                The number of examples in the validation set for each balanced class (classimb)
            'per_class_lake': int
                The number of examples in the lake set for each balanced class (classimb)
            'sel_cls_idx': list
                A list of classes that are affected by class imbalance. (classimb)
            'train_size': int
                The size of the train set (vanilla, duplicate)
            'val_size': int
                The size of the validation set (vanilla, duplicate)
            'lake_size': int
                The size of the lake set (vanilla, duplicate)
            'num_rep': int
                The number of times to repeat a selection in the lake set (duplicate)
            'lake_subset_repeat_size': int
                The size of the repeated selection in the lake set (duplicate)
            'num_cls_imbalance': int
                The number of classes to randomly affect by class imbalance. (classimb)
            'num_cls_idc': int
                The number of in-distribution classes to keep (ood)
            'per_idc_train': int
                The number of in-distribution examples to keep in the train set per class (ood)
            'per_idc_val': int
                The number of in-distribution examples to keep in the validation set per class (ood)
            'per_idc_lake': int
                The number of in-distribution examples to keep in the lake set per class (ood)
            'per_ood_train': int
                The number of OOD examples to keep in the train set per class (ood)
            'per_ood_val': int
                The number of OOD examples to keep in the validation set per class (ood)
            'per_ood_lake': int
                The number of OOD examples to keep in the lake set per class (ood)         
    augVal : bool, optional
        If True, the train set will also contain affected classes from the validation set. The default is False.
    dataAug : bool, optional
        If True, the all but the test set will be affected by random cropping and random horizontal flip. The default is True.

    Returns
    -------
    tuple
        Returns a train set, validation set, test set, lake set, and number of classes. Amount of returned items depends on specific configuration.
        Each set is an instance of torch.utils.data.Dataset
    """
    if(not(os.path.exists(datadir))):
        os.mkdir(datadir)

    if(dset_name[-5:]=="mnist"):
        num_cls=name_to_class[dset_name][1]
        datadir = datadir
        input_size = 32
        data_transforms = {
            'train' : transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5],std=[0.5])
            ]),
            'test' : transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5],std=[0.5])
            ])
        }

        Dataclass = name_to_class[dset_name][0]
        fullset = Dataclass(root=datadir,split="train",transform=data_transforms['train'],download=False)
        test_set = Dataclass(root=datadir,split="test",transform=data_transforms['test'],download=False)
        
        if(feature=="classimb"):
            train_set, val_set, lake_set, imb_cls_idx = create_class_imb_bio(dset_name, fullset, split_cfg, num_cls, augVal)
            print(dset_name+" Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
        elif(feature=="longtail"):
            train_set, val_set, lake_set, imb_cls_idx = create_longtail(dset_name, fullset, split_cfg, num_cls, augVal)
            print(dset_name+" Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls

    if(dset_name=="breast_cancer"):
        num_cls=2
        data_dir = datadir
        input_size=224
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        fullset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
        test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
        if(feature=="classimb"):
            train_set, val_set, lake_set, imb_cls_idx = create_class_imb_bio(dset_name, fullset, split_cfg, num_cls, augVal)
            print("Breast-Cancer Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
        elif(feature=="longtail"):
            train_set, val_set, lake_set, imb_cls_idx = create_longtail(dset_name, fullset, split_cfg, num_cls, augVal)
            print("Breast-Cancer Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls


    
    if(dset_name=="breast_density"):
        num_cls=4
        data_dir = datadir
        input_size=224
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        fullset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
        test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
        if(feature=="classimb"):
            train_set, val_set, lake_set, imb_cls_idx = create_class_imb_bio(dset_name, fullset, split_cfg, num_cls, augVal)
            print("Breast-density Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
        elif(feature=="longtail"):
            train_set, val_set, lake_set, imb_cls_idx = create_longtail(dset_name, fullset, split_cfg, num_cls, augVal)
            print("Breast-density Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls            
    