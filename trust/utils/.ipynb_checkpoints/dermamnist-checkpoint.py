import numpy as np
import os
import torch
import torchvision
from sklearn import datasets
from torchvision import datasets, transforms
import PIL.Image as Image
from .utils import *
np.random.seed(42)
torch.manual_seed(42)
from torchvision.datasets import cifar
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader

class DermaDataset(Dataset):
    def __init__(self, data, root='/mnt/data2/akshit/data/', transform=None):
        self.root = root
        self.transform = transform
        self.images = data['images']
        self.targets = data['labels'].flatten()

        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = Image.fromarray(np.uint8(self.images[idx])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.targets[idx]
        return img, label


def getOODtargets(targets, sel_cls_idx, ood_cls_id):
    
    ood_targets = []
    targets_list = list(targets)
    for i in range(len(targets_list)):
        if(targets_list[i] in list(sel_cls_idx)):
            ood_targets.append(targets_list[i])
        else:
            ood_targets.append(ood_cls_id)
    print("num ood samples: ", ood_targets.count(ood_cls_id))
    return torch.Tensor(ood_targets)

def create_ood_data(fullset, testset, split_cfg, num_cls, augVal):
    
    np.random.seed(42)
    train_idx = []
    val_idx = []
    lake_idx = []
    test_idx = [] 
    selected_classes = np.array(list(range(split_cfg['num_cls_idc'])))
    for i in range(num_cls): #all_classes
        full_idx_class = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
        if(i in selected_classes):
            test_idx_class = list(torch.where(torch.Tensor(testset.targets) == i)[0].cpu().numpy())
            test_idx += test_idx_class
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_idc_train'], replace=True))
            train_idx += class_train_idx
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_idc_val'], replace=True))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_idc_lake'], replace=True))
        else:
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_ood_train'], replace=False)) #always 0
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_ood_val'], replace=False)) #Only for CG ood val has samples
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_ood_lake'], replace=False)) #many ood samples in lake
    
        if(augVal and (i in selected_classes)): #augment with samples only from the imbalanced classes
            train_idx += class_val_idx
        val_idx += class_val_idx
        lake_idx += class_lake_idx
    
    train_set = SubsetWithTargets(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx])
    val_set = SubsetWithTargets(fullset, val_idx, torch.Tensor(fullset.targets)[val_idx])
    lake_set = SubsetWithTargets(fullset, lake_idx, getOODtargets(torch.Tensor(fullset.targets)[lake_idx], selected_classes, split_cfg['num_cls_idc']))
    test_set = SubsetWithTargets(testset, test_idx, torch.Tensor(testset.targets)[test_idx])

    return train_set, val_set, test_set, lake_set, selected_classes

############
# OOD TYPE 1
############

def load_dataset_custom_1(datadir, feature, split_cfg, augVal=False, dataAug=True):
    
    num_cls = 8
    path = '/mnt/data2/akshit/'
    download_path = '/mnt/data2/akshit/data/cifar10'    
    train_data = np.load(f'{path}data/dermamnist/dm_train.npz', allow_pickle=True)
    val_data = np.load(f'{path}data/dermamnist/dm_val.npz', allow_pickle=True)
    test_data = np.load(f'{path}data/dermamnist/dm_test_balanced.npz', allow_pickle=True)
    ptrain={
    'images': np.concatenate((train_data['images'],val_data['images'])),
    'labels': np.concatenate((train_data['labels'],val_data['labels']))
    }

    # Define the number of classes in our modified CIFAR10, which is 6. We also define our ID classes
    cifar_training_transform = transforms.Compose([transforms.RandomCrop(28), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    cifar_test_transform = transforms.Compose([transforms.Resize(28), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    cifar_label_transform = lambda x: 7

    # Get the dataset objects from PyTorch. Here, CIFAR10 is downloaded, and the transform is applied when points 
    # are retrieved.
    cifar10_full_train = cifar.CIFAR10(download_path, train=True, download=False, transform=cifar_training_transform, target_transform=cifar_label_transform)
    cifar10_test = cifar.CIFAR10(download_path, train=False, download=False, transform=cifar_test_transform, target_transform=cifar_label_transform)

    
    derma_full_train = DermaDataset(data=ptrain, transform=cifar_training_transform)
    derma_test = DermaDataset(data=test_data, transform=cifar_test_transform)
    
    fullset = ConcatDataset([derma_full_train, cifar10_full_train])
    fullset.targets = np.append(derma_full_train.targets, [7 for i in range(50000)])
    test_set = derma_test
    
    if(feature=="ood"):
        train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data(fullset, test_set, split_cfg, num_cls, augVal)
        print("Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
        return train_set, val_set, test_set, lake_set, ood_cls_idx, split_cfg['num_cls_idc']

#############
## Taking examples from different classes instead of oversampling
#############    
def create_ood_data_2(fullset, testset, split_cfg, num_cls, augVal):
    
    np.random.seed(42)
    train_idx = []
    val_idx = []
    lake_idx = []
    test_idx = [] 
    selected_classes = np.array(list(range(split_cfg['num_cls_idc'])))
    for i in range(num_cls): #all_classes
        full_idx_class = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
        if(i in selected_classes):
            test_idx_class = list(torch.where(torch.Tensor(testset.targets) == i)[0].cpu().numpy())
            test_idx += test_idx_class
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_idc_train'], replace=False))
            train_idx += class_train_idx
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_idc_val'], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            ## Taking examples from different classes instead of oversampling
            if len(remain_idx)>=split_cfg['per_idc_lake']:
                class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_idc_lake'], replace=False))
            elif len(remain_idx)<split_cfg['per_idc_lake']:
                class_lake_idx = list(np.random.choice(np.array(remain_idx), size=len(remain_idx), replace=False)) 
            
        else:
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_ood_train'], replace=False)) #always 0
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_ood_val'], replace=False)) #Only for CG ood val has samples
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_ood_lake'], replace=False)) #many ood samples in lake
    
        if(augVal and (i in selected_classes)): #augment with samples only from the imbalanced classes
            train_idx += class_val_idx
        val_idx += class_val_idx
        lake_idx += class_lake_idx
    
    train_set = SubsetWithTargets(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx])
    val_set = SubsetWithTargets(fullset, val_idx, torch.Tensor(fullset.targets)[val_idx])
    lake_set = SubsetWithTargets(fullset, lake_idx, getOODtargets(torch.Tensor(fullset.targets)[lake_idx], selected_classes, split_cfg['num_cls_idc']))
    test_set = SubsetWithTargets(testset, test_idx, torch.Tensor(testset.targets)[test_idx])

    return train_set, val_set, test_set, lake_set, selected_classes    
    
def load_dataset_custom_2(datadir, feature, split_cfg, augVal=False, dataAug=True):
    
    num_cls = 8
    path = '/mnt/data2/akshit/'
    download_path = '/mnt/data2/akshit/data/cifar10'    
    train_data = np.load(f'{path}data/dermamnist/dm_train.npz', allow_pickle=True)
    val_data = np.load(f'{path}data/dermamnist/dm_val.npz', allow_pickle=True)
    test_data = np.load(f'{path}data/dermamnist/dm_test_balanced.npz', allow_pickle=True)
    ptrain={
    'images': np.concatenate((train_data['images'],val_data['images'])),
    'labels': np.concatenate((train_data['labels'],val_data['labels']))
    }

    # Define the number of classes in our modified CIFAR10, which is 6. We also define our ID classes
    cifar_training_transform = transforms.Compose([transforms.RandomCrop(28), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    cifar_test_transform = transforms.Compose([transforms.Resize(28), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    cifar_label_transform = lambda x: 7

    # Get the dataset objects from PyTorch. Here, CIFAR10 is downloaded, and the transform is applied when points 
    # are retrieved.
    cifar10_full_train = cifar.CIFAR10(download_path, train=True, download=False, transform=cifar_training_transform, target_transform=cifar_label_transform)
    cifar10_test = cifar.CIFAR10(download_path, train=False, download=False, transform=cifar_test_transform, target_transform=cifar_label_transform)

    
    derma_full_train = DermaDataset(data=ptrain, transform=cifar_training_transform)
    derma_test = DermaDataset(data=test_data, transform=cifar_test_transform)
    
    fullset = ConcatDataset([derma_full_train, cifar10_full_train])
    fullset.targets = np.append(derma_full_train.targets, [7 for i in range(50000)])
    test_set = derma_test
    
    if(feature=="ood"):
        train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data_2(fullset, test_set, split_cfg, num_cls, augVal)
        print("Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
        return train_set, val_set, test_set, lake_set, ood_cls_idx, split_cfg['num_cls_idc']
