import numpy as np
import os
import cv2
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
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_idc_train'], replace=False))
            train_idx += class_train_idx
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_idc_val'], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            
            ### Taking examples from different classes instead of oversampling
            if len(remain_idx)>=split_cfg['per_idc_lake']:
                class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_idc_lake'], replace=False))
            elif len(remain_idx)<split_cfg['per_idc_lake']:
                class_lake_idx = list(np.random.choice(np.array(remain_idx), size=len(remain_idx), replace=False)) 
            ###################################################################
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


class OrganDataset(Dataset):
    def __init__(self, data, root='/mnt/data2/akshit/data/organmnist/', transform=None):
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

class OOD3Dataset(Dataset):
    def __init__(self, data, root='/mnt/data2/akshit/data/organmnist/', transform=None, oodcls=11):
        self.root = root
        self.transform = transform
        self.images = data
        self.targets = [oodcls for i in range(len(data))]

        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = Image.fromarray(np.uint8(self.images[idx])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.targets[idx]
        return img, label

def load_dataset_custom_aid(datadir, feature, split_cfg, augVal=False, dataAug=True):
    
    num_cls = 6 #including OOD
    path = '/mnt/data2/akshit/data/organmnist/'
    # download_path = '/mnt/data2/akshit/data/cifar10'    
    train_id = np.load(f'{path}a/train.npz', allow_pickle=True)
    test_id = np.load(f'{path}a/test_balanced.npz', allow_pickle=True)
    
    #concatenating only images from c,s
    train_ood = np.concatenate((np.load(f'{path}c/train.npz', allow_pickle=True)['images'],np.load(f'{path}s/train.npz', allow_pickle=True)['images']))

    id_training_transform = transforms.Compose([transforms.RandomCrop(28), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    id_test_transform = transforms.Compose([transforms.Resize(28), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    # Get the dataset objects from PyTorch. Here, CIFAR10 is downloaded, and the transform is applied when points 
    # are retrieved.
    organ_full_train = OrganDataset(data=train_id, transform=id_training_transform)
    organ_test = OrganDataset(data=test_id, transform=id_test_transform)
    ood_train = OOD3Dataset(data=train_ood, transform=id_training_transform,oodcls=num_cls-1)
    # ood_test = OOD2Dataset('test',transform=id_test_transform)
    # ood_val = OOD2Dataset('val',transform=id_training_transform)
    fullset = ConcatDataset([organ_full_train, ood_train])
    fullset.targets = np.append(organ_full_train.targets, ood_train.targets)
    test_set = organ_test
    
    if(feature=="ood"):
        train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data(fullset, test_set, split_cfg, num_cls, augVal)
        print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
        return train_set, val_set, test_set, lake_set, ood_cls_idx, split_cfg['num_cls_idc']

def load_dataset_custom_cid(datadir, feature, split_cfg, augVal=False, dataAug=True):
    
    num_cls = 6 #including OOD
    path = '/mnt/data2/akshit/data/organmnist/'
    # download_path = '/mnt/data2/akshit/data/cifar10'    
    train_id = np.load(f'{path}c/train.npz', allow_pickle=True)
    test_id = np.load(f'{path}c/test_balanced.npz', allow_pickle=True)
    
    #concatenating only images from a,s
    train_ood = np.concatenate((np.load(f'{path}a/train.npz', allow_pickle=True)['images'],np.load(f'{path}s/train.npz', allow_pickle=True)['images']))

    id_training_transform = transforms.Compose([transforms.RandomCrop(28), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    id_test_transform = transforms.Compose([transforms.Resize(28), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    # Get the dataset objects from PyTorch. Here, CIFAR10 is downloaded, and the transform is applied when points 
    # are retrieved.
    organ_full_train = OrganDataset(data=train_id, transform=id_training_transform)
    organ_test = OrganDataset(data=test_id, transform=id_test_transform)
    ood_train = OOD3Dataset(data=train_ood, transform=id_training_transform, oodcls=num_cls-1)
    # ood_test = OOD2Dataset('test',transform=id_test_transform)
    # ood_val = OOD2Dataset('val',transform=id_training_transform)
    fullset = ConcatDataset([organ_full_train, ood_train])
    fullset.targets = np.append(organ_full_train.targets, ood_train.targets)
    test_set = organ_test
    
    if(feature=="ood"):
        train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data(fullset, test_set, split_cfg, num_cls, augVal)
        print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
        return train_set, val_set, test_set, lake_set, ood_cls_idx, split_cfg['num_cls_idc']

def load_dataset_custom_sid(datadir, feature, split_cfg, augVal=False, dataAug=True):
    
    num_cls = 6 #including OOD
    path = '/mnt/data2/akshit/data/organmnist/'
    # download_path = '/mnt/data2/akshit/data/cifar10'    
    train_id = np.load(f'{path}s/train.npz', allow_pickle=True)
    test_id = np.load(f'{path}s/test_balanced.npz', allow_pickle=True)
    
    #concatenating only images from a,c
    train_ood = np.concatenate((np.load(f'{path}a/train.npz', allow_pickle=True)['images'],np.load(f'{path}c/train.npz', allow_pickle=True)['images']))

    id_training_transform = transforms.Compose([transforms.RandomCrop(28), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    id_test_transform = transforms.Compose([transforms.Resize(28), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    # Get the dataset objects from PyTorch. Here, CIFAR10 is downloaded, and the transform is applied when points 
    # are retrieved.
    organ_full_train = OrganDataset(data=train_id, transform=id_training_transform)
    organ_test = OrganDataset(data=test_id, transform=id_test_transform)
    ood_train = OOD3Dataset(data=train_ood, transform=id_training_transform,oodcls=num_cls-1)
    # ood_test = OOD2Dataset('test',transform=id_test_transform)
    # ood_val = OOD2Dataset('val',transform=id_training_transform)
    fullset = ConcatDataset([organ_full_train, ood_train])
    fullset.targets = np.append(organ_full_train.targets, ood_train.targets)
    test_set = organ_test
    
    if(feature=="ood"):
        train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data(fullset, test_set, split_cfg, num_cls, augVal)
        print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
        return train_set, val_set, test_set, lake_set, ood_cls_idx, split_cfg['num_cls_idc']
