# %% [markdown]
# # Targeted Selection Demo For Biomedical Datasets With Rare Classes

# %% [markdown]
# ### Imports 

# %%
import time
import random
import datetime
import copy
import numpy as np
from tabulate import tabulate
import os
import csv
import json
import subprocess
import sys
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
from matplotlib import pyplot as plt
import sys
sys.path.append('/home/wassal/trust-wassal/')

from trust.utils.models.resnet import ResNet18
from trust.utils.models.resnet import ResNet50
from trust.utils.custom_dataset import load_dataset_custom
from torch.utils.data import Subset
from torch.autograd import Variable
import tqdm
from math import floor
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from trust.strategies.smi import SMI
from trust.strategies.scmi import SCMI
from trust.strategies.random_sampling import RandomSampling
from trust.strategies.wassal_multiclass import WASSAL_Multiclass
from trust.strategies.wassal_private import WASSAL_P

sys.path.append('/home/wassal/distil')
from distil.active_learning_strategies.entropy_sampling import EntropySampling
from distil.active_learning_strategies.badge import BADGE
from distil.active_learning_strategies.glister import GLISTER
from distil.active_learning_strategies.gradmatch_active import  GradMatchActive
from distil.active_learning_strategies.core_set import CoreSet
from distil.active_learning_strategies.least_confidence_sampling import LeastConfidenceSampling
from distil.active_learning_strategies.margin_sampling import MarginSampling

seed=42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
from trust.utils.utils import *
from trust.utils.viz import tsne_smi
import math
# %% [markdown]
# ### Helper functions

# %%
def model_eval_loss(data_loader, model, criterion):
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, querys) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
                
def create_model(name, num_cls, device, embedding_type):
    if name == 'ResNet18':
        if embedding_type == "gradients":
            model = ResNet18(num_cls)
        else:
            model = models.resnet18()
    elif name == 'ResNet50':
        if embedding_type == "gradients":
            model = ResNet50(num_cls)
        else:
            model = models.resnet50()
    elif name == 'MnistNet':
        model = MnistNet()
    elif name == 'ResNet164':
        model = ResNet164(num_cls)
    model.apply(init_weights)
    model = model.to(device)
    return model

def loss_function():
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction='none')
    return criterion, criterion_nored

def optimizer_with_scheduler(model, num_epochs, learning_rate, m=0.9, wd=5e-4):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=m, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    return optimizer, scheduler

def optimizer_without_scheduler(model, learning_rate, m=0.9, wd=5e-4):
#     optimizer = optim.Adam(model.parameters(),weight_decay=wd)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=m, weight_decay=wd)
    return optimizer

def generate_cumulative_timing(mod_timing):
    tmp = 0
    mod_cum_timing = np.zeros(len(mod_timing))
    for i in range(len(mod_timing)):
        tmp += mod_timing[i]
        mod_cum_timing[i] = tmp
    return mod_cum_timing/3600

def displayTable(val_err_log, tst_err_log):
    col1 = [str(i) for i in range(10)]
    val_acc = [str(100-i) for i in val_err_log]
    tst_acc = [str(100-i) for i in tst_err_log]
    table = [col1, val_acc, tst_acc]
    table = map(list, zip(*table))
    print(tabulate(table, headers=['Class', 'Val Accuracy', 'Test Accuracy'], tablefmt='orgtbl'))

def find_err_per_class(test_set, val_set, final_val_classifications, final_val_predictions, final_tst_classifications, 
                       final_tst_predictions, saveDir, prefix):
    val_err_idx = list(np.where(np.array(final_val_classifications) == False)[0])
    tst_err_idx = list(np.where(np.array(final_tst_classifications) == False)[0])
    val_class_err_idxs = []
    tst_err_log = []
    val_err_log = []
    for i in range(num_cls):
        tst_class_idxs = list(torch.where(torch.Tensor(test_set.targets) == i)[0].cpu().numpy())
        val_class_idxs = list(torch.where(torch.Tensor(val_set.targets.float()) == i)[0].cpu().numpy())
        #err classifications per class
        val_err_class_idx = set(val_err_idx).intersection(set(val_class_idxs))
        tst_err_class_idx = set(tst_err_idx).intersection(set(tst_class_idxs))
        if(len(val_class_idxs)>0):
            val_error_perc = round((len(val_err_class_idx)/len(val_class_idxs))*100,2)
        else:
            val_error_perc = 0
        tst_error_perc = round((len(tst_err_class_idx)/len(tst_class_idxs))*100,2)
#         print("val, test error% for class ", i, " : ", val_error_perc, tst_error_perc)
        val_class_err_idxs.append(val_err_class_idx)
        tst_err_log.append(tst_error_perc)
        val_err_log.append(val_error_perc)
    displayTable(val_err_log, tst_err_log)
    tst_err_log.append(sum(tst_err_log)/len(tst_err_log))
    val_err_log.append(sum(val_err_log)/len(val_err_log))
    return tst_err_log, val_err_log, val_class_err_idxs


def aug_train_subset(train_set, lake_set, true_lake_set, subset, lake_subset_idxs, budget, augrandom=False):
    all_lake_idx = list(range(len(lake_set)))
    if(not(len(subset)==budget) and augrandom):
        print("Budget not filled, adding ", str(int(budget) - len(subset)), " randomly.")
        remain_budget = int(budget) - len(subset)
        remain_lake_idx = list(set(all_lake_idx) - set(subset))
        random_subset_idx = list(np.random.choice(np.array(remain_lake_idx), size=int(remain_budget), replace=False))
        subset += random_subset_idx
    if str(type(true_lake_set.targets)) == "<class 'numpy.ndarray'>":
        lake_ss = SubsetWithTargets(true_lake_set, subset, torch.Tensor(true_lake_set.targets.astype(np.float))[subset])
    else:
        lake_ss = SubsetWithTargets(true_lake_set, subset, torch.Tensor(true_lake_set.targets.float())[subset])
    remain_lake_idx = list(set(all_lake_idx) - set(lake_subset_idxs))
    if str(type(true_lake_set.targets)) == "<class 'numpy.ndarray'>":
        remain_lake_set = SubsetWithTargets(lake_set, remain_lake_idx, torch.Tensor(lake_set.targets.astype(np.float))[remain_lake_idx])
    else:
        remain_lake_set = SubsetWithTargets(lake_set, remain_lake_idx, torch.Tensor(lake_set.targets.float())[remain_lake_idx])
    if str(type(true_lake_set.targets)) == "<class 'numpy.ndarray'>":
        remain_true_lake_set = SubsetWithTargets(true_lake_set, remain_lake_idx, torch.Tensor(true_lake_set.targets.astype(np.float))[remain_lake_idx])
    else:
        remain_true_lake_set = SubsetWithTargets(true_lake_set, remain_lake_idx, torch.Tensor(true_lake_set.targets.float())[remain_lake_idx])
#     print(len(lake_ss),len(remain_lake_set),len(lake_set))
    aug_train_set = ConcatWithTargets(train_set, lake_ss)
    aug_trainloader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True, pin_memory=True)
    return aug_train_set, remain_lake_set, remain_true_lake_set, lake_ss
                        
def getQuerySet(val_set,imb_cls_idx):
    
    miscls_idx = []
    
    
    for i in imb_cls_idx:
        imb_cls_samples = list(torch.where(torch.Tensor(val_set.targets.float()) == i)[0].cpu().numpy())
        miscls_idx += imb_cls_samples
    print("Total samples from imbalanced classes as Queries (Size of query set): ", len(miscls_idx))
    return SubsetWithTargets(val_set, miscls_idx, val_set.targets[miscls_idx])

def getPrivateSet(val_set,imb_cls_idx):
    # Get all the indices in the val_set
    all_idx = list(range(len(val_set.targets)))
    miscls_idx = []
    
    
    for i in imb_cls_idx:
        imb_cls_samples = list(torch.where(torch.Tensor(val_set.targets.float()) == i)[0].cpu().numpy())
        miscls_idx += imb_cls_samples
     # Get indices that aren't in the query class samples
    private_idx = list(set(all_idx) - set(miscls_idx))
    print("Total samples from imbalanced classes as Private (Size of private set): ", len(private_idx))
    return SubsetWithTargets(val_set, private_idx, val_set.targets[private_idx])



def getPerClassSel(lake_set, subset, num_cls):
    perClsSel = []
    if str(type(lake_set.targets)) == "<class 'numpy.ndarray'>":
        subset_cls = torch.Tensor(lake_set.targets.astype(np.float))[subset]
    else:
        subset_cls = torch.Tensor(lake_set.targets.float())[subset]
    for i in range(num_cls):
        cls_subset_idx = list(torch.where(subset_cls == i)[0].cpu().numpy())
        perClsSel.append(len(cls_subset_idx))
    return perClsSel

def print_final_results(res_dict, sel_cls_idx):
    print("Gain in overall test accuracy: ", res_dict['test_acc'][1]-res_dict['test_acc'][0])
    #bf_sel_cls_acc = np.array(res_dict['all_class_acc'][0])[sel_cls_idx]
    #af_sel_cls_acc = np.array(res_dict['all_class_acc'][1])[sel_cls_idx]
    #print("Gain in targeted test accuracy: ", np.mean(af_sel_cls_acc-bf_sel_cls_acc))

def analyze_simplex(args, unlabeled_set, simplex_query):
    print("======== analysis on simplex =========")
    unlabeled_loader = torch.utils.data.DataLoader(dataset=unlabeled_set, batch_size=len(unlabeled_set), shuffle=False)
    u_imgs, u_labels = next(iter(unlabeled_loader))
    u_imgs, u_labels = u_imgs.to(args['device']), u_labels.to(args['device'])
    nz_query_idx = simplex_query.nonzero()

    # Using a loop to accommodate an array of target values
    total_correctly_identified = 0
    for query_value in args['target']:
        num_nz_query = (u_labels[nz_query_idx] == query_value).nonzero().shape[0]
        total_correctly_identified += num_nz_query
    print("no of query labels identified correctly: {}/{}".format(total_correctly_identified, nz_query_idx.shape[0]))

    total_query_weight = 0
    for query_value in args['target']:
        query_idx = torch.where(u_labels == query_value)
        query_weight = torch.sum(simplex_query[query_idx])
        total_query_weight += query_weight
    print("Weight of Query samples in simplex_query: {}".format(total_query_weight))
    
class WeightedDataset(Dataset):
    def __init__(self, imgs,targets, simplex_query,private_targets,simplex_private):
        self.imgs = imgs
        self.targets=targets
        self.simplex_query = simplex_query
        self.private_targets = private_targets
        self.simplex_private = simplex_private
        

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.imgs[idx]
        target=self.targets[idx]
        t = self.simplex_query[idx].item()
        private_target = self.private_targets[idx] if self.private_targets is not None else []
        p = self.simplex_private[idx].item() if self.simplex_private is not None else []
        
        return (image, target,t,private_target,p)

#return the elements from the simplex_query that contribute to the given percentage
def top_elements_contribute_to_percentage(simplex_query, n_percent):
   # Pair each value with its original index
    indexed_simplex = list(enumerate(simplex_query))
    
    # Sort based on the value (in descending order)
    sorted_simplex = sorted(indexed_simplex, key=lambda x: x[1], reverse=True)

    # Calculate the total sum of the array
    total_sum = sum(value for index, value in sorted_simplex)
    
    # If the array doesn't sum up to 1, you might want to handle this case
    if total_sum != 1:
        print('Total sum of simplex is', total_sum)

    target_sum = n_percent / 100.0   # Convert percentage to fraction
    cumulative_sum = 0
    selected_indices = []

    # Iterate over the sorted array
    for i, (index, value) in enumerate(sorted_simplex):
        cumulative_sum += value
        selected_indices.append(index)
        if cumulative_sum >= target_sum:
            break

    # Return values and their original indices
    selected_values = [simplex_query[i] for i in selected_indices]
    return selected_values, selected_indices


# %% [markdown]
# # Data, Model & Experimental Settings
# The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes.The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class. The training set contains 50,000 images and test set contains 10,000 images. We will use custom_dataset() function in Trust to simulated a class imbalance scenario using the split_cfg dictionary given below. We then use a ResNet18 model as our task DNN and train it on the simulated imbalanced version of the CIFAR-10 dataset. Next we perform targeted selection using various SMI functions and compare their gain in overall accuracy as well as on the imbalanced classes.

# %%
feature = "classimb"

# datadir = 'data/'
datadir = '/data' #contains the npz file of the data_name dataset listed below
data_name = 'cifar10'

learning_rate = 0.0003
computeClassErrorLog = True
device_id = 0
device = "cuda:"+str(device_id) if torch.cuda.is_available() else "cpu"
miscls = False #Set to True if only the misclassified examples from the imbalanced classes is to be used

num_cls = 10
#budget = 10
visualize_tsne = False
split_cfg = {"train_size":100, #Number of rare classes
             "val_size":200, #Number of samples per rare class in the train dataset
             "lake_size":5000, #Number of samples per rare class in the validation dataset
             
            #  "per_class_train":1000,  #Number of samples per unrare class in the train dataset
            #  "per_class_val":5, #Number of samples per unrare class in the validation dataset
            #  "per_class_lake":3000
             } #Number of samples per unrare class in the unlabeled dataset

print("split_cfg:",split_cfg)

# %% [markdown]
# # Targeted Selection Algorithm
# 1. Given: Initial Labeled set of Examples: 𝐸, large unlabeled dataset: 𝑈, A target subset/slice where we want to improve accuracy: 𝑇, Loss function 𝐿 for learning
# 2. Train model with loss $\mathcal L$ on labeled set $E$ and obtain parameters $\theta_E$
# 3. Compute the gradients $\{\nabla_{\theta_E} \mathcal L(x_i, y_i), i \in U\}$ (using hypothesized labels) and $\{\nabla_{\theta_E} \mathcal L(x_i, y_i), i \in T\}$. 
# (This notebook uses gradients for representation. However, any other representation can be used. Trust also supports using features via the API.)
# 4. Compute the similarity kernels $S$ (this includes kernel of the elements within $U$, within $T$ and between $U$ and $T$) and define a submodular function $f$ and diversity function $g$
# 5. Compute subset $\hat{A}$ by mazximizing the SMI function: $\hat{A} \gets \max_{A \subseteq U, |A|\leq k} I_f(A;T) + \gamma g(A)$
# 6. Obtain the labels of the elements in $A^*$: $L(\hat{A})$
# 7. Train a model on the combined labeled set $E \cup L(\hat{A})$

# %%
def run_targeted_selection(dataset_name, datadir, feature, model_name, budget, split_cfg, learning_rate, run,
                device, computeErrorLog, strategy="SIM", sf="",embedding_type="features"):

    #load the dataset in the class imbalance setting
    train_set, val_set, test_set, lake_set, sel_cls_idx, num_cls = load_dataset_custom(datadir, dataset_name, feature, split_cfg, False, False)
    print("Indices of randomly selected classes for imbalance: ", sel_cls_idx)
    
   #Set batch size for train, validation and test datasets
    N = len(train_set)
    trn_batch_size = len(train_set)
    val_batch_size = len(val_set)
    tst_batch_size = len(test_set)

    #Create dataloaders
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=trn_batch_size,
                                              shuffle=True, pin_memory=True)

    valloader = torch.utils.data.DataLoader(val_set, batch_size=val_batch_size, 
                                            shuffle=False, pin_memory=True)

    tstloader = torch.utils.data.DataLoader(test_set, batch_size=tst_batch_size,
                                             shuffle=False, pin_memory=True)
    
    lakeloader = torch.utils.data.DataLoader(lake_set, batch_size=tst_batch_size,
                                         shuffle=False, pin_memory=True)
    true_lake_set = copy.deepcopy(lake_set)
    # Budget for subset selection
    bud = budget
    #soft subset max budget % of the lake set
    ss_max_budget = 20
    # Variables to store accuracies
    num_rounds=2 #The first round is for training the initial model and the second round is to train the final model
    fulltrn_losses = np.zeros(num_rounds)
    val_losses = np.zeros(num_rounds)
    tst_losses = np.zeros(num_rounds)
    timing = np.zeros(num_rounds)
    val_acc = np.zeros(num_rounds)
    full_trn_acc = np.zeros(num_rounds)
    tst_acc = np.zeros(num_rounds)
    final_tst_predictions = []
    final_tst_classifications = []
    best_val_acc = -1
    csvlog = []
    val_csvlog = []
    # Results logging file
    all_logs_dir = '/home/wassal/trust-wassal/tutorials/results/' + dataset_name  + '/' + feature+"/rounds"+str(num_rounds) + '/'+  sf + '/' + str(bud) + '/' + str(run)
    print("Saving results to: ", all_logs_dir)
    subprocess.run(["mkdir", "-p", all_logs_dir]) #Uncomment for saving results
    exp_name = dataset_name + "_" + feature +  "_" + strategy + "_" + str(len(sel_cls_idx))  +"_" + sf +  '_budget:' + str(bud) + '_rounds:' + str(num_rounds) + '_runs' + str(run)

    #Create a dictionary for storing results and the experimental setting
    res_dict = {"dataset":data_name, 
                "feature":feature, 
                "sel_func":sf,
                "sel_budget":budget, 
                "num_selections":num_rounds-1, 
                "model":model_name, 
                "learning_rate":learning_rate, 
                "setting":split_cfg, 
                "all_class_acc":None, 
                "test_acc":[],
                "sel_per_cls":[], 
                "sel_cls_idx":sel_cls_idx}
    
   
  
    strategy_args = {'batch_size': 4000, 'device':device, 'embedding_type':embedding_type, 'keep_embedding':True,'lr':learning_rate}
    unlabeled_lake_set = LabeledToUnlabeledDataset(lake_set)
    
    if(strategy == "AL"):
        if(sf=="badge"):
            strategy_sel = BADGE(train_set, unlabeled_lake_set, model, num_cls, strategy_args)
        elif(sf=="us"):
            strategy_sel = EntropySampling(train_set, unlabeled_lake_set, model, num_cls, strategy_args)
        elif(sf=="glister" or sf=="glister-tss"):
            strategy_sel = GLISTER(train_set, unlabeled_lake_set, model, num_cls, strategy_args, val_set, typeOf='rand', lam=0.1)
        elif(sf=="gradmatch-tss"):
            strategy_sel = GradMatchActive(train_set, unlabeled_lake_set, model, num_cls, strategy_args, val_set)
        elif(sf=="coreset"):
            strategy_sel = CoreSet(train_set, unlabeled_lake_set, model, num_cls, strategy_args)
        elif(sf=="leastconf"):
            strategy_sel = LeastConfidenceSampling(train_set, unlabeled_lake_set, model, num_cls, strategy_args)
        elif(sf=="margin"):
            strategy_sel = MarginSampling(train_set, unlabeled_lake_set, model, num_cls, strategy_args)
    #if AL_WITHSOFT
    elif(strategy == "AL_WITHSOFT"):
        for_query_set = getQuerySet(ConcatWithTargets(train_set,val_set),sel_cls_idx)
        #
        strategy_softsubset = WASSAL_Multiclass(train_set, unlabeled_lake_set,for_query_set, model,num_cls,strategy_args)
            
        if(sf[:-5]=="badge"):
           
            strategy_sel = BADGE(train_set, unlabeled_lake_set, model, num_cls, strategy_args)
        elif(sf[:-5]=="us"):
            
            strategy_sel = EntropySampling(train_set, unlabeled_lake_set, model, num_cls, strategy_args)
        elif(sf[:-5]=="glister" or sf=="glister-tss"):
            strategy_sel = GLISTER(train_set, unlabeled_lake_set, model, num_cls, strategy_args, val_set, typeOf='rand', lam=0.1)
        elif(sf[:-5]=="gradmatch-tss"):
            strategy_sel = GradMatchActive(train_set, unlabeled_lake_set, model, num_cls, strategy_args, val_set)
        elif(sf[:-5]=="coreset"):
            strategy_sel = CoreSet(train_set, unlabeled_lake_set, model, num_cls, strategy_args)
        elif(sf[:-5]=="leastconf"):
            strategy_sel = LeastConfidenceSampling(train_set, unlabeled_lake_set, model, num_cls, strategy_args)
        elif(sf[:-5]=="margin"):
            strategy_sel = MarginSampling(train_set, unlabeled_lake_set, model, num_cls, strategy_args)



    elif(strategy == "SIM"):
        strategy_args['smi_function'] = sf
        strategy_args['optimizer'] = 'LazyGreedy'
        for_query_set = getQuerySet(ConcatWithTargets(train_set,val_set),sel_cls_idx)
        strategy_sel = SMI(train_set, unlabeled_lake_set, for_query_set, model, num_cls, strategy_args)
    elif(strategy == "SIM_WITHSOFT"):
        #remove the '_soft' from the sf name
        strategy_args['smi_function'] = sf[:-5]
        strategy_args['optimizer'] = 'LazyGreedy'
        for_query_set = getQuerySet(ConcatWithTargets(train_set,val_set),sel_cls_idx)
        strategy_softsubset = WASSAL(train_set, unlabeled_lake_set,for_query_set, model,num_cls,strategy_args)
        strategy_sel = SMI(train_set, unlabeled_lake_set, for_query_set, model, num_cls, strategy_args)
    
    elif(strategy == "SCMI"):
        strategy_args['scmi_function'] = sf
        strategy_args['optimizer'] = 'LazyGreedy'
        for_query_set = getQuerySet(ConcatWithTargets(train_set,val_set),sel_cls_idx)
        for_private_set = getPrivateSet(ConcatWithTargets(train_set,val_set),sel_cls_idx)
        strategy_sel=SCMI(train_set, unlabeled_lake_set, for_query_set, for_private_set, model, num_cls, strategy_args)
    elif(strategy == "SCMI_WITHSOFT"):
        #remove the '_soft' from the sf name
        strategy_args['scmi_function'] = sf[:-5]
        strategy_args['optimizer'] = 'LazyGreedy'
        for_query_set = getQuerySet(ConcatWithTargets(train_set,val_set),sel_cls_idx)
        for_private_set = getPrivateSet(ConcatWithTargets(train_set,val_set),sel_cls_idx)
        strategy_softsubset = WASSAL_P(train_set, unlabeled_lake_set,for_query_set, for_private_set,model,num_cls,strategy_args)
        strategy_sel=SCMI(train_set, unlabeled_lake_set, for_query_set, for_private_set, model, num_cls, strategy_args)
    
    elif(strategy == "random"):
        strategy_sel = RandomSampling(train_set, unlabeled_lake_set, model, num_cls, strategy_args)
    if(strategy == "WASSAL"):
        for_query_set = getQuerySet(ConcatWithTargets(train_set,val_set),sel_cls_idx)
        strategy_sel = WASSAL_Multiclass(train_set, unlabeled_lake_set,for_query_set, model,num_cls,strategy_args)
    elif(strategy == "WASSAL_P"):
        for_query_set = getQuerySet(ConcatWithTargets(train_set,val_set),sel_cls_idx)
        for_private_set = getPrivateSet(ConcatWithTargets(train_set,val_set),sel_cls_idx)
        strategy_sel = WASSAL_P(train_set, unlabeled_lake_set,for_query_set, for_private_set,model,num_cls,strategy_args)

        
    # Loss Functions
    criterion, criterion_nored = loss_function()

    # Getting the optimizer and scheduler
    optimizer = optimizer_without_scheduler(model, learning_rate)

    for i in range(num_rounds):
        tst_loss = 0
        tst_correct = 0
        tst_total = 0
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        if(i==0):
            print("Initial training epoch")
            if(os.path.exists(initModelPath)): #Read the initial trained model if it exists
                model.load_state_dict(torch.load(initModelPath, map_location=device))
                print("Init model loaded from disk, skipping init training: ", initModelPath)
                model.eval()
                with torch.no_grad():
                    final_val_predictions = []
                    final_val_classifications = []
                    for batch_idx, (inputs, targets) in enumerate(valloader):
                        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()
                        final_val_predictions += list(predicted.cpu().numpy())
                        final_val_classifications += list(predicted.eq(targets).cpu().numpy())
  
                    final_tst_predictions = []
                    final_tst_classifications = []
                    for batch_idx, (inputs, targets) in enumerate(tstloader):
                        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        tst_loss += loss.item()
                        _, predicted = outputs.max(1)
                        tst_total += targets.size(0)
                        tst_correct += predicted.eq(targets).sum().item()
                        final_tst_predictions += list(predicted.cpu().numpy())
                        final_tst_classifications += list(predicted.eq(targets).cpu().numpy())                
                    best_val_acc = (val_correct/val_total)
                    val_acc[i] = val_correct / val_total
                    tst_acc[i] = tst_correct / tst_total
                    val_losses[i] = val_loss
                    tst_losses[i] = tst_loss
                    res_dict["test_acc"].append(tst_acc[i]*100)
                continue
        else:
            #Remove true labels from the unlabeled dataset, the hypothesized labels are computed when select is called
            unlabeled_lake_set = LabeledToUnlabeledDataset(lake_set)
            strategy_sel.update_data(train_set, unlabeled_lake_set)
            strategy_sel.update_model(model)
            #compute the error log before every selection
            if(computeErrorLog):
                tst_err_log, val_err_log, val_class_err_idxs = find_err_per_class(test_set, val_set, final_val_classifications, final_val_predictions, final_tst_classifications, final_tst_predictions, all_logs_dir, sf+"_"+str(bud))
                csvlog.append([100-x for x in tst_err_log])
                val_csvlog.append([100-x for x in val_err_log])
            ####SIM####
            if(strategy=="SIM" or strategy=="SF"):
                if(sf.endswith("mi")):
                    if(feature=="classimb"):
                        #make a dataloader for the misclassifications - only for experiments with targets
                        for_query_set = getQuerySet(ConcatWithTargets(train_set, val_set),sel_cls_idx)
                        strategy_sel.update_queries(for_query_set)
                        
                        print('size of query set',len(for_query_set))
            #if SIM_WITHSOFT
            elif(strategy=="SIM_WITHSOFT"):
                if(sf.endswith("mi")):
                    if(feature=="classimb"):
                        #make a dataloader for the misclassifications - only for experiments with targets
                        for_query_set = getQuerySet(ConcatWithTargets(train_set, val_set),sel_cls_idx)
                        strategy_softsubset.update_queries(for_query_set)
                        strategy_sel.update_queries(for_query_set)
                        
                        print('size of query set',len(for_query_set))
            #if SCMI_WITHSOFT
            elif(strategy=="SCMI_WITHSOFT"):
                if(sf.endswith("mi")):
                    if(feature=="classimb"):
                        #make a dataloader for the misclassifications - only for experiments with targets
                        for_query_set = getQuerySet(ConcatWithTargets(train_set, val_set),sel_cls_idx)
                        for_private_set = getPrivateSet(ConcatWithTargets(train_set,val_set),sel_cls_idx)
                        strategy_softsubset.update_queries(for_query_set)
                        strategy_softsubset.update_privates(for_private_set)
                        strategy_sel.update_queries(for_query_set)
                        strategy_sel.update_privates(for_private_set)
                        
                        print('size of query set',len(for_query_set))
            #if AL_WITHSOFT
            elif(strategy=="AL_WITHSOFT"):
                if(sf=="glister-tss" or sf=="gradmatch-tss"):
                    for_query_set = getQuerySet(ConcatWithTargets(train_set,val_set),sel_cls_idx)
                    strategy_softsubset.update_queries(for_query_set)
                    strategy_sel.update_queries(for_query_set)
                    print('size of query set',len(for_query_set))
                else:
                    for_query_set = getQuerySet(ConcatWithTargets(train_set,val_set),sel_cls_idx)
                    strategy_softsubset.update_queries(for_query_set)
                    
                    print('size of query set',len(for_query_set))

            elif(strategy=="AL"):
                if(sf=="glister-tss" or sf=="gradmatch-tss"):
                    miscls_set = getQuerySet(ConcatWithTargets(train_set,val_set),sel_cls_idx)
                    strategy_sel.update_queries(miscls_set)
                    print("reinit AL with targeted miscls samples")
                
            elif(strategy=="WASSAL"):
                #concatina the train and val sets
                for_query_set = getQuerySet(ConcatWithTargets(train_set,val_set),sel_cls_idx)
                strategy_sel.update_queries(for_query_set)
                print('size of query set',len(for_query_set))
            elif(strategy=="WASSAL_P"):
                #concatina the train and val sets
                for_query_set = getQuerySet(ConcatWithTargets(train_set,val_set),sel_cls_idx)
                for_private_set=getPrivateSet(ConcatWithTargets(train_set,val_set),sel_cls_idx)
                strategy_sel.update_queries(for_query_set)
                strategy_sel.update_privates(for_private_set)
                print('size of query set',len(for_query_set))
            elif(strategy=="SCMI"):
                #concatina the train and val sets
                for_query_set = getQuerySet(ConcatWithTargets(train_set,val_set),sel_cls_idx)
                for_private_set=getPrivateSet(ConcatWithTargets(train_set,val_set),sel_cls_idx)
                
                strategy_sel.update_queries(for_query_set)
                strategy_sel.update_privates(for_private_set)
                
                print('size of query set',len(for_query_set))

            
            if(strategy=="WASSAL"):
                subset,classwise_final_indices_simplex = strategy_sel.select(budget)
                temp_args={}
                temp_args['device'] = device
                temp_args['target'] = sel_cls_idx
                #analyze_simplex(temp_args,lake_set,simplex_query)
            elif(strategy=="WASSAL_P"):
                subset,simplex_query,simplex_private = strategy_sel.select(budget)
                temp_args={}
                temp_args['device'] = device
                temp_args['target'] = sel_cls_idx
                #analyze_simplex(temp_args,lake_set,simplex_query)
            elif(strategy=="SCMI_WITHSOFT"):
                softsubset,soft_simplex_query,soft_simplex_private = strategy_softsubset.select(budget)
                subset = strategy_sel.select(budget)
            elif(strategy=="SMI_WITHSOFT"or strategy=="AL_WITHSOFT"):
                classwise_final_indices_simplex = strategy_softsubset.select(budget)
                subset = strategy_sel.select(budget)
            else:
                subset = strategy_sel.select(budget)

            print("#### Selection Complete, Now re-training with augmented subset ####")
            if(visualize_tsne):
                tsne_plt = tsne_smi(strategy_sel.unlabeled_data_embedding.cpu(),
                                    lake_set.targets,
                                    strategy_sel.query_embedding.cpu(),
                                    sel_cls_idx,
                                    subset)
                print("Computed TSNE plot of the selection")
            lake_subset_idxs = subset #indices wrt to lake that need to be removed from the lake
            perClsSel = getPerClassSel(true_lake_set, lake_subset_idxs, num_cls)
            res_dict['sel_per_cls'].append(perClsSel)

            #if SCMI_WITHSOFT do a softsubsetted training before AL training
            if(strategy=="SCMI_WITHSOFT"):
                print('Softsubsetting for SCMI functions  using WASSAL_P')
                # Extract images and targets from weighted_lake_set
                images = [lake_set[i][0] for i in range(len(lake_set))]
                targets=torch.tensor(sel_cls_idx[0])
                targets=targets.repeat(len(lake_set)//len(sel_cls_idx))
                private_targets=torch.tensor(0)
                private_targets=private_targets.repeat(len(lake_set)//len(sel_cls_idx))
                sofftsimplex_query=soft_simplex_query.detach().cpu().numpy()
                sofftsimplex_private=soft_simplex_private.detach().cpu().numpy()
                 #choose the top simplex_query that contributes 30% to the total simplex_query
                _,top_n_indices=top_elements_contribute_to_percentage(sofftsimplex_query, ss_max_budget)
                _,top_n_private_indices=top_elements_contribute_to_percentage(sofftsimplex_private, ss_max_budget)
                small_images = [images[i] for i in top_n_indices]
                small_targets = targets[top_n_indices.copy()]
                small_simplex_query = sofftsimplex_query[top_n_indices]
                small_private_targets = private_targets[top_n_private_indices.copy()]
                small_simplex_private = sofftsimplex_private[top_n_private_indices]
                weighted_lake_set = WeightedDataset(small_images,small_targets, small_simplex_query,small_private_targets,small_simplex_private)


                #load weighted_lakset into a weighted dataloader
                weighted_lakeloader = torch.utils.data.DataLoader(weighted_lake_set, batch_size=trn_batch_size,
                                                shuffle=False, pin_memory=True)
                
                #start training
                print("starting weighted training for {} with hypothesized labels:"+str(sel_cls_idx),strategy)
                start_time = time.time()
                num_ep=1
                while(full_trn_acc[i]<0.99 and num_ep<100):
                    model.train()
                    for batch_idx, (inputs, targets,simplex_query,private_targets,simplex_private) in enumerate(weighted_lakeloader):
                        # Variables in Pytorch are differentiable.
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        private_targets = private_targets.to(device)
                        simplex_query=simplex_query.to(device)
                        simplex_private=simplex_private.to(device)
                        # This will zero out the gradients for this batch.
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        target_loss_per_sample=criterion(outputs, targets)
                        private_loss_per_sample=criterion(outputs, private_targets)
                        loss = (simplex_query*target_loss_per_sample).sum()+(simplex_private*private_loss_per_sample).sum()
                        loss.backward()
                        optimizer.step()


                    full_trn_loss = 0
                    full_trn_correct = 0
                    full_trn_total = 0
                    model.eval()
                    with torch.no_grad():
                        for batch_idx, (inputs, targets,simplex_query,private_targets,simplex_private) in enumerate(weighted_lakeloader):
                            # Variables in Pytorch are differentiable.
                            inputs = inputs.to(device)
                            targets = targets.to(device)
                            private_targets = private_targets.to(device)
                            simplex_query=simplex_query.to(device)
                            simplex_private=simplex_private.to(device)
                            # This will zero out the gradients for this batch.
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            target_loss_per_sample=criterion(outputs, targets)
                            private_loss_per_sample=criterion(outputs, private_targets)
                            loss = (simplex_query*target_loss_per_sample).sum()+(simplex_private*private_loss_per_sample).sum()
                            full_trn_loss += loss.item()
                            _, predicted = outputs.max(1)
                            full_trn_total += targets.size(0)
                            full_trn_correct += predicted.eq(targets).sum().item()
                        full_trn_acc[i] = full_trn_correct / full_trn_total
                        print("Selection Epoch ", i, " Training epoch [" , num_ep, "]" , " Training Acc: ", full_trn_acc[i], end="\r")
                        num_ep+=1
                    timing[i] = time.time() - start_time

            if(strategy=="SMI_WITHSOFT" or strategy=="AL_WITHSOFT"):
                print('Softsubsetting for SMI or any other AL_SOFT functions using WASSAL')
                # Aggregation lists
                all_small_images = []
                all_small_targets = []
                all_small_simplex_query = []
                
                for (sel_cls_idx, simplex_query, class_idx) in classwise_final_indices_simplex:

                    # Extract images and targets from weighted_lake_set
                    images = [lake_set[i][0] for i in range(len(lake_set))]
                    targets=torch.tensor([class_idx])
                    targets=targets.repeat(len(lake_set))
                    sofftsimplex_query=simplex_query.detach().cpu().numpy()
                    #choose the top simplex_query that contributes 30% to the total simplex_query
                    _,top_n_indices=top_elements_contribute_to_percentage(sofftsimplex_query, ss_max_budget)
                    # Collect the data
                    all_small_images.extend([images[i] for i in top_n_indices])
                    all_small_targets.extend(targets[top_n_indices.copy()].tolist())
                    all_small_simplex_query.extend(sofftsimplex_query[top_n_indices].tolist())

                # Convert lists to tensors
                all_small_targets = torch.tensor(all_small_targets)
                all_small_simplex_query = torch.tensor(all_small_simplex_query)

                # Form the combined weighted dataset
                weighted_lake_set = WeightedDataset(all_small_images, all_small_targets, all_small_simplex_query, None, None)

                # Load into a dataloader
                weighted_lakeloader = torch.utils.data.DataLoader(weighted_lake_set, batch_size=trn_batch_size, shuffle=True, pin_memory=True)

                
                
                #start training
                print("starting weighted training for {} with hypothesized labels:"+str(sel_cls_idx),strategy)
                start_time = time.time()
                num_ep=1
                while(full_trn_acc[i]<0.99 and num_ep<100):
                    model.train()
                    for batch_idx, (inputs, targets,simplex_query,_,_) in enumerate(weighted_lakeloader):
                        # Variables in Pytorch are differentiable.
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        
                        simplex_query=simplex_query.to(device)
                        # This will zero out the gradients for this batch.
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        target_loss_per_sample=criterion(outputs, targets)
                        loss = (simplex_query*target_loss_per_sample).sum()
                        loss.backward()
                        optimizer.step()
                    full_trn_loss = 0
                    full_trn_correct = 0
                    full_trn_total = 0
                    model.eval()
                    with torch.no_grad():
                        for batch_idx, (inputs, targets,simplex_query,_,_) in enumerate(weighted_lakeloader):
                            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            full_trn_loss += loss.item()
                            _, predicted = outputs.max(1)
                            full_trn_total += targets.size(0)
                            full_trn_correct += predicted.eq(targets).sum().item()
                        full_trn_acc[i] = full_trn_correct / full_trn_total
                        print("Selection Epoch ", i, " Training epoch [" , num_ep, "]" , " Training Acc: ", full_trn_acc[i], end="\r")
                        num_ep+=1
                    timing[i] = time.time() - start_time

            


            #if WASSAL do a softsubsetted training before AL training
            if(strategy=="WASSAL"):
            #label all the samples in the lake with the hypothesized labels of target classes and do weighted training of simplex_query
                
                # Extract images and targets from weighted_lake_set
                images = [lake_set[i][0] for i in range(len(lake_set))]
                targets=torch.tensor(sel_cls_idx[0])
                targets=targets.repeat(len(lake_set)//len(sel_cls_idx))
                #just take the top 10 based on sorted simplex_query
                simplex_query=simplex_query.detach().cpu().numpy()
                #choose the top simplex_query that contributes 30% to the total simplex_query

                _,top_n_indices=top_elements_contribute_to_percentage(simplex_query, ss_max_budget)
                
                # # Get the indices that would sort the array in descending order
                # sorted_indices = simplex_query.argsort()[::-1]
                # # Extract the top n indices
                # #top_n_indices = sorted_indices[budget+1:n*2]
                # top_n_indices = sorted_indices[:softSubbudget]
                # # Reorder images and targets based on these indices
                small_images = [images[i] for i in top_n_indices]
                small_targets = targets[top_n_indices.copy()]
                # Update simplex_query to only contain top n values
                small_simplex_query = simplex_query[top_n_indices]

                weighted_lake_set = WeightedDataset(small_images,small_targets, small_simplex_query,None,None)

                #load weighted_lakset into a weighted dataloader
                weighted_lakeloader = torch.utils.data.DataLoader(weighted_lake_set, batch_size=trn_batch_size,
                                                shuffle=False, pin_memory=True)
                
                #start training
                print("starting weighted training for WASSAL with hypothesized labels:"+str(sel_cls_idx))
                start_time = time.time()
                num_ep=1
                while(full_trn_acc[i]<0.99 and num_ep<100):
                    model.train()
                    for batch_idx, (inputs, targets,simplex_query,_,_) in enumerate(weighted_lakeloader):
                        # Variables in Pytorch are differentiable.
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        
                        simplex_query=simplex_query.to(device)
                        # This will zero out the gradients for this batch.
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        target_loss_per_sample=criterion(outputs, targets)
                        loss = (simplex_query*target_loss_per_sample).sum()
                        loss.backward()
                        optimizer.step()
                    full_trn_loss = 0
                    full_trn_correct = 0
                    full_trn_total = 0
                    model.eval()
                    with torch.no_grad():
                        for batch_idx, (inputs, targets,simplex_query,_,_) in enumerate(weighted_lakeloader):
                            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            full_trn_loss += loss.item()
                            _, predicted = outputs.max(1)
                            full_trn_total += targets.size(0)
                            full_trn_correct += predicted.eq(targets).sum().item()
                        full_trn_acc[i] = full_trn_correct / full_trn_total
                        print("Selection Epoch ", i, " Training epoch [" , num_ep, "]" , " Training Acc: ", full_trn_acc[i], end="\r")
                        num_ep+=1
                    timing[i] = time.time() - start_time
            #if WASSAL_P   do a softsubsetted training with query and private before AL training
            elif(strategy=="WASSAL_P"):
            #label all the samples in the lake with the hypothesized labels of target classes and do weighted training of simplex_query
                softSubbudget=10
                # Extract images and targets from weighted_lake_set
                images = [lake_set[i][0] for i in range(len(lake_set))]
                targets=torch.tensor(sel_cls_idx[0])
                targets=targets.repeat(len(lake_set)//len(sel_cls_idx))
                private_targets=torch.tensor(0)
                private_targets=private_targets.repeat(len(lake_set)//len(sel_cls_idx))

                #just take the top 10 based on sorted simplex_query
                simplex_query=simplex_query.detach().cpu().numpy()
                _,top_n_indices=top_elements_contribute_to_percentage(simplex_query, ss_max_budget)
                simplex_private=simplex_private.detach().cpu().numpy()
                _,top_n_private_indices=top_elements_contribute_to_percentage(simplex_private, ss_max_budget)
                # # Get the indices that would sort the array in descending order
                # sorted_indices = simplex_query.argsort()[::-1]
                # # Extract the top n indices
                # #top_n_indices = sorted_indices[budget+1:n*2]
                # top_n_indices = sorted_indices[:softSubbudget]
                # Reorder images and targets based on these indices
                small_images = [images[i] for i in top_n_indices]
                small_targets = targets[top_n_indices.copy()]
                small_private_targets = private_targets[top_n_private_indices.copy()]
                # Update simplex_query to only contain top n values
                small_simplex_query = simplex_query[top_n_indices]
                small_simplex_private = simplex_private[top_n_private_indices]

                weighted_lake_set = WeightedDataset(small_images,small_targets, small_simplex_query,small_private_targets,small_simplex_private)


                #load weighted_lakset into a weighted dataloader
                weighted_lakeloader = torch.utils.data.DataLoader(weighted_lake_set, batch_size=trn_batch_size,
                                                shuffle=False, pin_memory=True)
                
                #start training
                print("starting weighted training for WASSAL_P with hypothesized labels:"+str(sel_cls_idx))
                start_time = time.time()
                num_ep=1
                while(full_trn_acc[i]<0.99 and num_ep<100):
                    model.train()
                    for batch_idx, (inputs, targets,simplex_query,private_targets,simplex_private) in enumerate(weighted_lakeloader):
                        # Variables in Pytorch are differentiable.
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        private_targets = private_targets.to(device)
                        simplex_query=simplex_query.to(device)
                        simplex_private=simplex_private.to(device)
                        # This will zero out the gradients for this batch.
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        target_loss_per_sample=criterion(outputs, targets)
                        private_loss_per_sample=criterion(outputs, private_targets)
                        loss = (simplex_query*target_loss_per_sample).sum()+(simplex_private*private_loss_per_sample).sum()
                        loss.backward()
                        optimizer.step()


                    full_trn_loss = 0
                    full_trn_correct = 0
                    full_trn_total = 0
                    model.eval()
                    with torch.no_grad():
                        for batch_idx, (inputs, targets,simplex_query,private_targets,simplex_private) in enumerate(weighted_lakeloader):
                            # Variables in Pytorch are differentiable.
                            inputs = inputs.to(device)
                            targets = targets.to(device)
                            private_targets = private_targets.to(device)
                            simplex_query=simplex_query.to(device)
                            simplex_private=simplex_private.to(device)
                            # This will zero out the gradients for this batch.
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            target_loss_per_sample=criterion(outputs, targets)
                            private_loss_per_sample=criterion(outputs, private_targets)
                            loss = (simplex_query*target_loss_per_sample).sum()+(simplex_private*private_loss_per_sample).sum()
                            full_trn_loss += loss.item()
                            _, predicted = outputs.max(1)
                            full_trn_total += targets.size(0)
                            full_trn_correct += predicted.eq(targets).sum().item()
                        full_trn_acc[i] = full_trn_correct / full_trn_total
                        print("Selection Epoch ", i, " Training epoch [" , num_ep, "]" , " Training Acc: ", full_trn_acc[i], end="\r")
                        num_ep+=1
                    timing[i] = time.time() - start_time
        

            


                    
            print("starting AL training")
            #augment the train_set with selected indices from the lake
            train_set, lake_set, true_lake_set, add_val_set = aug_train_subset(train_set, lake_set, true_lake_set, subset, lake_subset_idxs, budget, True) #aug train with random if budget is not filled
            print("After augmentation, size of train_set: ", len(train_set), " unlabeled set: ", len(lake_set), " val set: ", len(val_set))
    
#           Reinit train and lake loaders with new splits and reinit the model
            trainloader = torch.utils.data.DataLoader(train_set, batch_size=trn_batch_size, shuffle=True, pin_memory=True)
            lakeloader = torch.utils.data.DataLoader(lake_set, batch_size=tst_batch_size, shuffle=False, pin_memory=True)
            #model = create_model(model_name, num_cls, device, strategy_args['embedding_type'])
            #optimizer = optimizer_without_scheduler(model, learning_rate)
            

        #Start training
        start_time = time.time()
        num_ep=1
#         while(num_ep<150):
        #first train until full training accuracy is 0.99
        while(full_trn_acc[i]<0.99 and num_ep<100):
            model.train()
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                # Variables in Pytorch are differentiable.
                inputs, target = Variable(inputs), Variable(inputs)
                # This will zero out the gradients for this batch.
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
#             scheduler.step()
          
            full_trn_loss = 0
            full_trn_correct = 0
            full_trn_total = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(trainloader): #Compute Train accuracy
                    inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    full_trn_loss += loss.item()
                    _, predicted = outputs.max(1)
                    full_trn_total += targets.size(0)
                    full_trn_correct += predicted.eq(targets).sum().item()
                full_trn_acc[i] = full_trn_correct / full_trn_total
                print("Selection Epoch ", i, " Training epoch [" , num_ep, "]" , " Training Acc: ", full_trn_acc[i], end="\r")
                num_ep+=1
            timing[i] = time.time() - start_time
        with torch.no_grad():
            final_val_predictions = []
            final_val_classifications = []
            for batch_idx, (inputs, targets) in enumerate(valloader): #Compute Val accuracy
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                final_val_predictions += list(predicted.cpu().numpy())
                final_val_classifications += list(predicted.eq(targets).cpu().numpy())

            final_tst_predictions = []
            final_tst_classifications = []
            for batch_idx, (inputs, targets) in enumerate(tstloader): #Compute test accuracy
                inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                tst_loss += loss.item()
                _, predicted = outputs.max(1)
                tst_total += targets.size(0)
                tst_correct += predicted.eq(targets).sum().item()
                final_tst_predictions += list(predicted.cpu().numpy())
                final_tst_classifications += list(predicted.eq(targets).cpu().numpy())                
            val_acc[i] = val_correct / val_total
            tst_acc[i] = tst_correct / tst_total
            val_losses[i] = val_loss
            fulltrn_losses[i] = full_trn_loss
            tst_losses[i] = tst_loss
            full_val_acc = list(np.array(val_acc))
            full_timing = list(np.array(timing))
            res_dict["test_acc"].append(tst_acc[i]*100)
            print('Epoch:', i + 1, 'FullTrn,TrainAcc,ValLoss,ValAcc,TstLoss,TstAcc,Time:', full_trn_loss, full_trn_acc[i], val_loss, val_acc[i], tst_loss, tst_acc[i], timing[i])
            print("Gain in accuracy: ",res_dict['test_acc'][i]-res_dict['test_acc'][i-1])
        if(i==0): 
            print("Saving initial model") 
            torch.save(model.state_dict(), initModelPath) #save initial train model if not present
            
    #Compute the statistics of the final model
    if(computeErrorLog):
        print("**** Final Metrics after Targeted Learning ****")
        tst_err_log, val_err_log, val_class_err_idxs = find_err_per_class(test_set, val_set, final_val_classifications, final_val_predictions, final_tst_classifications, final_tst_predictions, all_logs_dir, sf+"_"+str(bud))
        csvlog.append([100-x for x in tst_err_log])
        val_csvlog.append([100-x for x in val_err_log])
        res_dict["all_class_acc"] = csvlog
        res_dict["all_val_class_acc"] = val_csvlog
        with open(os.path.join(all_logs_dir, exp_name+".csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerows(csvlog)
    #save results dir with test acc and per class selections
    with open(os.path.join(all_logs_dir, exp_name+".json"), 'w') as fp:
        json.dump(res_dict, fp)
    #Print overall acc improvement and rare class acc improvement, show that TL selected relevant points in space, is possible show some images
    print_final_results(res_dict, sel_cls_idx)
    print("Total gain in accuracy: ",res_dict['test_acc'][i]-res_dict['test_acc'][0])
    
#     tsne_plt.show()
    




# %%
experiments=['exp1','exp2','exp3','exp4','exp5']
seeds=[42,43,44,45,46]
budgets=[5,10,15,20,25]
device_id = 0
device = "cuda:"+str(device_id) if torch.cuda.is_available() else "cpu"

# embedding_type = "features" #Type of the representation to use (gradients/features)
# model_name = 'ResNet18' #Model to use for training
# initModelPath = "/home/wassal/trust-wassal/tutorials/results/"+data_name + "_" + model_name+"_"+embedding_type + "_" + str(learning_rate) + "_" + str(split_cfg["sel_cls_idx"])
#  # Model Creation
# model = create_model(model_name, num_cls, device, embedding_type)
# #List of strategies
# strategies = [
    
    
#     ("WASSAL", "WASSAL"),
#     ("WASSAL_P", "WASSAL_P"),
#     ("SIM", 'fl1mi'),
#     ("SIM", 'fl2mi'),
#     ("SIM", 'gcmi'),
#     ("SIM", 'logdetmi'),
#     ('SCMI', 'flcmi'),
#     ('SCMI', 'logdetcmi'),
#     ("SIM_WITHSOFT", 'fl1mi_soft'),
#     ("SIM_WITHSOFT", 'fl2mi_soft'),
#     ("SIM_WITHSOFT", 'gcmi_soft'),
#     ("SIM_WITHSOFT", 'logdetmi_soft'),
#     ('SCMI_WITHSOFT', 'flcmi_soft'),
#     ('SCMI_WITHSOFT', 'logdetcmi_soft'),
#     ("random", 'random'),
    
# ]

# for i,experiment in enumerate(experiments):
#     seed=seeds[i]
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     run=experiment
    
#     # Loop for each budget from 50 to 400 in intervals of 50
#     for b in budgets:
#         # Loop through each strategy
#         for strategy, method in strategies:
#             print("Budget ",b," Strategy ",strategy," Method ",method)
#             run_targeted_selection(data_name, 
#                                     datadir, 
#                                     feature, 
#                                     model_name, 
#                                     b,             # updated budget
#                                     split_cfg, 
#                                     learning_rate, 
#                                     run, 
#                                     device, 
#                                     computeClassErrorLog,
#                                     strategy, 
#                                     method)


embedding_type = "gradients" #Type of the representation to use (gradients/features)
model_name = 'ResNet18' #Model to use for training
initModelPath = "/home/wassal/trust-wassal/tutorials/results/"+data_name + "_" + model_name+"_"+embedding_type + "_" + str(learning_rate) + "_" + str(split_cfg["sel_cls_idx"])
 # Model Creation
model = create_model(model_name, num_cls, device, embedding_type)
strategies = [
    
     #al soft
    #("AL_WITHSOFT", "badge"),
    ("AL_WITHSOFT", 'us_soft'),
    ("AL_WITHSOFT", "glister_soft"),
    ("AL_WITHSOFT", 'gradmatch-tss_soft'),
    ("AL_WITHSOFT", 'coreset_soft'),
    ("AL_WITHSOFT", 'leastconf_soft'),
    ("AL_WITHSOFT", 'margin_soft'),
    ("random", 'random'),

    ("AL", "badge"),
    ("AL", 'us'),
    ("AL", "glister"),
    ("AL", 'gradmatch-tss'),
    ("AL", 'coreset'),
    ("AL", 'leastconf'),
    ("AL", 'margin'),
   
    
    
    
]

for i,experiment in enumerate(experiments):
    seed=seeds[i]
    torch.manual_seed(seed)
    np.random.seed(seed)
    run=experiment

    # Loop for each budget from 50 to 400 in intervals of 50
    for b in budgets:
        # Loop through each strategy
        for strategy, method in strategies:
            print("Budget ",b," Strategy ",strategy," Method ",method)
            run_targeted_selection(data_name, 
                                    datadir, 
                                    feature, 
                                    model_name, 
                                    b,             # updated budget
                                    split_cfg, 
                                    learning_rate, 
                                    run, 
                                    device, 
                                    computeClassErrorLog,
                                    strategy, 
                                    method,embedding_type)
