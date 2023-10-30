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

sys.path.append("/home/wassal/trust-wassal/")

from trust.utils.models.resnet import ResNet18
from trust.utils.models.resnet import ResNet50
from trust.utils.custom_dataset_medmnist import load_biodataset_custom
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

sys.path.append("/home/wassal/distil")
from distil.active_learning_strategies.entropy_sampling import EntropySampling
from distil.active_learning_strategies.badge import BADGE
from distil.active_learning_strategies.glister import GLISTER
from distil.active_learning_strategies.gradmatch_active import GradMatchActive
from distil.active_learning_strategies.core_set import CoreSet
from distil.active_learning_strategies.least_confidence_sampling import (
    LeastConfidenceSampling,
)
from distil.active_learning_strategies.margin_sampling import MarginSampling

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
from trust.utils.utils import *
from trust.utils.viz import tsne_smi
import math
from random import shuffle

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
    if name == "ResNet18":
        if embedding_type == "gradients":
            model = ResNet18(num_cls)
        else:
            model = ResNet18(num_cls)
            # model = models.resnet18()
            # model.fc = nn.Linear(512, num_cls)
    elif name == "ResNet50":
        if embedding_type == "gradients":
            model = ResNet50(num_cls)
        else:
            model = models.resnet50()
    elif name == "MnistNet":
        model = MnistNet()
    elif name == "ResNet164":
        model = ResNet164(num_cls)
    model.apply(init_weights)
    model = model.to(device)
    return model


def loss_function():
    criterion = nn.CrossEntropyLoss()
    criterion_nored = nn.CrossEntropyLoss(reduction="none")
    return criterion, criterion_nored


def optimizer_with_scheduler(model, num_epochs, learning_rate, m=0.9, wd=5e-4):
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=m, weight_decay=wd
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    return optimizer, scheduler


def optimizer_without_scheduler(model, learning_rate, m=0.9, wd=5e-4):
    #     optimizer = optim.Adam(model.parameters(),weight_decay=wd)
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=m, weight_decay=wd
    )
    return optimizer


def generate_cumulative_timing(mod_timing):
    tmp = 0
    mod_cum_timing = np.zeros(len(mod_timing))
    for i in range(len(mod_timing)):
        tmp += mod_timing[i]
        mod_cum_timing[i] = tmp
    return mod_cum_timing / 3600


def displayTable(val_err_log, tst_err_log):
    col1 = [str(i) for i in range(10)]
    val_acc = [str(100 - i) for i in val_err_log]
    tst_acc = [str(100 - i) for i in tst_err_log]
    table = [col1, val_acc, tst_acc]
    table = map(list, zip(*table))
    print(
        tabulate(
            table, headers=["Class", "Val Accuracy", "Test Accuracy"], tablefmt="orgtbl"
        )
    )


def find_err_per_class(
    test_set,
    val_set,
    final_val_classifications,
    final_val_predictions,
    final_tst_classifications,
    final_tst_predictions,
    saveDir,
    prefix,
    doIdisplayTable=True,
):
    val_err_idx = list(np.where(np.array(final_val_classifications) == False)[0])
    tst_err_idx = list(np.where(np.array(final_tst_classifications) == False)[0])
    val_class_err_idxs = []
    tst_err_log = []
    val_err_log = []
    for i in range(num_cls):
        tst_class_idxs = list(
            torch.where(torch.Tensor(test_set.targets) == i)[0].cpu().numpy()
        )
        val_class_idxs = list(
            torch.where(torch.Tensor(val_set.targets.float()) == i)[0].cpu().numpy()
        )
        # err classifications per class
        val_err_class_idx = set(val_err_idx).intersection(set(val_class_idxs))
        tst_err_class_idx = set(tst_err_idx).intersection(set(tst_class_idxs))
        if len(val_class_idxs) > 0:
            val_error_perc = round(
                (len(val_err_class_idx) / len(val_class_idxs)) * 100, 2
            )
        else:
            val_error_perc = 0
        tst_error_perc = round((len(tst_err_class_idx) / len(tst_class_idxs)) * 100, 2)
        #         print("val, test error% for class ", i, " : ", val_error_perc, tst_error_perc)
        val_class_err_idxs.append(val_err_class_idx)
        tst_err_log.append(tst_error_perc)
        val_err_log.append(val_error_perc)

    if doIdisplayTable:
        displayTable(val_err_log, tst_err_log)

    tst_err_log.append(sum(tst_err_log) / len(tst_err_log))
    val_err_log.append(sum(val_err_log) / len(val_err_log))
    return tst_err_log, val_err_log, val_class_err_idxs


def aug_train_subset(
    train_set,
    lake_set,
    true_lake_set,
    subset,
    lake_subset_idxs,
    budget,
    augrandom=False,
):
    all_lake_idx = list(range(len(lake_set)))
    if not (len(subset) == budget) and augrandom:
        print(
            "Budget not filled, adding ", str(int(budget) - len(subset)), " randomly."
        )
        remain_budget = int(budget) - len(subset)
        remain_lake_idx = list(set(all_lake_idx) - set(subset))
        random_subset_idx = list(
            np.random.choice(
                np.array(remain_lake_idx), size=int(remain_budget), replace=False
            )
        )
        subset += random_subset_idx
    if str(type(true_lake_set.targets)) == "<class 'numpy.ndarray'>":
        lake_ss = SubsetWithTargets(
            true_lake_set,
            subset,
            torch.Tensor(true_lake_set.targets.astype(np.float))[subset],
        )
    else:
        lake_ss = SubsetWithTargets(
            true_lake_set, subset, torch.Tensor(true_lake_set.targets.float())[subset]
        )
    remain_lake_idx = list(set(all_lake_idx) - set(lake_subset_idxs))
    if str(type(true_lake_set.targets)) == "<class 'numpy.ndarray'>":
        remain_lake_set = SubsetWithTargets(
            lake_set,
            remain_lake_idx,
            torch.Tensor(lake_set.targets.astype(np.float))[remain_lake_idx],
        )
    else:
        remain_lake_set = SubsetWithTargets(
            lake_set,
            remain_lake_idx,
            torch.Tensor(lake_set.targets.float())[remain_lake_idx],
        )
    if str(type(true_lake_set.targets)) == "<class 'numpy.ndarray'>":
        remain_true_lake_set = SubsetWithTargets(
            true_lake_set,
            remain_lake_idx,
            torch.Tensor(true_lake_set.targets.astype(np.float))[remain_lake_idx],
        )
    else:
        remain_true_lake_set = SubsetWithTargets(
            true_lake_set,
            remain_lake_idx,
            torch.Tensor(true_lake_set.targets.float())[remain_lake_idx],
        )
    #print(len(lake_ss), len(remain_lake_set), len(lake_set))
    aug_train_set = ConcatWithTargets(train_set, lake_ss)
    aug_trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=1000, shuffle=True, pin_memory=True
    )
    return aug_train_set, remain_lake_set, remain_true_lake_set, lake_ss


def get_balanced_samples(class_samples, n_samples):
    shuffle(class_samples)
    return class_samples[:n_samples]


def penalized_samples_size(total_samples, proportion):
    """Calculate penalized sample size based on proportion."""
    return int(total_samples * (1 - proportion))


def getQuerySet(val_set, imb_cls_idx, recipe="asis"):
    targets_tensor = torch.Tensor(val_set.targets).float()

    if recipe == "asis":
        miscls_idx = []
        for i in imb_cls_idx:
            imb_cls_samples = list(torch.where(targets_tensor == i)[0].cpu().numpy())
            miscls_idx += imb_cls_samples
        print(
            "Total samples from imbalanced classes as Queries (Size of query set): ",
            len(miscls_idx),
        )
        return SubsetWithTargets(val_set, miscls_idx, val_set.targets[miscls_idx])

    elif recipe == "balanced":
        n_samples_per_class = []

        for i in imb_cls_idx:
            imb_cls_samples = list(torch.where(targets_tensor == i)[0].cpu().numpy())
            n_samples_per_class.append(len(imb_cls_samples))

        n_samples = min(n_samples_per_class)
        miscls_idx = []

        for i in imb_cls_idx:
            imb_cls_samples = list(torch.where(targets_tensor == i)[0].cpu().numpy())
            miscls_idx += get_balanced_samples(imb_cls_samples, n_samples)

        print(
            "Total balanced samples from imbalanced classes as Queries (Size of query set): ",
            len(miscls_idx),
        )

        return SubsetWithTargets(val_set, miscls_idx, val_set.targets[miscls_idx])

    elif recipe == "penalized":
        class_counts = {i: 0 for i in imb_cls_idx}
        total_samples = len(targets_tensor)

        # Calculate the number of samples for each class in imb_cls_idx
        for i in imb_cls_idx:
            class_counts[i] = torch.sum(targets_tensor == i).item()

        # Calculate proportion for each class
        class_proportions = {
            i: count / total_samples for i, count in class_counts.items()
        }

        miscls_idx = []
        for i in imb_cls_idx:
            imb_cls_samples = list(torch.where(targets_tensor == i)[0].cpu().numpy())
            penalized_size = penalized_samples_size(
                len(imb_cls_samples), class_proportions[i]
            )
            miscls_idx += get_balanced_samples(imb_cls_samples, penalized_size)
            print(
                "Query selection for class {} with penalized size {}".format(
                    i, penalized_size
                )
            )
        print(
            "Total penalized samples from imbalanced classes as Queries (Size of query set): ",
            len(miscls_idx),
        )

        return SubsetWithTargets(val_set, miscls_idx, val_set.targets[miscls_idx])

    else:
        raise ValueError(f"Unknown recipe type: {recipe}")


def getPrivateSet(val_set, imb_cls_idx):
    # Get all the indices in the val_set
    all_idx = list(range(len(val_set.targets)))
    miscls_idx = []

    for i in imb_cls_idx:
        imb_cls_samples = list(
            torch.where(torch.Tensor(val_set.targets.float()) == i)[0].cpu().numpy()
        )
        miscls_idx += imb_cls_samples
    # Get indices that aren't in the query class samples
    private_idx = list(set(all_idx) - set(miscls_idx))
    print(
        "Total samples from imbalanced classes as Private (Size of private set): ",
        len(private_idx),
    )
    return SubsetWithTargets(val_set, private_idx, val_set.targets[private_idx])


def getHigheestClassNumber(trainset):
    targets_tensor = torch.Tensor(trainset.targets).float()
    class_counts = {}
    for i in range(num_cls):
        class_counts[i] = torch.sum(targets_tensor == i).item()
    return max(class_counts.values())


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


def plotsimpelxDistribution(lake_set, classwise_final_indices_simplex,folder_name):
    # Plot the distribution of the simplex query colorcoded based odn the true labels
    for simplex_query, simplex_refrain, class_idx in classwise_final_indices_simplex:
        # Determine histogram bin edges
        counts, bin_edges = np.histogram(simplex_query, bins=10)

        # Create a color map for your targets
        unique_targets = np.unique(lake_set.targets)
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_targets)))
        color_map = {target: color for target, color in zip(unique_targets, colors)}

        # Prepare bin data to color based on target
        bin_data = {target: [] for target in unique_targets}

        for i in range(len(bin_edges) - 1):
            bin_mask = (simplex_query >= bin_edges[i]) & (
                simplex_query < bin_edges[i + 1]
            )
            bin_targets = np.array(lake_set.targets)[bin_mask]
            for target in unique_targets:
                count_target = np.sum(bin_targets == target)
                # Add the count to the bin_data if it's less than or equal to 100
                if count_target <= 200:
                    bin_data[target].append(count_target)
                else:
                    bin_data[target].append(0)

        # Plot
        plt.figure(figsize=(10, 5))
        bottom = np.zeros(len(bin_edges) - 1)
        for target, counts in bin_data.items():
            plt.bar(
                bin_edges[:-1],
                counts,
                width=np.diff(bin_edges),
                align="edge",
                label=str(target),
                bottom=bottom,
                color=color_map[target],
            )
            bottom += counts

        plt.title(
            "Distributions of the simplex query for hypothesised Class: "
            + str(class_idx)
        )
        plt.xlabel("Query values")
        plt.ylabel("Frequency")
        plt.legend(title="Targets", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name,"cifar10_simplex_distribution_class_{}.png".format(class_idx)))
        plt.close()


def print_final_results(res_dict, sel_cls_idx):
    print(
        "Gain in overall test accuracy: ",
        res_dict["test_acc"][1] - res_dict["test_acc"][0],
    )
    # bf_sel_cls_acc = np.array(res_dict['all_class_acc'][0])[sel_cls_idx]
    # af_sel_cls_acc = np.array(res_dict['all_class_acc'][1])[sel_cls_idx]
    # print("Gain in targeted test accuracy: ", np.mean(af_sel_cls_acc-bf_sel_cls_acc))


def analyze_simplex(args, unlabeled_set, simplex_query):
    print("======== analysis on simplex =========")
    unlabeled_loader = torch.utils.data.DataLoader(
        dataset=unlabeled_set, batch_size=len(unlabeled_set), shuffle=False
    )
    u_imgs, u_labels = next(iter(unlabeled_loader))
    u_imgs, u_labels = u_imgs.to(args["device"]), u_labels.to(args["device"])
    nz_query_idx = simplex_query.nonzero()

    # Using a loop to accommodate an array of target values
    total_correctly_identified = 0
    for query_value in args["target"]:
        num_nz_query = (u_labels[nz_query_idx] == query_value).nonzero().shape[0]
        total_correctly_identified += num_nz_query
    print(
        "no of query labels identified correctly: {}/{}".format(
            total_correctly_identified, nz_query_idx.shape[0]
        )
    )

    total_query_weight = 0
    for query_value in args["target"]:
        query_idx = torch.where(u_labels == query_value)
        query_weight = torch.sum(simplex_query[query_idx])
        total_query_weight += query_weight
    print("Weight of Query samples in simplex_query: {}".format(total_query_weight))


class WeightedDataset(Dataset):
    def __init__(self, imgs, targets, simplex_query, private_targets, simplex_private):
        self.imgs = imgs
        self.targets = targets
        self.simplex_query = simplex_query
        self.private_targets = private_targets
        self.simplex_private = simplex_private

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.imgs[idx]
        target = self.targets[idx]
        t = self.simplex_query[idx].item()
        private_target = (
            self.private_targets[idx] if self.private_targets is not None else []
        )
        p = self.simplex_private[idx].item() if self.simplex_private is not None else []

        return (image, target, t, private_target, p)


# return the elements from the simplex_query that contribute to the given percentage
def top_elements_contribute_to_percentage(simplex_query, n_percent, budget):
    # Pair each value with its original index
    indexed_simplex = list(enumerate(simplex_query))

    # Sort based on the value (in descending order)
    sorted_simplex = sorted(indexed_simplex, key=lambda x: x[1], reverse=True)

    # Calculate the total sum of the array
    total_sum = sum(value for index, value in sorted_simplex)

    # If the array doesn't sum up to 1, you might want to handle this case
    if total_sum != 1:
        print("Total sum of simplex is", total_sum)

    target_sum = n_percent / 100.0  # Convert percentage to fraction
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
    # if(len(selected_values)>budget) return only the top budget elements:
    if len(selected_values) > budget:
        selected_values = selected_values[:budget]
        selected_indices = selected_indices[:budget]

    return selected_values, selected_indices


# %% [markdown]
# # Data, Model & Experimental Settings
# The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes.The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class. The training set contains 50,000 images and test set contains 10,000 images. We will use custom_dataset() function in Trust to simulated a class imbalance scenario using the split_cfg dictionary given below. We then use a ResNet18 model as our task DNN and train it on the simulated imbalanced version of the CIFAR-10 dataset. Next we perform targeted selection using various SMI functions and compare their gain in overall accuracy as well as on the imbalanced classes.

# %%
feature = "classimb"

# datadir = 'data/'
datadir = (
    "data/medmnist"  # contains the npz file of the data_name dataset listed below
)
data_name = "pneumoniamnist"

learning_rate = 0.0003
computeClassErrorLog = True
device_id = 0
device = "cuda:" + str(device_id) if torch.cuda.is_available() else "cpu"
miscls = False  # Set to True if only the misclassified examples from the imbalanced classes is to be used

num_cls = 2
# budget = 10
visualize_tsne = False
split_cfg = {
    #  "per_class_train":{0:20,1:20},
    #  "per_class_val":{0:10,1:10},
    #  "per_class_lake":{0:600,1:600},
    #  "per_class_test":{0:200,1:200},
    "sel_cls_idx": [0, 1],
    "per_imbclass_train": {0: 50, 1: 50},
    "per_imbclass_val": {0: 50, 1: 50},
    "per_imbclass_lake": {0: 1000, 1: 3000},
    "per_imbclass_test": {0: 300, 1: 300},
    # "sel_cls_idx": [0, 1],
    # "per_imbclass_train": {0: 5, 1: 5},
    # "per_imbclass_val": {0: 10, 1: 10},
    # "per_imbclass_lake": {0: 600, 1: 1000},
    # "per_imbclass_test": {0: 300, 1: 600},
}
print("split_cfg:", split_cfg)

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


def run_targeted_selection(
    dataset_name,
    datadir,
    feature,
    model_name,
    budget,
    split_cfg,
    learning_rate,
    run,
    device,
    computeErrorLog,
    strategy="SIM",
    sf="",
    embedding_type="features",
):
    # load the dataset in the class imbalance setting
    (
        train_set,
        val_set,
        test_set,
        lake_set,
        sel_cls_idx,
        num_cls,
    ) = load_biodataset_custom(datadir, dataset_name, feature, split_cfg, False, False)

    print("Indices of randomly selected classes for imbalance: ", sel_cls_idx)

    # Set batch size for train, validation and test datasets
    N = len(train_set)
    trn_batch_size = 3000
    val_batch_size = 1000
    tst_batch_size = 1000

    # # Create dataloaders
    # trainloader = torch.utils.data.DataLoader(
    #     train_set, batch_size=trn_batch_size, shuffle=True, pin_memory=True
    # )

    valloader = torch.utils.data.DataLoader(
        val_set, batch_size=val_batch_size, shuffle=False, pin_memory=True
    )

    tstloader = torch.utils.data.DataLoader(
        test_set, batch_size=tst_batch_size, shuffle=False, pin_memory=True
    )

    # lakeloader = torch.utils.data.DataLoader(
    #     lake_set, batch_size=tst_batch_size, shuffle=False, pin_memory=True
    # )
    true_lake_set = copy.deepcopy(lake_set)
    # Budget for subset selection
    bud = budget
    # soft subset max budget % of the lake set
    ss_max_budget_percentage = 80
    # Variables to store accuracies
    num_rounds = 10  # The first round is for training the initial model and the second round is to train the final model
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
    all_logs_dir = (
        "/home/wassal/trust-wassal/tutorials/results/"
        + dataset_name
        + "/"
        + feature
        + "/rounds"
        + str(num_rounds)
        + "/"
        + sf
        + "/"
        + str(bud)
        + "/"
        + str(run)
    )
    print("Saving results to: ", all_logs_dir)
    subprocess.run(["mkdir", "-p", all_logs_dir])  # Uncomment for saving results
    exp_name = (
        dataset_name
        + "_"
        + feature
        + "_"
        + strategy
        + "_"
        + str(len(sel_cls_idx))
        + "_"
        + sf
        + "_budget:"
        + str(bud)
        + "_rounds:"
        + str(num_rounds)
        + "_runs"
        + str(run)
    )

    # Create a dictionary for storing results and the experimental setting
    res_dict = {
        "dataset": data_name,
        "feature": feature,
        "sel_func": sf,
        "sel_budget": budget,
        "num_selections": num_rounds - 1,
        "model": model_name,
        "learning_rate": learning_rate,
        "setting": split_cfg,
        "all_class_acc": None,
        "test_acc": [],
        "sel_per_cls": [],
        "sel_cls_idx": sel_cls_idx,
    }

    # strategy_args = {'batch_size': 4000, 'device':device, 'embedding_type':embedding_type, 'keep_embedding':True,'lr':learning_rate}
    strategy_args = {
        "batch_size": 4000,
        "device": device,
        "embedding_type": embedding_type,
        "keep_embedding": True,
        "lr": 0.8,
        "iterations": 15,
        "step_size": 3,
        "min_iteration": 5,
    }
    unlabeled_lake_set = LabeledToUnlabeledDataset(lake_set)
    if "WITHSOFT" in strategy or strategy == "WASSAL":
        print("initaizing models for soft subset training")
        for_query_set = getQuerySet(train_set, sel_cls_idx, recipe="asis")
        #
        strategy_softsubset = WASSAL_Multiclass(
            train_set, unlabeled_lake_set, for_query_set, model, num_cls, strategy_args
        )

    print("initailizing strategy class for " + sf)
    if strategy == "AL" or strategy == "AL_WITHSOFT":
        if sf == "badge" or sf == "badge_withsoft":
            strategy_sel = BADGE(
                train_set, unlabeled_lake_set, model, num_cls, strategy_args
            )
        elif sf == "us" or sf == "us_withsoft":
            strategy_sel = EntropySampling(
                train_set, unlabeled_lake_set, model, num_cls, strategy_args
            )
        elif sf == "glister" or sf == "glister_withsoft":
            strategy_sel = GLISTER(
                train_set,
                unlabeled_lake_set,
                model,
                num_cls,
                strategy_args,
                val_set,
                typeOf="rand",
                lam=0.1,
            )
        elif sf == "gradmatch-tss" or sf == "gradmatch-tss_withsoft":
            strategy_sel = GradMatchActive(
                train_set, unlabeled_lake_set, model, num_cls, strategy_args, val_set
            )
        elif sf == "coreset" or sf == "coreset_withsoft":
            strategy_sel = CoreSet(
                train_set, unlabeled_lake_set, model, num_cls, strategy_args
            )
        elif sf == "leastconf" or sf == "leastconf_withsoft":
            strategy_sel = LeastConfidenceSampling(
                train_set, unlabeled_lake_set, model, num_cls, strategy_args
            )
        elif sf == "margin" or sf == "margin_withsoft":
            strategy_sel = MarginSampling(
                train_set, unlabeled_lake_set, model, num_cls, strategy_args
            )

    elif strategy == "SIM" or strategy == "SIM_WITHSOFT":
        strategy_args["smi_function"] = sf
        strategy_args["optimizer"] = "LazyGreedy"
        for_query_set = getQuerySet(train_set, sel_cls_idx)
        strategy_sel = SMI(
            train_set, unlabeled_lake_set, for_query_set, model, num_cls, strategy_args
        )

    elif strategy == "SCMI" or strategy == "SCMI_WITHSOFT":
        strategy_args["scmi_function"] = sf
        strategy_args["optimizer"] = "LazyGreedy"
        for_query_set = getQuerySet(train_set, sel_cls_idx)
        for_private_set = getPrivateSet(train_set, sel_cls_idx)
        strategy_sel = SCMI(
            train_set,
            unlabeled_lake_set,
            for_query_set,
            for_private_set,
            model,
            num_cls,
            strategy_args,
        )

    if strategy == "random":
        strategy_sel = RandomSampling(
            train_set, unlabeled_lake_set, model, num_cls, strategy_args
        )
    if strategy == "WASSAL" or strategy == "WASSAL_WITHSOFT":
        for_query_set = getQuerySet(train_set, sel_cls_idx, recipe="asis")
        strategy_sel = WASSAL_Multiclass(
            train_set, unlabeled_lake_set, for_query_set, model, num_cls, strategy_args
        )

    if strategy == "WASSAL_P" or strategy == "WASSAL_P_WITHSOFT":
        for_query_set = getQuerySet(train_set, sel_cls_idx)
        for_private_set = getPrivateSet(train_set, sel_cls_idx)
        strategy_sel = WASSAL_P(
            train_set,
            unlabeled_lake_set,
            for_query_set,
            for_private_set,
            model,
            num_cls,
            strategy_args,
        )

    # Loss Functions
    criterion, criterion_nored = loss_function()

    # Getting the optimizer and scheduler
    optimizer = optimizer_without_scheduler(model, learning_rate)
    final_val_predictions = []
    final_val_classifications = []
    final_tst_predictions = []
    final_tst_classifications = []

    for i in range(num_rounds):
        tst_loss = 0
        tst_correct = 0
        tst_total = 0
        val_loss = 0
        val_correct = 0
        val_total = 0

        if i == 0:
            print("Initial training epoch")
            if os.path.exists(
                initModelPath
            ):  # Read the initial trained model if it exists
                model.load_state_dict(torch.load(initModelPath, map_location=device))
                print(
                    "Init model loaded from disk, skipping init training: ",
                    initModelPath,
                )
                model.eval()
                with torch.no_grad():
                    final_val_predictions = []
                    final_val_classifications = []
                    for batch_idx, (inputs, targets) in enumerate(valloader):
                        inputs, targets = inputs.to(device), targets.to(
                            device, non_blocking=True
                        )
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()
                        final_val_predictions += list(predicted.cpu().numpy())
                        final_val_classifications += list(
                            predicted.eq(targets).cpu().numpy()
                        )

                    final_tst_predictions = []
                    final_tst_classifications = []
                    for batch_idx, (inputs, targets) in enumerate(tstloader):
                        inputs, targets = inputs.to(device), targets.to(
                            device, non_blocking=True
                        )
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        tst_loss += loss.item()
                        _, predicted = outputs.max(1)
                        tst_total += targets.size(0)
                        tst_correct += predicted.eq(targets).sum().item()
                        final_tst_predictions += list(predicted.cpu().numpy())
                        final_tst_classifications += list(
                            predicted.eq(targets).cpu().numpy()
                        )
                    best_val_acc = val_correct / val_total
                    val_acc[i] = val_correct / val_total
                    tst_acc[i] = tst_correct / tst_total
                    val_losses[i] = val_loss
                    tst_losses[i] = tst_loss
                    res_dict["test_acc"].append(tst_acc[i] * 100)
                continue
        else:
            # Remove true labels from the unlabeled dataset, the hypothesized labels are computed when select is called
            unlabeled_lake_set = LabeledToUnlabeledDataset(lake_set)
            print(
                "Started a new AL round, and updating the model, queryset and data(train and unlabeled_set) for strategy "
                + sf
            )
            strategy_sel.update_data(train_set, unlabeled_lake_set)
            strategy_sel.update_model(model)

           
            ####SIM####
            if (
                strategy == "SIM"
                or strategy == "SIM_WITHSOFT"
                or strategy == "WASSAL"
                or strategy == "WASSAL_WITHSOFT"
            ):
                if('WASSAL' in strategy):
                    for_query_set = getQuerySet(train_set, sel_cls_idx,recipe="asis")
                else:
                # make a dataloader for the misclassifications - only for experiments with targets
                    for_query_set = getQuerySet(train_set, sel_cls_idx)
                print("updating queryset for strategy " + sf)
                strategy_sel.update_queries(for_query_set)

                print("size of query set", len(for_query_set))
            # if SCMI_WITHSOFT
            elif (
                strategy == "SCMI_WITHSOFT"
                or strategy == "SCMI"
                or strategy == "WASSAL_P"
                or strategy == "WASSAL_P_WITHSOFT"
            ):
                if sf.endswith("mi"):
                    if feature == "classimb":
                        # make a dataloader for the misclassifications - only for experiments with targets
                        for_query_set = getQuerySet(train_set, sel_cls_idx)
                        for_private_set = getPrivateSet(train_set, sel_cls_idx)
                        print("updating queryset for strategy " + sf)
                        strategy_sel.update_queries(for_query_set)
                        strategy_sel.update_privates(for_private_set)

                        print("size of query set", len(for_query_set))
            # if AL_WITHSOFT
            elif strategy == "AL_WITHSOFT" or strategy == "AL":
                if sf == "glister-tss" or sf == "gradmatch-tss":
                    for_query_set = getQuerySet(train_set, sel_cls_idx, recipe="asis")
                    print("updating queryset for strategy " + sf)
                    strategy_sel.update_queries(for_query_set)

                    print("size of query set", len(for_query_set))
            
            # compute the error log before every selection
            if computeErrorLog:
                tst_err_log, val_err_log, val_class_err_idxs = find_err_per_class(
                    test_set,
                    val_set,
                    final_val_classifications,
                    final_val_predictions,
                    final_tst_classifications,
                    final_tst_predictions,
                    all_logs_dir,
                    sf + "_" + str(bud),
                )

                csvlog.append([100 - x for x in tst_err_log])
                val_csvlog.append([100 - x for x in val_err_log])

            #update softsubset model and query if WITHSOFT
            if "WITHSOFT" in strategy or strategy=="WASSAL":
                print(
                    "Updating softsoft data, queryset and model for strategy " + sf,
                )

                strategy_softsubset.update_data(train_set, unlabeled_lake_set)
                strategy_softsubset.update_model(model)

                for_query_set = getQuerySet(train_set, sel_cls_idx, recipe="asis")
                strategy_softsubset.update_queries(for_query_set)
                
            classwise_final_indices_simplex = None
           
            #get simplex_query
            if "WITHSOFT" in strategy or strategy== "WASSAL":
                print(
                        "Calculating simplexes since we need to do softsubsetting for strategy "
                        + sf
                )
    
                subset,classwise_final_indices_simplex = strategy_softsubset.select(budget)
                classwise_final_indices_simplex_cpu = [
                    (
                       
                        tensor1.clone().cpu().detach(),
                        tensor2.clone().cpu().detach(),
                        class_idx,
                    )
                    for (
                        
                        tensor1,
                        tensor2,
                        class_idx,
                    ) in classwise_final_indices_simplex
                ]
                #create a folder to save the simplex plots
                simplex_dir = (
                    "/home/wassal/trust-wassal/tutorials/results/"
                    + dataset_name
                    + "/"
                    + feature
                    + "/rounds"
                    + str(num_rounds)
                    + "/"
                    + sf
                    + "/"
                    + str(bud)
                     +"/"
                    +str(run)                
                    + "/simplex_viz/al_round_"
                    + "/simplex/"
                    +str(i)
                    
                )
                subprocess.run(["mkdir", "-p", simplex_dir])
                plotsimpelxDistribution(
                    lake_set, classwise_final_indices_simplex_cpu,simplex_dir
                )

                

            #selecting subset using an AL strategy
            # subset = []
            # print("Selecing AL data for strategy " + sf)
            # if strategy == "WASSAL" or strategy == "WASSAL_WITHSOFT":
            #     print('selecting subset as well for '+sf)
                
            #     for (
            #         selected_indices,
            #         simplex_query,
            #         simplex_refrain,
            #         class_idx,
            #     ) in classwise_final_indices_simplex:
            #         subset += selected_indices

            #     # analyze_simplex(temp_args,lake_set,simplex_query)
            if strategy == "WASSAL_P" or strategy == "WASSAL_P_WITHSOFT":
                subset, simplex_query, simplex_private = strategy_sel.select(budget)

            # for other strategies simple to get subset
            elif "WASSAL" not in strategy:
                subset = strategy_sel.select(budget)

            lake_subset_idxs = (
                subset  # indices wrt to lake that need to be removed from the lake
            )

            perClsSel = getPerClassSel(true_lake_set, lake_subset_idxs, num_cls)
            res_dict["sel_per_cls"].append(perClsSel)

            if visualize_tsne:
                tsne_plt = tsne_smi(
                    strategy_sel.unlabeled_data_embedding.cpu(),
                    lake_set.targets,
                    strategy_sel.query_embedding.cpu(),
                    sel_cls_idx,
                    subset,
                )
                print("Computed TSNE plot of the selection")

            print("#### Selection Complete, Now re-training with augmented subset ####")

            weighted_lakeloader=None
            weighted_refrain_lakeloader=None

            #preparing weighted loader for weighted training
            if 'WITHSOFT' in strategy:
                    
                # Aggregation lists
                all_small_images = []
                all_small_targets = []
                all_small_simplex_query = []
                all_small_refrain_images = []
                all_small_refrain_targets = []
                all_small_simplex_refrain = []
                all_soft_selected_indices = []
                for (
                   
                    simplex_query,
                    simplex_refrain,
                    class_idx,
                ) in classwise_final_indices_simplex:
                    # Extract images and targets from weighted_lake_set
                    images = [lake_set[i][0] for i in range(len(lake_set))]
                    targets = torch.tensor(class_idx)
                    targets_refrain = torch.tensor(class_idx)
                    targets = targets.repeat(len(lake_set))
                    targets_refrain = targets_refrain.repeat(len(lake_set))
                    sofftsimplex_query = simplex_query.detach().cpu().numpy()
                    softsimplex_refrain = simplex_refrain.detach().cpu().numpy()
                    ss_budget =len(sofftsimplex_query)
                    # choose the top simplex_query that contributes 30% to the size of that class in trainset
                    _, top_n_indices = top_elements_contribute_to_percentage(
                        sofftsimplex_query, ss_max_budget_percentage, ss_budget
                    )

                    (
                        _,
                        top_n_refrain_indices,
                    ) = top_elements_contribute_to_percentage(
                        softsimplex_refrain, ss_max_budget_percentage, ss_budget
                    )
                    all_soft_selected_indices += top_n_indices
                    # Collect the data
                    all_small_images += [images[i] for i in top_n_indices]
                    all_small_refrain_images += [
                        images[i] for i in top_n_refrain_indices
                    ]
                    all_small_targets += targets[top_n_indices.copy()].tolist()
                    all_small_refrain_targets += targets_refrain[
                        top_n_refrain_indices.copy()
                    ].tolist()
                    softsimplex_query_normed = sofftsimplex_query[top_n_indices] / (
                        sofftsimplex_query[top_n_indices].sum()
                    )
                    all_small_simplex_query += sofftsimplex_query[
                        top_n_indices
                    ].tolist()
                    softsimplex_refrain_normed = softsimplex_refrain[
                        top_n_refrain_indices
                    ] / (softsimplex_refrain[top_n_refrain_indices].sum())
                    all_small_simplex_refrain += softsimplex_refrain_normed.tolist()
                    
                #print the size of simplex_query for given strategy and budget
                print("size of simplex_query for strategy "+sf+" and budget "+str(budget)+" is "+str(len(all_small_simplex_query))+"in round "+str(i))

                # Convert lists to tensors
                all_small_targets = torch.tensor(all_small_targets)
                all_small_refrain_targets = torch.tensor(all_small_refrain_targets)
                all_small_simplex_query = torch.tensor(all_small_simplex_query)
                all_small_simplex_refrain = torch.tensor(all_small_simplex_refrain)

                # Form the combined weighted dataset
                weighted_lake_set = WeightedDataset(
                    all_small_images,
                    all_small_targets,
                    all_small_simplex_query,
                    None,
                    None,
                )
                weighted_refrain_lake_set = WeightedDataset(
                    all_small_refrain_images,
                    all_small_refrain_targets,
                    all_small_simplex_refrain,
                    None,
                    None,
                )

                # Load into a dataloader
                weighted_lakeloader = torch.utils.data.DataLoader(
                    weighted_lake_set,
                    batch_size=trn_batch_size,
                    shuffle=True,
                    pin_memory=True,
                )
                # weighted_refrain_lakeloader = torch.utils.data.DataLoader(
                #     weighted_refrain_lake_set,
                #     batch_size=len(weighted_refrain_lake_set),
                #     shuffle=True,
                #     pin_memory=True,
                # )
            

            
            # augment the train_set with selected indices from the lake
            train_set, lake_set, true_lake_set, add_val_set = aug_train_subset(
                train_set,
                lake_set,
                true_lake_set,
                subset,
                lake_subset_idxs,
                budget,
                True,
            )  # aug train with random if budget is not filled
            print(
                "After augmentation, size of train_set: ",
                len(train_set),
                " unlabeled set: ",
                len(lake_set),
                " val set: ",
                len(val_set),
            )

            #Reinit train and lake loaders with new splits and reinit the model
            trainloader = torch.utils.data.DataLoader(
                train_set, batch_size=trn_batch_size, shuffle=True, pin_memory=True
            )

            # lakeloader = torch.utils.data.DataLoader(
            #     lake_set, batch_size=tst_batch_size, shuffle=False, pin_memory=True
            # )
            # model = create_model(model_name, num_cls, device, strategy_args['embedding_type'])
            # optimizer = optimizer_without_scheduler(model, learning_rate)

            # Start training
            start_time = time.time()
            num_ep = 1
            #         while(num_ep<150):
            # first train until full training accuracy is 0.99
            while full_trn_acc[i] < 0.99 and num_ep < 100:
                loss=0.0
                soft_loss=0.0
                hard_loss=0.0

                model.train()
                optimizer.zero_grad()
                if "WITHSOFT" in strategy:
                    for batch_idx, (
                                inputs,
                                targets,
                                simplex_query,
                                _,
                                _,
                            ) in enumerate(weighted_lakeloader):
                                # Variables in Pytorch are differentiable.
                                inputs = inputs.to(device)
                                targets = targets.to(device)
                                # normalize simplex_query
                                loss = 0.0
                                simplex_query = simplex_query.to(device)
                                # This will zero out the gradients for this batch.
                                
                                soft_outputs = model(inputs)
                                target_loss_per_sample = criterion(soft_outputs, targets)
                                
                                soft_loss += (simplex_query * target_loss_per_sample).sum()
                
                    
                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(device), targets.to(
                        device, non_blocking=True
                    )
                    # Variables in Pytorch are differentiable.
                    inputs, target = Variable(inputs), Variable(inputs)
                    # This will zero out the gradients for this batch.
                    
                    outputs = model(inputs)
                    hard_loss += criterion(outputs, targets)
                
                loss=hard_loss+(3*soft_loss)
                loss.backward()
                optimizer.step()
                #             scheduler.step()

                full_trn_loss = 0
                full_trn_correct = 0
                full_trn_total = 0
                model.eval()
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(
                        trainloader
                    ):  # Compute Train accuracy
                        inputs, targets = inputs.to(device), targets.to(
                            device, non_blocking=True
                        )
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        full_trn_loss += loss.item()
                        _, predicted = outputs.max(1)
                        full_trn_total += targets.size(0)
                        full_trn_correct += predicted.eq(targets).sum().item()
                    full_trn_acc[i] = full_trn_correct / full_trn_total
                    print(
                        "Selection Epoch ",
                        i,
                        " Training epoch [",
                        num_ep,
                        "]",
                        " Training Acc: ",
                        full_trn_acc[i],
                        end="\r",
                    )
                    num_ep += 1
                timing[i] = time.time() - start_time

            with torch.no_grad():
                final_val_predictions = []
                final_val_classifications = []
                for batch_idx, (inputs, targets) in enumerate(
                    valloader
                ):  # Compute Val accuracy
                    inputs, targets = inputs.to(device), targets.to(
                        device, non_blocking=True
                    )
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
                    final_val_predictions += list(predicted.cpu().numpy())
                    final_val_classifications += list(
                        predicted.eq(targets).cpu().numpy()
                    )

                final_tst_predictions = []
                final_tst_classifications = []
                for batch_idx, (inputs, targets) in enumerate(
                    tstloader
                ):  # Compute test accuracy
                    inputs, targets = inputs.to(device), targets.to(
                        device, non_blocking=True
                    )
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    tst_loss += loss.item()
                    _, predicted = outputs.max(1)
                    tst_total += targets.size(0)
                    tst_correct += predicted.eq(targets).sum().item()
                    final_tst_predictions += list(predicted.cpu().numpy())
                    final_tst_classifications += list(
                        predicted.eq(targets).cpu().numpy()
                    )
            val_acc[i] = val_correct / val_total
            tst_acc[i] = tst_correct / tst_total
            val_losses[i] = val_loss
            fulltrn_losses[i] = full_trn_loss
            tst_losses[i] = tst_loss
            full_val_acc = list(np.array(val_acc))
            full_timing = list(np.array(timing))
            res_dict["test_acc"].append(tst_acc[i] * 100)
            print(
                "Epoch:",
                i + 1,
                "FullTrn,TrainAcc,ValLoss,ValAcc,TstLoss,TstAcc,Time:",
                full_trn_loss,
                full_trn_acc[i],
                val_loss,
                val_acc[i],
                tst_loss,
                tst_acc[i],
                timing[i],
            )
            print(
                "Gain in accuracy: ",
                res_dict["test_acc"][i] - res_dict["test_acc"][i - 1],
            )


        if i == 0:
            print("Saving initial model")
            torch.save(
                model.state_dict(), initModelPath
            )  # save initial train model if not present

    # Compute the statistics of the final model
    if computeErrorLog:
        print("**** Final Metrics after Targeted Learning ****")
        tst_err_log, val_err_log, val_class_err_idxs = find_err_per_class(
            test_set,
            val_set,
            final_val_classifications,
            final_val_predictions,
            final_tst_classifications,
            final_tst_predictions,
            all_logs_dir,
            sf + "_" + str(bud),
        )
        csvlog.append([100 - x for x in tst_err_log])
        val_csvlog.append([100 - x for x in val_err_log])
        res_dict["all_class_acc"] = csvlog
        res_dict["all_val_class_acc"] = val_csvlog
        with open(os.path.join(all_logs_dir, exp_name + ".csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerows(csvlog)
    # save results dir with test acc and per class selections
    with open(os.path.join(all_logs_dir, exp_name + ".json"), "w") as fp:
        json.dump(res_dict, fp)
    # Print overall acc improvement and rare class acc improvement, show that TL selected relevant points in space, is possible show some images
    print_final_results(res_dict, sel_cls_idx)
    print("Total gain in accuracy: ", res_dict["test_acc"][i] - res_dict["test_acc"][0])


#     tsne_plt.show()


# %%
experiments = ["exp2", "exp3", "exp4", "exp5"]
seeds = [24, 48, 86, 28, 92]
budgets = [40, 50, 60, 70, 80, 90, 100]
#budgets = [100]
device_id = 0
device = "cuda:" + str(device_id) if torch.cuda.is_available() else "cpu"

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


embedding_type = "features"  # Type of the representation to use (gradients/features)
model_name = "ResNet18"  # Model to use for training
initModelPath = (
    "/home/wassal/trust-wassal/tutorials/results/"
    + data_name
    + "_"
    + model_name
    + "_"
    + embedding_type
    + "_"
    + str(learning_rate)
    + "_"
    + str(split_cfg["sel_cls_idx"])
)
#skip strategies that are already run
skip_strategies = []
skip_budgets = []
# Model Creation
model = create_model(model_name, num_cls, device, embedding_type)
strategies = [
    # al soft
    ("WASSAL", "WASSAL"),
    ("WASSAL_WITHSOFT", "WASSAL_WITHSOFT"),
    ("AL", "glister"),
    ("AL_WITHSOFT", "glister_withsoft"),
    ("AL", "gradmatch-tss"),
    ("AL_WITHSOFT", "gradmatch-tss_withsoft"),
    ("AL", "coreset"),
    ("AL_WITHSOFT", "coreset_withsoft"),
    ("AL", "leastconf"),
    ("AL_WITHSOFT", "leastconf_withsoft"),
    ("AL", "margin"),
    ("AL_WITHSOFT", "margin_withsoft"),
    ("random", "random"),
    ("AL", "badge"),
    ("AL", "badge_withsoft"),
    ("AL_WITHSOFT", "us_withsoft"),
    ("AL", "us"),
    
]

for i, experiment in enumerate(experiments):
    seed = seeds[i]
    torch.manual_seed(seed)
    np.random.seed(seed)
    run = experiment

    # Loop for each budget from 50 to 400 in intervals of 50
    for b in budgets:
        # Loop through each strategy
        for strategy, method in strategies:
            #skip strategies that are already run
            if strategy in skip_strategies and b in skip_budgets:
                continue
            print("Budget ", b, " Strategy ", strategy, " Method ", method)
            run_targeted_selection(
                data_name,
                datadir,
                feature,
                model_name,
                b,  # updated budget
                split_cfg,
                learning_rate,
                run,
                device,
                computeClassErrorLog,
                strategy,
                method,
                embedding_type,
            )
