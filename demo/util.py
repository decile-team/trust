import sys

sys.path.append("..")


from trust.utils.utils import *
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from trust.utils.models.lenet import LeNet
from trust.utils.custom_dataset import load_dataset_custom
from torch.utils.data import Subset
from torch.autograd import Variable
from trust.strategies.smi import SMI
from trust.strategies.random_sampling import RandomSampling


def model_eval_loss(data_loader, model, criterion):
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(
                device, non_blocking=True)
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
    if embedding_type == "gradients":
        model = LeNet(num_cls)
    else:
        model = LeNet()
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs)
    return optimizer, scheduler


def optimizer_without_scheduler(model, learning_rate, m=0.9, wd=5e-4):
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
    col1 = [str(i) for i in range(len(val_err_log))]
    val_acc = [str(100-i) for i in val_err_log]
    tst_acc = [str(100-i) for i in tst_err_log]
    table = [col1, val_acc, tst_acc]
    table = map(list, zip(*table))
    d = {'Class': col1, 'Val Accuracy': val_acc, 'Test Accuracy': tst_acc}
    df = pd.DataFrame(data=d)
    return df


def find_err_per_class(test_set, val_set, final_val_classifications, final_val_predictions, final_tst_classifications,
                       final_tst_predictions, prefix, feature, num_cls):
    val_err_idx = list(
        np.where(np.array(final_val_classifications) == False)[0])
    tst_err_idx = list(
        np.where(np.array(final_tst_classifications) == False)[0])
    val_class_err_idxs = []
    tst_err_log = []
    val_err_log = []
    for i in range(num_cls):
        if(feature == "classimb"):
            tst_class_idxs = list(torch.where(torch.Tensor(
                test_set.targets) == i)[0].cpu().numpy())
        if(feature=="ood"): 
            tst_class_idxs = list(torch.where(torch.Tensor(test_set.targets.float()) == i)[0].cpu().numpy())
            
        val_class_idxs = list(torch.where(torch.Tensor(
            val_set.targets.float()) == i)[0].cpu().numpy())
        # err classifications per class
        val_err_class_idx = set(val_err_idx).intersection(set(val_class_idxs))
        tst_err_class_idx = set(tst_err_idx).intersection(set(tst_class_idxs))
        if(len(val_class_idxs) > 0):
            val_error_perc = round(
                (len(val_err_class_idx)/len(val_class_idxs))*100, 2)
        else:
            val_error_perc = 0
        
        if(len(tst_class_idxs) > 0):
            tst_error_perc = round(
                (len(tst_err_class_idx)/len(tst_class_idxs))*100, 2)
        else:
            tst_error_perc = 0
            
        val_class_err_idxs.append(val_err_class_idx)
        tst_err_log.append(tst_error_perc)
        val_err_log.append(val_error_perc)
    df = displayTable(val_err_log, tst_err_log)
    tst_err_log.append(sum(tst_err_log)/len(tst_err_log))
    val_err_log.append(sum(val_err_log)/len(val_err_log))
    return tst_err_log, val_err_log, val_class_err_idxs, df


def aug_train_subset(train_set, lake_set, true_lake_set, subset, lake_subset_idxs, budget, augrandom=False):
    all_lake_idx = list(range(len(lake_set)))
    if(not(len(subset) == budget) and augrandom):
        print("Budget not filled, adding ", str(
            int(budget) - len(subset)), " randomly.")
        remain_budget = int(budget) - len(subset)
        remain_lake_idx = list(set(all_lake_idx) - set(subset))
        random_subset_idx = list(np.random.choice(
            np.array(remain_lake_idx), size=int(remain_budget), replace=False))
        subset += random_subset_idx
    lake_ss = SubsetWithTargets(true_lake_set, subset, torch.Tensor(
        true_lake_set.targets.float())[subset])
    remain_lake_idx = list(set(all_lake_idx) - set(lake_subset_idxs))
    remain_lake_set = SubsetWithTargets(lake_set, remain_lake_idx, torch.Tensor(
        lake_set.targets.float())[remain_lake_idx])
    remain_true_lake_set = SubsetWithTargets(true_lake_set, remain_lake_idx, torch.Tensor(
        true_lake_set.targets.float())[remain_lake_idx])
    aug_train_set = torch.utils.data.ConcatDataset([train_set, lake_ss])
    return aug_train_set, remain_lake_set, remain_true_lake_set, lake_ss


def getQuerySet(val_set, val_class_err_idxs, imb_cls_idx, miscls):
    miscls_idx = []
    if(miscls):
        for i in range(len(val_class_err_idxs)):
            if i in imb_cls_idx:
                miscls_idx += val_class_err_idxs[i]
        print("Total misclassified examples from imbalanced classes (Size of query set): ", len(
            miscls_idx))
    else:
        for i in imb_cls_idx:
            imb_cls_samples = list(torch.where(torch.Tensor(
                val_set.targets.float()) == i)[0].cpu().numpy())
            miscls_idx += imb_cls_samples
        print("Total samples from imbalanced classes as targets (Size of query set): ", len(
            miscls_idx))
    return Subset(val_set, miscls_idx), val_set.targets[miscls_idx]


def getPerClassSel(lake_set, subset, num_cls):
    perClsSel = []
    subset_cls = torch.Tensor(lake_set.targets.float())[subset]
    for i in range(num_cls):
        cls_subset_idx = list(torch.where(subset_cls == i)[0].cpu().numpy())
        perClsSel.append(len(cls_subset_idx))
    return perClsSel


def get_final_results(res_dict, sel_cls_idx):
    ans = []
    ans.append(str(res_dict['test_acc'][1]-res_dict['test_acc'][0]))
    bf_sel_cls_acc = np.array(res_dict['all_class_acc'][0])[sel_cls_idx]
    af_sel_cls_acc = np.array(res_dict['all_class_acc'][1])[sel_cls_idx]
    ans.append(str(np.mean(af_sel_cls_acc-bf_sel_cls_acc)))
    return ans


device_id = 1
datadir = 'data/'
device = "cuda:"+str(device_id) if torch.cuda.is_available() else "cpu"
miscls = False  # Set to True if only the misclassified examples from the imbalanced classes is to be used
# Type of the representation to use (gradients/features)
embedding_type = "gradients"



trn_batch_size = 20
val_batch_size = 10
tst_batch_size = 10


def run_targeted_selection(dataset_name, feature, model_name, budget, split_cfg, learning_rate, strategy, sf, is_full):

    computeErrorLog = True

    # load the dataset in the class imbalance setting
    train_set, val_set, test_set, lake_set, sel_cls_idx, num_cls = load_dataset_custom(
        datadir, dataset_name, feature, split_cfg, False, False)

    # Set batch size for train, validation and test datasets
    if(feature == "ood"):
        num_cls += 1

    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=trn_batch_size,
                                              shuffle=True, pin_memory=True)

    valloader = torch.utils.data.DataLoader(val_set, batch_size=val_batch_size,
                                            shuffle=False, pin_memory=True)

    tstloader = torch.utils.data.DataLoader(test_set, batch_size=tst_batch_size,
                                            shuffle=False, pin_memory=True)

    true_lake_set = copy.deepcopy(lake_set)
    bud = budget

    # Variables to store accuracies
    num_rounds = 2  # The first round is for training the initial model and the second round is to train the final model
    fulltrn_losses = np.zeros(num_rounds)
    val_losses = np.zeros(num_rounds)
    tst_losses = np.zeros(num_rounds)
    timing = np.zeros(num_rounds)
    val_acc = np.zeros(num_rounds)
    full_trn_acc = np.zeros(num_rounds)
    tst_acc = np.zeros(num_rounds)
    final_tst_predictions = []
    final_tst_classifications = []
    csvlog = []
    val_csvlog = []

    # Create a dictionary for storing results and the experimental setting
    res_dict = {"dataset": dataset_name,
                "feature": feature,
                "sel_func": sf,
                "sel_budget": budget,
                "num_selections": num_rounds-1,
                "model": model_name,
                "learning_rate": learning_rate,
                "setting": split_cfg,
                "all_class_acc": None,
                "test_acc": [],
                "sel_per_cls": [],
                "sel_cls_idx": list(sel_cls_idx)}

    # Model Creation
    model = create_model(model_name, num_cls, device, embedding_type)
    strategy_args = {'batch_size': 20, 'device': device,
                     'embedding_type': 'gradients', 'keep_embedding': True}
    unlabeled_lake_set = LabeledToUnlabeledDataset(lake_set)

    if(strategy == "SMI"):
        strategy_args['smi_function'] = sf
        strategy_sel = SMI(train_set, unlabeled_lake_set,
                           val_set, model, num_cls, strategy_args)
    if(strategy == "random"):
        strategy_sel = RandomSampling(
            train_set, unlabeled_lake_set, model, num_cls, strategy_args)

    # Loss Functions
    criterion, _ = loss_function()

    # Getting the optimizer and scheduler
    optimizer = optimizer_without_scheduler(model, learning_rate)

    training_progress = {}

    for i in range(num_rounds):
        training_progress[str(i)] = [[], []]
        tst_loss = 0
        tst_correct = 0
        tst_total = 0
        val_loss = 0
        val_correct = 0
        val_total = 0

        if(i == 0):
            print("Initial training epoch")
        else:
            # Remove true labels from the unlabeled dataset, the hypothesized labels are computed when select is called
            unlabeled_lake_set = LabeledToUnlabeledDataset(lake_set)
            strategy_sel.update_data(train_set, unlabeled_lake_set)
            # compute the error log before every selection
            if(computeErrorLog):
                tst_err_log, val_err_log, val_class_err_idxs, initial_metrics = find_err_per_class(
                    test_set, val_set, final_val_classifications, final_val_predictions, final_tst_classifications, final_tst_predictions, sf+"_"+str(bud), feature, num_cls)
                csvlog.append([100-x for x in tst_err_log])
                val_csvlog.append([100-x for x in val_err_log])
            ####SMI####
            if(strategy == "SMI" or strategy == "SF"):
                if(sf.endswith("mi")):
                    if(feature == "classimb"):
                        # make a dataloader for the misclassifications - only for experiments with targets
                        miscls_set, _ = getQuerySet(
                            val_set, val_class_err_idxs, sel_cls_idx, miscls)
                        strategy_sel.update_queries(miscls_set)

            strategy_sel.update_model(model)
            subset = strategy_sel.select(budget)
            lake_subset_idxs = subset
            perClsSel = getPerClassSel(
                true_lake_set, lake_subset_idxs, num_cls)
            res_dict['sel_per_cls'].append(perClsSel)
            if(feature == "classimb"):
                percentage_selection = 0
                for j in split_cfg['sel_cls_idx_frontend']:
                    percentage_selection += perClsSel[j]
                percentage_selection = (percentage_selection / sum(perClsSel)) * 100
            else:
                percentage_selection = 0
                for j in split_cfg['idc_classes_frontend']:
                    percentage_selection += perClsSel[j]
                percentage_selection = (percentage_selection / sum(perClsSel)) * 100
                
            # augment the train_set with selected indices from the lake
            if(feature == "classimb"):
                train_set, lake_set, true_lake_set, lake_ss = aug_train_subset(
                    train_set, lake_set, true_lake_set, subset, lake_subset_idxs, budget, True)  # aug train with random if budget is not filled
            else:
                train_set, lake_set, true_lake_set, lake_ss = aug_train_subset(
                    train_set, lake_set, true_lake_set, subset, lake_subset_idxs, budget)
            print("After augmentation, size of train_set: ", len(train_set),
                  " unlabeled set: ", len(lake_set), " val set: ", len(val_set))

            if not is_full:
                with torch.no_grad():
                    torch.cuda.empty_cache()
                return process_imgs(lake_ss, dataset_name)

#           Reinit train and lake loaders with new splits and reinit the model
            trainloader = torch.utils.data.DataLoader(
                train_set, batch_size=trn_batch_size, shuffle=True, pin_memory=True)
            model = create_model(model_name, 10, device,
                                 strategy_args['embedding_type'])
            optimizer = optimizer_without_scheduler(model, learning_rate)

        # Start training
        start_time = time.time()
        num_ep = 1
        while(full_trn_acc[i] < 0.99 and num_ep < 300):
            model.train()
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(
                    device, non_blocking=True)
                # Variables in Pytorch are differentiable.
                inputs, _ = Variable(inputs), Variable(inputs)
                # This will zero out the gradients for this batch.
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            full_trn_loss = 0
            full_trn_correct = 0
            full_trn_total = 0
            model.eval()
            with torch.no_grad():
                # Compute Train accuracy
                for inputs, targets in trainloader:
                    inputs, targets = inputs.to(device), targets.to(
                        device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    full_trn_loss += loss.item()
                    _, predicted = outputs.max(1)
                    full_trn_total += targets.size(0)
                    full_trn_correct += predicted.eq(targets).sum().item()
                full_trn_acc[i] = full_trn_correct / full_trn_total
                training_progress[str(i)][0].append(num_ep)
                training_progress[str(i)][1].append(full_trn_acc[i])
                num_ep += 1
            timing[i] = time.time() - start_time
        with torch.no_grad():
            final_val_predictions = []
            final_val_classifications = []
            for inputs, targets in valloader:  # Compute Val accuracy
                inputs, targets = inputs.to(device), targets.to(
                    device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                if(feature=="ood"): 
                    _, predicted = outputs[...,:-1].max(1)
                else:
                    _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                final_val_predictions += list(predicted.cpu().numpy())
                final_val_classifications += list(
                    predicted.eq(targets).cpu().numpy())

            final_tst_predictions = []
            final_tst_classifications = []
            for inputs, targets in tstloader:  # Compute test accuracy
                inputs, targets = inputs.to(device), targets.to(
                    device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                tst_loss += loss.item()
                if(feature=="ood"): 
                    _, predicted = outputs[...,:-1].max(1)
                else:
                    _, predicted = outputs.max(1)
                tst_total += targets.size(0)
                tst_correct += predicted.eq(targets).sum().item()
                final_tst_predictions += list(predicted.cpu().numpy())
                final_tst_classifications += list(
                    predicted.eq(targets).cpu().numpy())
            val_acc[i] = val_correct / val_total
            tst_acc[i] = tst_correct / tst_total
            val_losses[i] = val_loss
            fulltrn_losses[i] = full_trn_loss
            tst_losses[i] = tst_loss
            res_dict["test_acc"].append(tst_acc[i]*100)

    # Compute the statistics of the final model
    if(computeErrorLog):
        tst_err_log, val_err_log, val_class_err_idxs, final_metrics = find_err_per_class(
            test_set, val_set, final_val_classifications, final_val_predictions, final_tst_classifications, final_tst_predictions,  sf+"_"+str(bud), feature, num_cls)
        csvlog.append([100-x for x in tst_err_log])
        val_csvlog.append([100-x for x in val_err_log])
        res_dict["all_class_acc"] = csvlog
        res_dict["all_val_class_acc"] = val_csvlog

    # Print overall acc improvement and rare class acc improvement, show that TL selected relevant points in space, is possible show some images

    # epoch_len = np.max(training_progress["0"][0], training_progress["1"][0])
    # training_progress["0"][1] = training_progress["0"][1] + \
    #     [np.nan for _ in range(len(epoch_len - training_progress["0"][1]))]

    # training_progress["1"][1] = training_progress["1"][1] + \
    #     [np.nan for _ in range(len(epoch_len - training_progress["1"][1]))]

    d1 = {'Epoch': training_progress["0"][0],
          'Training Accuracy': training_progress["0"][1]}
    df1 = pd.DataFrame(data=d1)

    d2 = {'Epoch': training_progress["1"][0],
          'Training Accuracy': training_progress["1"][1]}
    df2 = pd.DataFrame(data=d2)
    
    with torch.no_grad():
        torch.cuda.empty_cache()
    
    return get_final_results(res_dict, sel_cls_idx) + [initial_metrics, final_metrics] + [df1, df2] + [process_imgs(lake_ss, dataset_name)] + [percentage_selection]


def process_imgs(lake_ss, dataset_name):
    new_lake_ss = []
    for i, x in enumerate(lake_ss):
        img = x[0].cpu().detach().numpy()
        img = img[0, :, :]
        img = img / (1 + max(np.max(img), -1*np.min(img)))
        new_lake_ss.append(img)

    if dataset_name == 'mnist':
        new_lake_ss = collage(new_lake_ss, f=6)
    else:
        new_lake_ss = collage(new_lake_ss, f=6)

    return new_lake_ss


def collage(l, f):
    for i in range(f):
        if i % 2 == 0:
            l = stack(l, np.vstack)
        else:
            l = stack(l, np.hstack)
    return l


def stack(l, fun):
    new_l = []
    for i, x in enumerate(l):
        if i % 2 == 0:
            new_l.append(x)
        else:
            try:
                new_l[-1] = fun((new_l[-1], x))
            except:
                new_l.append(x)
    return new_l
