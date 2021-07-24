import math
import numpy as np

from torch.utils.data import Subset
from .strategy import Strategy

class PartitionStrategy(Strategy):
    
    """
    This strategy presents a wrapper around the other partitionable strategies within this library. Often, the unlabeled 
    dataset is very large, lending to sometimes intractable memory requirements for the operation of the other strategies. 
    Here, the unlabeled dataset is partitioned into a configurable number of sets. The wrapped strategy is then created using 
    all the relevant arguments provided to PartitionStrategy; however, the unlabeled_dataset argument receives one of the 
    partitions instead of the full unlabeled dataset. Using this newly created wrapped strategy that has one of the unlabeled 
    partitions, select() is called with a budget argument that is a fraction of the budget passed to PartitionStrategy's select().
    This fraction is proportional to the fraction between the partition size and the full unlabeled set size. Hence, the total 
    number of indices collected by PartitionStrategy's select() is equal to its budget argument.
    
    Parameters
    ----------
    labeled_dataset: torch.utils.data.Dataset
        The labeled dataset to be used in this strategy. For the purposes of selection, the labeled dataset is not used, 
        but it is provided to fit the common framework of the Strategy superclass.
    unlabeled_dataset: torch.utils.data.Dataset
        The unlabeled dataset to be used in this strategy. It is used in the selection process as described above.
        Importantly, the unlabeled dataset must return only a data Tensor; if indexing the unlabeled dataset returns a tuple of 
        more than one component, unexpected behavior will most likely occur.
    net: torch.nn.Module
        The neural network model to use for embeddings and predictions. Notably, all embeddings typically come from extracted 
        features from this network or from gradient embeddings based on the loss, which can be based on hypothesized gradients 
        or on true gradients (depending on the availability of the label).
    nclasses: int
        The number of classes being predicted by the neural network.
    args: dict
        A dictionary containing many configurable settings for this strategy and the wrapped strategy. 
        Each key-value pair is described below:
            'batch_size': int
                The batch size used internally for torch.utils.data.DataLoader objects. Default: 1
            'device': string
                The device to be used for computation. PyTorch constructs are transferred to this device. Usually is one 
                of 'cuda' or 'cpu'. Default: 'cuda' if a CUDA-enabled device is available; otherwise, 'cpu'
            'loss': function
                The loss function to be used in computations. Default: torch.nn.functional.cross_entropy
            'num_partitions': int
                The number of partitions to use for the unlabeled dataset. Default: 1
            'wrapped_strategy_class': class
                The class of the strategy to use for each partition. REQUIRED
    query_dataset: torch.utils.data.Dataset
        The query dataset to be used in the wrapped strategy. Not all wrapped strategies require this argument. Default: None
    private_dataset: torch.utils.data.Dataset
        The private dataset to be used in the wrapped strategy. Not all wrapped strategies require this argument. Default: None
    """
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}, query_dataset=None, private_dataset=None): #
        
        super(PartitionStrategy, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        
        if "num_partitions" not in args:
            self.num_partitions = 1
        else:
            self.num_partitions = args["num_partitions"]
            
        if "wrapped_strategy_class" not in args:
            raise ValueError("args dictionary requires 'wrapped_strategy_class' key")
            
        self.wrapped_strategy_class = args["wrapped_strategy_class"]
        self.query_dataset = query_dataset
        self.private_dataset = private_dataset

    def select(self, budget):
        
        # The number of partitions should be less than or equal to the budget.
        # This is because the budget is evenly divided among the partitions (roughly),
        # so having a smaller budget than the number of partitions results in one or 
        # more partitions having a 0 budget, which should not happen.
        if self.num_partitions > budget:
            raise ValueError("Budget cannot be less than the number of partitions!")
        
        # Furthermore, the number of partitions cannot be more than the size of the unlabeled set
        if self.num_partitions > len(self.unlabeled_dataset):
            raise ValueError("There cannot be more partitions than the size of the dataset!")
    
        # Calculate partition splits and budgets for each partition
        full_unlabeled_size = len(self.unlabeled_dataset)
        split_indices = [math.ceil(full_unlabeled_size * ((1+x) / self.num_partitions)) for x in range(self.num_partitions)]        
        partition_budget_splits = [math.ceil(budget * (split_index / full_unlabeled_size)) for split_index in split_indices]
                  
        beginning_split = 0
        
        selected_idx = []
        
        for i in range(self.num_partitions):
            
            end_split = split_indices[i]
            
            # Create a subset of the original unlabeled dataset as a partition.
            partition_index_list = list(range(beginning_split, end_split))
            current_partition = Subset(self.unlabeled_dataset, partition_index_list)
            
            # Calculate the budget for this partition
            if i == 0:
                partition_budget = partition_budget_splits[i]
            else:
                partition_budget = partition_budget_splits[i] - partition_budget_splits[i - 1]
                
            # With the new subset, create an instance of the wrapped strategy and call its select function.
            if(self.query_dataset != None and self.private_dataset != None):
                wrapped_strategy = self.wrapped_strategy_class(self.labeled_dataset, current_partition, self.query_dataset, self.private_dataset, self.model, self.target_classes, self.args)
            elif(self.query_dataset != None):
                wrapped_strategy = self.wrapped_strategy_class(self.labeled_dataset, current_partition, self.query_dataset, self.model, self.target_classes, self.args)
            elif(self.private_dataset != None):
                wrapped_strategy = self.wrapped_strategy_class(self.labeled_dataset, current_partition, self.private_dataset, self.model, self.target_classes, self.args)
            else:
                wrapped_strategy = self.wrapped_strategy_class(self.labeled_dataset, current_partition, self.model, self.target_classes, self.args)
            selected_partition_idxs = wrapped_strategy.select(partition_budget)
            
            # Use the partition_index_list to map the selected indices w/ respect to the current partition to the indices w/ respect to the dataset
            to_add_idxs = np.array(partition_index_list)[selected_partition_idxs]
            selected_idx.extend(to_add_idxs)
            beginning_split = end_split
            
        # Return the selected idx
        return selected_idx