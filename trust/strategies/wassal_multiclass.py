from .strategy import Strategy

import submodlib
import torch
from torch.autograd import Variable
from geomloss import SamplesLoss
from torch.utils.data import DataLoader, Dataset
import random
import math
from torchvision.models import resnet50, resnet18,resnet101

class customSampler(torch.utils.data.Sampler):
    def __init__(self, ind):
        self.ind = ind
        return
    def __iter__(self):
        return iter(self.ind)

class WASSAL_Multiclass(Strategy):
    
    """
    
    
    Parameters
    ----------
    labeled_dataset: torch.utils.data.Dataset
        The labeled dataset to be used in this strategy. For the purposes of selection, the labeled dataset is not used, 
        but it is provided to fit the common framework of the Strategy superclass.
    unlabeled_dataset: torch.utils.data.Dataset
        The unlabeled dataset to be used in this strategy. It is used in the selection process as described above.
        Importantly, the unlabeled dataset must return only a data Tensor; if indexing the unlabeled dataset returns a tuple of 
        more than one component, unexpected behavior will most likely occur.
    nclasses: int
        The number of classes being predicted by the neural network.
    args: dict
        A dictionary containing many configurable settings for this strategy. Each key-value pair is described below:
            
            - **b_size**: The batch size used internally for torch.utils.data.DataLoader objects. (int, optional)
            - **device**: The device to be used for computation. PyTorch constructs are transferred to this device. Usually is one of 'cuda' or 'cpu'. (string, optional)
            - **loss**: The loss function to be used in computations. (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional)
            - **optimizer**: The optimizer to use for submodular maximization. Can be one of 'NaiveGreedy', 'StochasticGreedy', 'LazyGreedy' and 'LazierThanLazyGreedy'. (string, optional)
            - **eta**: A magnification constant that is used in all but gcmi. It is used as a value of query-relevance vs diversity trade-off. Increasing eta tends to increase query-relevance while reducing query-coverage and diversity. (float)
            - **embedding_type**: The type of embedding to compute for similarity kernel computation. This can be either 'gradients' or 'features'. (string)
            - **verbose**: Gives a more verbose output when calling select() when True. (bool)
    """
    
    def __init__(self, labeled_dataset, unlabeled_dataset, query_dataset,net, nclasses, args={}): #
        #pretrained resnet18 as 
        #self.net = resnet50(pretrained=True)
        self.net=net
        #merge labeled and query dataset into query and non-query classes and finetune the resnet50
        
        
        super(WASSAL_Multiclass, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)        
        self.query_dataset = query_dataset
        self.args['verbose']=True
        
        

    def _proj_simplex(self,v):
        """
        v: PyTorch Tensor to be projected to a simplex

        Returns:
        w: PyTorch Tensor simplex projection of v
        """
        z = 1
        orig_shape = v.shape
        v = v.view(1, -1)
        shape = v.shape
        with torch.no_grad():
            mu = torch.sort(v, dim=1)[0]
            mu = torch.flip(mu, dims=(1,))
            cum_sum = torch.cumsum(mu, dim=1)
            j = torch.unsqueeze(torch.arange(1, shape[1] + 1, dtype=mu.dtype, device=mu.device), 0)
            rho = torch.sum(mu * j - cum_sum + z > 0.0, dim=1, keepdim=True) - 1.
            rho = rho.to(int)
            max_nn = cum_sum[torch.arange(shape[0]), rho[:, 0]]
            theta = (torch.unsqueeze(max_nn, -1) - z) / (rho.type(max_nn.dtype) + 1)
            w = torch.clamp(v - theta, min=0.0).view(orig_shape)
            return w

    def get_query_simplex(self):
        return self.simplex_query
    
    def select(self, budget):
        # venkat sir's code
        """
        Selects next set of points. Weights are all reset since in this 
        strategy the datapoints are removed
        
        Parameters
        ----------
        budget: int
            Number of data points to select for labeling
            
        Returns
        ----------
        idxs: list
            List of selected data point indices with respect to unlabeled_dataset
        """	
        
        #Get hyperparameters from args dict
        embedding_type = self.args['embedding_type'] if 'embedding_type' in self.args else "features"
        if(embedding_type=="features"):
            layer_name = self.args['layer_name'] if 'layer_name' in self.args else "avgpool"
        if(embedding_type=="gradients"):
            gradType = self.args['gradType'] if 'gradType' in self.args else "bias_linear"
        loss_func = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8)
        
        unlabeled_dataset_len=len(self.unlabeled_dataset)
        shuffled_indices = list(range(unlabeled_dataset_len))
        random.shuffle(shuffled_indices)
        sampler = customSampler(shuffled_indices)

        query_dataset_len = len(self.query_dataset)
        minibatch_size = 4000
       
        num_batches = math.ceil(unlabeled_dataset_len/minibatch_size)
        if(self.args['verbose']):
            print('There are',unlabeled_dataset_len,'Unlabeled dataset')
        self.num_classes = len(torch.unique(torch.stack([item[1] for item in self.query_dataset])))
        self.classwise_simplex_query = [torch.ones(unlabeled_dataset_len, requires_grad=True, device=self.device)/unlabeled_dataset_len for _ in range(self.num_classes)]
        self.classwise_indices = [[] for _ in range(self.num_classes)]
        for idx, (_, label) in enumerate(self.query_dataset):
            self.classwise_indices[label].append(idx)
        
        #multiclass selection
        classwise_final_indices = []
        for class_idx in range(self.num_classes):
            class_indices = self.classwise_indices[class_idx]
            for i in range(100):
                simplex_query = self.classwise_simplex_query[class_idx].detach()
                #uniform distribution of weights
                beta = torch.ones(query_dataset_len, requires_grad=False)/query_dataset_len
                unlabeled_dataloader = DataLoader(dataset=self.unlabeled_dataset, batch_size=minibatch_size, shuffle=False, sampler=sampler)
                query_dataloader = DataLoader(dataset=self.query_dataset, batch_size=len(self.query_dataset), shuffle=False)

                query_iter=iter(query_dataloader)

                query_imgs, _= next(query_iter)
        
        
                query_imgs=query_imgs.to(self.device)
                beta = beta.to(self.device)
                 # Hyperparameters for sinkhorn iterations
                lr = 0.0001
                step_size = 25
                m=0.9
                wd=5e-4
                optimizer = torch.optim.Adam([simplex_query], lr=lr)
        #optimizer = torch.optim.SGD([simplex_query], lr=lr,
         #                 momentum=m, weight_decay=wd)
         # Define the learning rate scheduler
                scheduler_query = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
                simplex_query.requires_grad = True
                # Create lists to store the loss values
        
                overall_loss=[]
        
                # Loop over the datasets 10 times
            
                simplex_query.grad = None  # Reset gradients at the beginning of each epoch
                batch_idx = 0
                # Initialize loss_avg as a tensor with requires_grad=True
                loss_avg = torch.tensor(0.0, requires_grad=True)
                
                optimizer.zero_grad()
                #batchwise WD calculation
                for unlabeled_imgs in unlabeled_dataloader:
                    
                    # Get the features using the pretrained model
                    if(embedding_type == "features"):
                        unlabeled_features = self.get_feature_embedding(unlabeled_imgs, True, layer_name)
                        query_features = self.get_feature_embedding(query_imgs, True, layer_name)
                    if(embedding_type == "gradients"):
                        unlabeled_features = self.get_grad_embedding(self.unlabeled_dataset, True, gradType)
                        query_features = self.get_grad_embedding(self.query_dataset, False, gradType)
            
                    unlabeled_features = unlabeled_features.view(unlabeled_features.shape[0], -1)
                    query_features = query_features.view(query_features.shape[0], -1)    
                    simplex_batch_query = simplex_query[batch_idx * unlabeled_dataloader.batch_size : (batch_idx + 1) * unlabeled_dataloader.batch_size]
                    #should we average or project?
                    simplex_batch_query = simplex_batch_query.clone() / simplex_batch_query.sum()

                    weights_batch = simplex_query[batch_idx*minibatch_size:(batch_idx+1)*minibatch_size]
                    simplex_batch_query=simplex_batch_query.to(self.device)
                    #unlabeled_imgs=unlabeled_imgs.to(self.device)
                    loss = loss_func(simplex_batch_query, unlabeled_features, beta, query_features)
                    #loss = loss_func(simplex_batch_query, unlabeled_imgs.view(len(unlabeled_imgs), -1), beta, query_imgs.view(len(query_imgs), -1))
                    overall_loss.append(loss.item())
                    
                    loss_avg = loss_avg + loss / num_batches
                    
                    batch_idx += 1
                    
               
                loss_avg.backward()
                optimizer.step()
                scheduler_query.step()
                
            
                with torch.no_grad():
                    simplex_query.data = self._proj_simplex(simplex_query.data)
                print("Epoch:[", i,"],Avg loss: [{}]".format(loss_avg),end="\r")
                if loss_avg.item() < 1:
                    break
            #store as an object variable    
            self.classwise_simplex_query[class_idx] = simplex_query

            sorted_simplex,indices=torch.sort(simplex_query,descending=True)
            inter_indices = torch.Tensor.tolist(indices[:budget])
            final_indices = [shuffled_indices[ind] for ind in inter_indices]
            classwise_final_indices.append((final_indices, simplex_query, class_idx))
            if(self.args['verbose']):
                print('length of unlabelled dataset:',unlabeled_dataset_len)
                print('Totals Probability of the budget:',str(torch.sum(sorted_simplex[:budget])))
                
        return  classwise_final_indices