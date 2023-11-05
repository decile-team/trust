from .strategy import Strategy

import submodlib
import torch
from torch.autograd import Variable
from geomloss import SamplesLoss
from torch.utils.data import DataLoader, Dataset
import random
import math
from torchvision.models import resnet50, resnet18,resnet101
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
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
        #self.pretrained_model=True 
        #self.net = resnet18(pretrained=True)
        self.pretrained_model=False 
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
    
    def _compute_features(self, dataset, embedding_type, layer_name=None, gradType=None,isLabeled=False):
        """Helper method to compute features for a dataset."""
        dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
        features = []
         # Ensure model is in evaluation mode for consistent feature extraction
        self.model.eval()
        with torch.no_grad():
            for batch_idx, inputs in enumerate(dataloader):
                if(isLabeled):
                    images, _ = inputs
                else:
                    images = inputs
                if embedding_type == "features":
                    batch_features= self.get_feature_embedding(images, True, layer_name).view(len(images), -1)
                elif embedding_type == "gradients":
                    batch_features= self.get_grad_embedding(images, False, gradType).view(len(images), -1)
                else:
                    raise ValueError("Unknown embedding type.")
                 # Move the batch features to CPU memory to free up GPU memory
                features.append(batch_features.cpu())
        # Stack all the features and move to the desired device, if needed
        features = torch.vstack(features)
        return features
    #update based on we needed AL model or pretrained model
    def update_model(self, clf):
        if not self.pretrained_model:
            self.model = clf

    def select_only_for_query(self,budget):
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
        gradType=None
        if(embedding_type=="gradients"):
            gradType = self.args['gradType'] if 'gradType' in self.args else "bias_linear"
        loss_func = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9,backend="online")
        
        unlabeled_dataset_len=len(self.unlabeled_dataset)
        shuffled_indices = list(range(unlabeled_dataset_len))
        random.shuffle(shuffled_indices)
        sampler = customSampler(shuffled_indices)

        query_dataset_len = len(self.query_dataset)
        minibatch_size = self.args['batch_size'] if 'batch_size' in self.args else 4000
        minibatch_size = 10000
       
        num_batches = math.ceil(unlabeled_dataset_len/minibatch_size)
        if(self.args['verbose']):
            print('There are',unlabeled_dataset_len,'Unlabeled dataset')
        self.classwise_simplex_query = []
        self.classwise_simplex_refrain = []
        for _ in range(self.num_classes):
            tensor = torch.ones(unlabeled_dataset_len, device=self.device)
            tensor = (tensor / unlabeled_dataset_len).clone().detach().requires_grad_(True)
            self.classwise_simplex_query.append(tensor)
        
        for _ in range(self.num_classes):
            tensor = torch.zeros(unlabeled_dataset_len, device=self.device)
            tensor = (tensor / unlabeled_dataset_len).clone().detach().requires_grad_(True)
            self.classwise_simplex_refrain.append(tensor)



        self.label_to_simplex_query = {}
        unique_labels = torch.unique(torch.stack([item[1] for item in self.query_dataset]))
        
        for i, label in enumerate(unique_labels):
            self.label_to_simplex_query[label.item()] = self.classwise_simplex_query[i]
            

         # Hyperparameters for sinkhorn iterations
         #if self.args has lr, use that else use 0.001
        lr = self.args['lr'] if 'lr' in self.args else 0.001
        min_iteration=self.args['min_iteration'] if 'min_iteration' in self.args else 50
        
        step_size = self.args['step_size'] if 'step_size' in self.args else 10
           
        optimizer = torch.optim.Adam(self.classwise_simplex_query, lr=lr)
        #optimizer = torch.optim.SGD(self.classwise_simplex_query+self.classwise_simplex_refrain, lr=lr)

        scheduler_query = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
        
        # 1. Precompute features for query_dataset
        query_dataset_features = self._compute_features(self.query_dataset, embedding_type, layer_name, gradType,True)
        query_dataset_len = len(self.query_dataset)
    
        # 2. Precompute features for unlabeled_dataset
        unlabeled_dataset_features = self._compute_features(self.unlabeled_dataset, embedding_type, layer_name, gradType,False)
        unlabeled_dataset_len = len(self.unlabeled_dataset)
        
        #multiclass selection
        #if self.args has iterations, use that else use 100
        iterations = self.args['iterations'] if 'iterations' in self.args else 100

        #first get query and refrain params ready
        
        for i in range(iterations):
            # Create lists to store the loss values           
           
            #print('entering iterations')
            # Initialize total loss as a tensor with requires_grad=True
            loss = 0.0
            total_loss=0.0
            optimizer.zero_grad()
            #calculate loss classwise in query dataset
            
            for class_idx in range(self.num_classes):
                #print('entering classwisecalculation')
                #filter query dataset based on class_idx
                class_mask = (torch.stack([item[1] for item in self.query_dataset]) == unique_labels[class_idx]).to(self.device)
                #find length of query dataset for that class_idx
                num_query_instances = len(query_dataset_features[torch.nonzero(class_mask).squeeze()])
               
                #for only one class, no refrain
                
    # For every non-class_idx, select equal instances
                query_features = query_dataset_features[torch.nonzero(class_mask).squeeze()]
                query_features=query_features.detach()
            
                query_features=query_features.to(self.device)
                #query_imgs=query_imgs.to(self.device)
                beta = torch.ones(len(query_features), requires_grad=False)/len(query_features)
                beta=beta.to(self.device)
                #get simplex_query for that class
                simplex_query = self.classwise_simplex_query[class_idx]
                
                unlabeled_dataloader = DataLoader(dataset=self.unlabeled_dataset, batch_size=minibatch_size, shuffle=False, sampler=sampler)
                loss_avg_query=0.0
                loss_avg_refrain=0.0
                loss_avg_query_refrain=0.0
                #calc num_batches
                num_batches = math.ceil(unlabeled_dataset_len/minibatch_size)
                #batchiwise WD calculation
            
                for batch_idx,unlabeled_imgs in enumerate(unlabeled_dataloader):
                    #print('entering batchwise')
                    # Get the features using the pretrained model
                
                
                # Handle the last batch size
                    current_batch_size = len(unlabeled_imgs)
                #if the current batch size is less than minibatch size, then we need to adjust the simplex batch query and simplex batch refrain
                    if(current_batch_size<minibatch_size):
                        diff=minibatch_size-current_batch_size
                    #for beginning index 0 add 1 to diff
                        
                        begindex=(batch_idx*minibatch_size)-diff
                        endindex=((batch_idx+1)*minibatch_size)-diff
                    else:
                        begindex=batch_idx*minibatch_size
                        endindex=(batch_idx+1)*minibatch_size
                #simplex batch query
                    simplex_batch_query = simplex_query[begindex : endindex]
                #should we average or project?
                    if(simplex_batch_query.sum()!=0):
                        simplex_batch_query = simplex_batch_query.clone() / simplex_batch_query.sum()
                    #simplex_batch_query.requires_grad = True
                    simplex_batch_query=simplex_batch_query.to(self.device)
                    #get minibatch unlabeled features
                    unlabeled_features = unlabeled_dataset_features[begindex : endindex]
                    
                    
                    
                    unlabeled_features=unlabeled_features.to(self.device)
                    loss_avg_query=loss_avg_query+(loss_func(simplex_batch_query, unlabeled_features, beta, query_features) / num_batches)
                    
                        
                        
            
            #once all batches are done, calculate average loss
            
                total_loss=loss_avg_query+loss_avg_refrain-0.3*loss_avg_query_refrain
                #add to the total loss
                loss = loss + total_loss

            #once one iteration is done for class, do backward and step
            loss.backward()
            optimizer.step()
            scheduler_query.step()
            #project to simplex
                           
            with torch.no_grad():
                for class_idx in range(self.num_classes):
                    self.classwise_simplex_query[class_idx].data = self._proj_simplex(self.classwise_simplex_query[class_idx].data)
                    
            print("Epoch:[", i,"],Avg loss: [{}]".format(loss),end="\r")
            #break if loss is less than 1 or greater than -1
                    


            if((loss.item()<0.2 and loss.item()>-0.2) and i>min_iteration):
                break
                
        #once iterations are over or loss is less than 1, return the necessary indices
        

        # This list will store the desired tuples
        output = []
       
        
        # Merge all masked_simplex_query tensors
        merged_simplex = torch.cat([self.classwise_simplex_query[i].detach().cpu() for i in range(self.num_classes)])

        # Find non-zero elements and sort them in ascending order
        non_zero_indices = torch.nonzero(merged_simplex).squeeze()
        sorted_indices = torch.argsort(merged_simplex[non_zero_indices], descending=False).cpu().numpy().tolist()
        # Convert sorted indices to original indices
        original_indices = [idx % len(self.unlabeled_dataset) for idx in sorted_indices]
        #select unqiue indices unti budget is reached
        selected_indices = []
        selected_indices_set = set()
        for idx in original_indices:
            if len(selected_indices) >= budget:
                break
            if idx not in selected_indices_set:
                selected_indices.append(idx)
                selected_indices_set.add(idx)

        
        

        for iteridx,(class_idx, simplex_query) in enumerate(self.label_to_simplex_query.items()):
            
            # Get values from simplex_query and plot them
            #plt.hist(simplex_values, bins=np.linspace(0, max(simplex_values), 50), alpha=0.5, label=f'Class {class_idx}')
            
           
            # Mask out the values in simplex_query and simplex_refrain tensors
            # corresponding to selected indices
            masked_simplex_query = simplex_query.clone()
            masked_simplex_refrain = self.classwise_simplex_refrain[iteridx].clone()
            for idx in selected_indices:
                masked_simplex_query[idx] = 0
                masked_simplex_refrain[idx] = 0
            
            # Each tuple contains the sorted indices, the simplex_query tensor, and the class_idx
            output.append((masked_simplex_query.detach().cpu(), masked_simplex_refrain.detach().cpu(), class_idx))
            # Update the set with the indices selected for the current class
            
        
        # plt.title('Distribution of Simplexes for Each Class')
        # plt.xlabel('Simplex Value')
        # plt.ylabel('Count')
        # plt.legend(loc='upper right')
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig('simplex_distribution.png')
        
        #free all GPUs
        
        with torch.no_grad():
            for tensor in [query_dataset_features, unlabeled_dataset_features] + self.classwise_simplex_query + self.classwise_simplex_refrain:
                del tensor
        torch.cuda.empty_cache()

        return selected_indices,output


    def select_for_query_refrain(self, budget):
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
        gradType=None
        if(embedding_type=="gradients"):
            gradType = self.args['gradType'] if 'gradType' in self.args else "bias_linear"
        loss_func = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.4,backend="online")
        
        unlabeled_dataset_len=len(self.unlabeled_dataset)
        shuffled_indices = list(range(unlabeled_dataset_len))
        random.shuffle(shuffled_indices)
        sampler = customSampler(shuffled_indices)

        query_dataset_len = len(self.query_dataset)
        minibatch_size = self.args['minibatch_size'] if 'minibatch_size' in self.args else 4000
       
        num_batches = math.ceil(unlabeled_dataset_len/minibatch_size)
        if(self.args['verbose']):
            print('There are',unlabeled_dataset_len,'Unlabeled dataset')
        self.num_classes = len(torch.unique(torch.stack([item[1] for item in self.query_dataset])))
        self.classwise_simplex_query = []
        self.classwise_simplex_refrain = []
        for _ in range(self.num_classes):
            tensor = torch.ones(unlabeled_dataset_len, device=self.device)
            tensor = (tensor / unlabeled_dataset_len).clone().detach().requires_grad_(True)
            self.classwise_simplex_query.append(tensor)
        
        for _ in range(self.num_classes):
            tensor = torch.ones(unlabeled_dataset_len, device=self.device)
            tensor = (tensor / unlabeled_dataset_len).clone().detach().requires_grad_(True)
            self.classwise_simplex_refrain.append(tensor)



        self.label_to_simplex_query = {}
        unique_labels = torch.unique(torch.stack([item[1] for item in self.query_dataset]))
        
        for i, label in enumerate(unique_labels):
            self.label_to_simplex_query[label.item()] = self.classwise_simplex_query[i]
            

         # Hyperparameters for sinkhorn iterations
         #if self.args has lr, use that else use 0.001
        lr = self.args['lr'] if 'lr' in self.args else 0.001
        min_iteration=self.args['min_iteration'] if 'min_iteration' in self.args else 50
        
        step_size = self.args['step_size'] if 'step_size' in self.args else 10
           
        optimizer = torch.optim.Adam(self.classwise_simplex_query+self.classwise_simplex_refrain, lr=lr)
        #optimizer = torch.optim.SGD(self.classwise_simplex_query+self.classwise_simplex_refrain, lr=lr)

        scheduler_query = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
        
        # 1. Precompute features for query_dataset
        query_dataset_features = self._compute_features(self.query_dataset, embedding_type, layer_name, gradType,True)
        query_dataset_len = len(self.query_dataset)
    
        # 2. Precompute features for unlabeled_dataset
        unlabeled_dataset_features = self._compute_features(self.unlabeled_dataset, embedding_type, layer_name, gradType,False)
        unlabeled_dataset_len = len(self.unlabeled_dataset)
        
        #multiclass selection
        #if self.args has iterations, use that else use 100
        iterations = self.args['iterations'] if 'iterations' in self.args else 100

        #first get query and refrain params ready
        
        for i in range(iterations):
            # Create lists to store the loss values           
           
            #print('entering iterations')
            # Initialize total loss as a tensor with requires_grad=True
            loss = 0.0
            total_loss=0.0
            optimizer.zero_grad()
            #calculate loss classwise in query dataset
            
            for class_idx in range(self.num_classes):
                #print('entering classwisecalculation')
                #filter query dataset based on class_idx
                class_mask = torch.stack([item[1] for item in self.query_dataset]) == unique_labels[class_idx]
                #find length of query dataset for that class_idx
                num_query_instances = len(query_dataset_features[torch.nonzero(class_mask).squeeze()])
               
                # For every non-class_idx, select equal instances
                num_samples_per_class = num_query_instances // (self.num_classes - 1)
                refrain_indices = []
                for other_class_idx in unique_labels:
                    
                    if other_class_idx != class_idx:
                        other_class_mask = torch.stack([item[1] for item in self.query_dataset]) == unique_labels[other_class_idx]
                        other_class_indices = torch.nonzero(other_class_mask).squeeze().tolist()
                        if(len(other_class_indices)>num_samples_per_class):
                            # Randomly sample indices without replacement
                            selected_indices = random.sample(other_class_indices, num_samples_per_class)
                            refrain_indices.extend(selected_indices)
                        else:
                            refrain_indices.extend(other_class_indices)
                 #Extract refrain_features
                query_features = query_dataset_features[torch.nonzero(class_mask).squeeze()]
                query_features=query_features.detach()
            
                refrain_features = query_dataset_features[torch.tensor(refrain_indices)]
                refrain_features=refrain_features.detach()
                query_features=query_features.to(self.device)
                refrain_features=refrain_features.to(self.device)
                #query_imgs=query_imgs.to(self.device)
                beta = torch.ones(len(query_features), requires_grad=False)/len(query_features)
                beta=beta.to(self.device)
                gamma = torch.ones(len(refrain_features), requires_grad=False)/len(refrain_features)
                gamma=gamma.to(self.device)
                #get simplex_query for that class
                simplex_query = self.classwise_simplex_query[class_idx]
                simplex_query.requires_grad = True
                #get simplex_refrain for that class
                simplex_refrain = self.classwise_simplex_refrain[class_idx]
                simplex_refrain.requires_grad=True
                #unlabeled_dataloader = DataLoader(dataset=self.unlabeled_dataset, batch_size=minibatch_size, shuffle=False, sampler=sampler)
                loss_avg_query=0.0
                loss_avg_refrain=0.0
                loss_avg_query_refrain=0.0
                #calc num_batches
                num_batches = math.ceil(unlabeled_dataset_len/minibatch_size)
                #batchiwise WD calculation
            
                for batch_idx in range(num_batches):
                    #print('entering batchwise')
                    # Get the features using the pretrained model
                
                
                # Handle the last batch size
                    current_batch_size = len(simplex_query[batch_idx*minibatch_size:])
                #if the current batch size is less than minibatch size, then we need to adjust the simplex batch query and simplex batch refrain
                    
                    if(current_batch_size<minibatch_size) and batch_idx !=  0:
                        diff=minibatch_size-current_batch_size
                    #for beginning index 0 add 1 to diff
                        
                        begindex=(batch_idx*minibatch_size)-diff
                        endindex=((batch_idx+1)*minibatch_size)-diff
                    else:
                        begindex=batch_idx*minibatch_size
                        endindex=(batch_idx+1)*minibatch_size
                #simplex batch query
                    simplex_batch_query = simplex_query[begindex : endindex]
                #should we average or project?
                    if(simplex_batch_query.sum()!=0):
                        simplex_batch_query = simplex_batch_query.clone() / simplex_batch_query.sum()
                    #simplex_batch_query.requires_grad = True
                    simplex_batch_query=simplex_batch_query.to(self.device)
                    simplex_batch_refrain = simplex_refrain[begindex : endindex]
                    #should we average or project?
                    if(simplex_batch_refrain.sum()!=0):
                        simplex_batch_refrain = simplex_batch_refrain.clone() / simplex_batch_refrain.sum()
                    #simplex_batch_refrain.requires_grad = True
                    simplex_batch_refrain=simplex_batch_refrain.to(self.device)
                    #get minibatch unlabeled features
                    unlabeled_features = unlabeled_dataset_features[begindex : endindex]
                    
                    
                    
                    unlabeled_features=unlabeled_features.to(self.device)
                    loss_avg_query=loss_avg_query+(loss_func(simplex_batch_query, unlabeled_features, beta, query_features) / num_batches)
                    
                    loss_avg_refrain=loss_avg_refrain+(loss_func(simplex_batch_refrain, unlabeled_features, gamma, refrain_features) / num_batches)
                    loss_avg_query_refrain=loss_avg_query_refrain+(loss_func(simplex_batch_query, unlabeled_features, simplex_batch_refrain, unlabeled_features) / num_batches)
                    
            
            #once all batches are done, calculate average loss
            
                total_loss=loss_avg_query+loss_avg_refrain-0.3*loss_avg_query_refrain
                #add to the total loss
                loss = loss + total_loss

            #once one iteration is done for class, do backward and step
            loss.backward()
            optimizer.step()
            scheduler_query.step()
            #project to simplex
                           
            with torch.no_grad():
                for class_idx in range(self.num_classes):
                    self.classwise_simplex_query[class_idx].data = self._proj_simplex(self.classwise_simplex_query[class_idx].data)
                    self.classwise_simplex_refrain[class_idx].data = self._proj_simplex(self.classwise_simplex_refrain[class_idx].data)
            
            print("Epoch:[", i,"],Avg loss: [{}]".format(loss),end="\r")
                    #break if loss is less than 1 or greater than -1
                    


            if((loss.item()<0.2 and loss.item()>-0.2) and i>min_iteration):
                break
                
        #once iterations are over or loss is less than 1, return the necessary indices
        

        # This list will store the desired tuples
        output = []
       
        
        # Merge all masked_simplex_query tensors
        merged_simplex = torch.cat([self.classwise_simplex_query[i].detach().cpu() for i in range(self.num_classes)])

        # Find non-zero elements and sort them in ascending order
        non_zero_indices = torch.nonzero(merged_simplex).squeeze()
        sorted_indices = torch.argsort(merged_simplex[non_zero_indices], descending=False).cpu().numpy().tolist()
        # Convert sorted indices to original indices
        original_indices = [idx % len(self.unlabeled_dataset) for idx in sorted_indices]
        #select unqiue indices unti budget is reached
        selected_indices = []
        selected_indices_set = set()
        for idx in original_indices:
            if len(selected_indices) >= budget:
                break
            if idx not in selected_indices_set:
                selected_indices.append(idx)
                selected_indices_set.add(idx)

        
        

        for iteridx,(class_idx, simplex_query) in enumerate(self.label_to_simplex_query.items()):
            
            # Get values from simplex_query and plot them
            #plt.hist(simplex_values, bins=np.linspace(0, max(simplex_values), 50), alpha=0.5, label=f'Class {class_idx}')
            
           
            # Mask out the values in simplex_query and simplex_refrain tensors
            # corresponding to selected indices
            masked_simplex_query = simplex_query.clone()
            masked_simplex_refrain = self.classwise_simplex_refrain[iteridx].clone()
            for idx in selected_indices:
                masked_simplex_query[idx] = 0
                masked_simplex_refrain[idx] = 0
            
            # Each tuple contains the sorted indices, the simplex_query tensor, and the class_idx
            output.append((masked_simplex_query.detach().cpu(), masked_simplex_refrain.detach().cpu(), class_idx))
            # Update the set with the indices selected for the current class
            
        
        # plt.title('Distribution of Simplexes for Each Class')
        # plt.xlabel('Simplex Value')
        # plt.ylabel('Count')
        # plt.legend(loc='upper right')
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig('simplex_distribution.png')
        
        #free all GPUs
        with torch.no_grad():
            for tensor in [query_dataset_features, unlabeled_dataset_features] + self.classwise_simplex_query + self.classwise_simplex_refrain:
                del tensor
        torch.cuda.empty_cache()
        return selected_indices,output

    def select(self, budget):
        self.num_classes = len(torch.unique(torch.stack([item[1] for item in self.query_dataset])))
        if(self.num_classes<=2):
            return self.select_only_for_query(budget)
        else:
            return self.select_for_query_refrain(budget)