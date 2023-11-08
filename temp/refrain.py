def select_for_refrain(self, budget):
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
        loss_func = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.7,backend="online")
        
        unlabeled_dataset_len=len(self.unlabeled_dataset)
        shuffled_indices = list(range(unlabeled_dataset_len))
        random.shuffle(shuffled_indices)
        sampler = customSampler(shuffled_indices)

        query_dataset_len = len(self.query_dataset)
        minibatch_size = self.args['batch_size'] if 'minibatch_size' in self.args else 4000
       
        num_batches = math.ceil(unlabeled_dataset_len/minibatch_size)
        if(self.args['verbose']):
            print('There are',unlabeled_dataset_len,'Unlabeled dataset')
        num_classes = len(torch.unique(torch.stack([item[1] for item in self.query_dataset])))
        classwise_simplex_query = []
        classwise_simplex_refrain = []
        for _ in range(num_classes):
            tensor = torch.ones(unlabeled_dataset_len, device=self.device)
            tensor = (tensor / unlabeled_dataset_len).clone().detach().requires_grad_(True)
            classwise_simplex_query.append(tensor)
        
        for _ in range(num_classes):
            tensor = torch.ones(unlabeled_dataset_len, device=self.device)
            tensor = (tensor / unlabeled_dataset_len).clone().detach().requires_grad_(True)
            classwise_simplex_refrain.append(tensor)



        label_to_simplex_query = {}
        unique_labels = torch.unique(torch.stack([item[1] for item in self.query_dataset]))
        
        for i, label in enumerate(unique_labels):
            label_to_simplex_query[label.item()] = classwise_simplex_query[i]
            

         # Hyperparameters for sinkhorn iterations
         #if self.args has lr, use that else use 0.001
        lr = self.args['lr'] if 'lr' in self.args else 0.001
        min_iteration=self.args['min_iteration'] if 'min_iteration' in self.args else 50
        
        step_size = self.args['step_size'] if 'step_size' in self.args else 10
           
        optimizer = torch.optim.Adam(classwise_simplex_query+classwise_simplex_refrain, lr=lr)
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
            
            for class_idx in range(num_classes):
                #print('entering classwisecalculation')
                #filter query dataset based on class_idx
                class_mask = torch.stack([item[1] for item in self.query_dataset]) == unique_labels[class_idx]
                #find length of query dataset for that class_idx
                num_query_instances = len(query_dataset_features[torch.nonzero(class_mask).squeeze()])
               
                # For every non-class_idx, select equal instances
                num_samples_per_class = num_query_instances // (num_classes - 1)
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
                simplex_query = classwise_simplex_query[class_idx]
                simplex_query.requires_grad = True
                #get simplex_refrain for that class
                simplex_refrain = classwise_simplex_refrain[class_idx]
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
                for class_idx in range(num_classes):
                    classwise_simplex_query[class_idx].data = self._proj_simplex(classwise_simplex_query[class_idx].data)
                    classwise_simplex_refrain[class_idx].data = self._proj_simplex(classwise_simplex_refrain[class_idx].data)
            
            print("Epoch:[", i,"],Avg loss: [{}]".format(loss),end="\r")
                    #break if loss is less than 1 or greater than -1
                    


            if((loss.item()<0.2 and loss.item()>-0.2) and i>min_iteration):
                break
                
        #once iterations are over or loss is less than 1, return the necessary indices
        

        # This list will store the desired tuples
        output = []
       
        
        # Merge all masked_simplex_query tensors
        merged_simplex = torch.cat([classwise_simplex_query[i].detach().cpu() for i in range(num_classes)])

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

        
        

        for iteridx,(class_idx, simplex_query) in enumerate(label_to_simplex_query.items()):
            
            # Get values from simplex_query and plot them
            #plt.hist(simplex_values, bins=np.linspace(0, max(simplex_values), 50), alpha=0.5, label=f'Class {class_idx}')
            
           
            # Mask out the values in simplex_query and simplex_refrain tensors
            # corresponding to selected indices
            masked_simplex_query = simplex_query.clone()
            masked_simplex_refrain = classwise_simplex_refrain[iteridx].clone()
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
        
        
        torch.cuda.empty_cache()
        return selected_indices,output