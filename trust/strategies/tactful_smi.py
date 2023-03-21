from .strategy import Strategy
import numpy as np


from trust.utils.DB import Database
from trust.utils.resnet_feat import ResNetFeat

import torch
from torch import nn
from scipy import stats
import submodlib
import os

class TACTFUL_SMI(Strategy):
    
    """
    This strategy implements the Submodular Mutual Information (SMI) selection paradigm discuss in the paper 
    SIMILAR: Submodular Information Measures Based Active Learning In Realistic Scenarios :footcite:`kothawade2021similar`. In this selection 
    paradigm, points from the unlabeled dataset are chosen in such a way that the submodular mutual information 
    between this set of points and a provided query set is maximized. Doing so allows a practitioner to select 
    points from an unlabeled set that are SIMILAR to points that they have provided in a active learning query.
    
    These submodular mutual information functions rely on formulating embeddings for the points in the query set 
    and the unlabeled set. Once these embeddings are formed, one or more similarity kernels (depending on the 
    SMI function used) are formed from these embeddings based on a similarity metric. Once these similarity kernels 
    are formed, they are used in computing the value of each submodular mutual information function. Hence, common 
    techniques for submodular maximization subject to a cardinality constraint can be used, such as the naive greedy 
    algorithm, the lazy greedy algorithm, and so forth.
    
    In this framework, we set the cardinality constraint to be the active learning selection budget; hence, a list of 
    indices with a total length less than or equal to this cardinality constraint will be returned. Depending on the 
    maximization configuration, one can ensure that the length of this list will be equal to the cardinality constraint.
    
    Currently, five submodular mutual information functions are implemented: fl1mi, fl2mi, gcmi, logdetmi, and com. Each 
    function is obtained by applying the definition of a submodular mutual information function using common submodular 
    functions. Facility Location Mutual Information (fl1mi) models pairwise similarities of points in the query set to 
    points in the unlabeled dataset AND pairwise similarities of points within the unlabeled datasets. Another variant of 
    Facility Location Mutual Information (fl2mi) models pairwise similarities of points in the query set to points in 
    the unlabeled dataset ONLY. Graph Cut Mutual Information (gcmi), Log-Determinant Mutual Information (logdetmi), and 
    Concave-Over-Modular Mutual Information (com) are all obtained by applying the usual submodular function under this 
    definition. For more information-theoretic discussion, consider referring to the paper Submodular Combinatorial 
    Information Measures with Applications in Machine Learning :footcite:`iyer2021submodular`.
    
    Parameters
    ----------
    labeled_dataset: torch.utils.data.Dataset
        The labeled dataset to be used in this strategy. For the purposes of selection, the labeled dataset is not used, 
        but it is provided to fit the common framework of the Strategy superclass.
    unlabeled_dataset: torch.utils.data.Dataset
        The unlabeled dataset to be used in this strategy. It is used in the selection process as described above.
        Importantly, the unlabeled dataset must return only a data Tensor; if indexing the unlabeled dataset returns a tuple of 
        more than one component, unexpected behavior will most likely occur.
    query_dataset: torch.utils.data.Dataset
        The query dataset to be used in this strategy. It is used in the selection process as described above. Notably, 
        the query dataset should be labeled; hence, indexing the query dataset should return a data/label pair. This is 
        done in this fashion to allow for gradient embeddings.
    net: torch.nn.Module
        The neural network model to use for embeddings and predictions. Notably, all embeddings typically come from extracted 
        features from this network or from gradient embeddings based on the loss, which can be based on hypothesized gradients 
        or on true gradients (depending on the availability of the label).
    nclasses: int
        The number of classes being predicted by the neural network.
    args: dict
        A dictionary containing many configurable settings for this strategy. Each key-value pair is described below:
            'batch_size': int
                The batch size used internally for torch.utils.data.DataLoader objects. Default: 1
            'device': string
                The device to be used for computation. PyTorch constructs are transferred to this device. Usually is one 
                of 'cuda' or 'cpu'. Default: 'cuda' if a CUDA-enabled device is available; otherwise, 'cpu'
            'loss': function
                The loss function to be used in computations. Default: torch.nn.functional.cross_entropy
            'smi_function': string
                The submodular mutual information function to use in optimization. Must be one of 'fl1mi', 'fl2mi', 'gcmi', 
                'logdetmi', 'com'. REQUIRED
            'optimizer': string
                The optimizer to use for submodular maximization. Can be one of 'NaiveGreedy', 'StochasticGreedy', 
                'LazyGreedy' and 'LazierThanLazyGreedy'. Default: 'NaiveGreedy'
            'metric': string
                The similarity metric to use for similarity kernel computation. This can be either 'cosine' or 'euclidean'. 
                Default: 'cosine'
            'eta': float
                A magnification constant that is used in all but gcmi. It is used as a value of query-relevance vs diversity 
                trade-off. Increasing eta tends to increase query-relevance while reducing query-coverage and diversity. 
                Default: 1
            'embedding_type': string
                The type of embedding to compute for similarity kernel computation. This can be either 'gradients' or 
                'features'. Default: 'gradients'
            'gradType': string
                When 'embedding_type' is 'gradients', this defines the type of gradient to use. 'bias' creates gradients from 
                the loss function with respect to the biases outputted by the model. 'linear' creates gradients from the 
                loss function with respect to the last linear layer features. 'bias_linear' creates gradients from the 
                loss function using both. Default: 'bias_linear'
            'layer_name': string
                When 'embedding_type' is 'features', this defines the layer within the neural network that is used to extract 
                feature embeddings. Namely, this argument must be the name of a module used in the forward() computation of 
                the model. Default: 'avgpool'
            'stopIfZeroGain': bool
                Controls if the optimizer should cease maximization if there is zero gain in the submodular objective.
                Default: False
            'stopIfNegativeGain': bool
                Controls if the optimizer should cease maximization if there is negative gain in the submodular objective.
                Default: False
            'verbose': bool
                Gives a more verbose output when calling select() when True. Default: False
    """
    
    def __init__(self, labeled_dataset=None, unlabeled_dataset=None, net=None, f_net=None, nclasses=None, args={}): #
        
        super(TACTFUL_SMI, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)        

    def select(self, budget):
        """
        Selects a set of points from the unlabeled dataset to label based on this strategy's methodology.
        
        Parameters
        ----------
        budget: int
            Number of points to choose from the unlabeled dataset
        
        Returns
        ----------
        chosen: list
            List of selected data point indices with respect to the unlabeled dataset
        """ 
        model_path = self.args['model_path']
        clazz = self.args['class']
        iteration = self.args['iteration']
        eta = self.args['eta']
        optimizer = self.args['optimizer'] if 'optimizer' in self.args else 'NaiveGreedy'
        metric = self.args['metric'] if 'metric' in self.args else 'cosine'
        eta = self.args['eta'] if 'eta' in self.args else 1
        stopIfZeroGain = self.args['stopIfZeroGain'] if 'stopIfZeroGain' in self.args else False
        stopIfNegativeGain = self.args['stopIfNegativeGain'] if 'stopIfNegativeGain' in self.args else False
        verbose = self.args['verbose'] if 'verbose' in self.args else False
        
        db2 = Database(dir=os.path.join(model_path, "query_images"), csv=os.path.join(
                model_path, "data_query.csv"))
        
        f_model = ResNetFeat()
        query_set_embeddings = f_model.make_samples(
                db2, cache_path="query-"+str(iteration))
        
        db = Database(dir=os.path.join(model_path, "lake_images"),
                            csv=os.path.join(model_path, "data.csv"))

        lake_set_embeddings = f_model.make_samples(
                db, cache_path="lake-" + str(iteration), RES_model="resnet101", pick_layer="avg")
        

        query_embedding = []

        for idx, query in enumerate(query_set_embeddings, start=1):
            q_img, q_cls, q_hist = query['img'], query['cls'], query['hist']
            if q_cls in clazz:
                query_embedding.append(q_hist)

        lake_embedding = []
        lake_image_list = []
        for idx, sample in enumerate(lake_set_embeddings, start=1):
            s_img, s_cls, s_hist = sample['img'], sample['cls'], sample['hist']
            lake_embedding.append(s_hist)
            lake_image_list.append(s_img)
        

        if (len(lake_embedding) < budget):
            budget = len(lake_embedding) - 1

        lake_embedding = torch.tensor(lake_embedding)
        query_embedding = torch.tensor(query_embedding)
        if(self.args['smi_function']=='fl1mi'):
            obj = submodlib.FacilityLocationMutualInformationFunction(n=lake_embedding.shape[0],
                                                                      num_queries=query_embedding.shape[0], 
                                                                      data=lake_embedding , 
                                                                      queryData=query_embedding, 
                                                                      magnificationEta=eta)

        if(self.args['smi_function']=='fl2mi'):
            obj = submodlib.FacilityLocationVariantMutualInformationFunction(n=lake_embedding.shape[0],
                                                                      num_queries=query_embedding.shape[0], 
                                                                      data=lake_embedding,
                                                                      queryData=query_embedding,
                                                                      queryDiversityEta=eta)
        
        if(self.args['smi_function']=='com'):
            from submodlib_cpp import ConcaveOverModular
            obj = submodlib.ConcaveOverModularFunction(n=lake_embedding.shape[0],
                                                                      num_queries=query_embedding.shape[0], 
                                                                      data=lake_embedding,
                                                                      queryData=query_embedding, 
                                                                      queryDiversityEta=eta,
                                                                      mode=ConcaveOverModular.logarithmic)
        if(self.args['smi_function']=='gcmi'):
            obj = submodlib.GraphCutMutualInformationFunction(n=lake_embedding.shape[0],
                                                                      num_queries=query_embedding.shape[0],
                                                                      data=lake_embedding,
                                                                      queryData=query_embedding, 
                                                                      queryDiversityEta=eta, 
                                                                      metric=metric)
        if(self.args['smi_function']=='logdetmi'):
            lambdaVal = self.args['lambdaVal'] if 'lambdaVal' in self.args else 1
            obj = submodlib.LogDeterminantMutualInformationFunction(n=lake_embedding.shape[0],
                                                                    num_queries=query_embedding.shape[0],
                                                                    data=lake_embedding,  
                                                                    queryData=query_embedding,                                                                    
                                                                    magnificationEta=eta,
                                                                    lambdaVal=lambdaVal)
        greedyList = obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=stopIfZeroGain, 
                              stopIfNegativeGain=stopIfNegativeGain, verbose=verbose)
        greedyIndices = [x[0] for x in greedyList]
        return lake_image_list, greedyIndices