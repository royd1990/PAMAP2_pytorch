# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import torch as T
import utils.ts_processor as tsp
from utils.helper import get_device_id

from abc import ABC, abstractmethod

device_id = get_device_id(T.cuda.is_available())
device = T.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def stratified_split_fn(X,y):

    train_ind, test_ind, train_moe, test_moe = 0.7, 0.2, 0.05, 0.05
    class_set   = np.unique(y)
    
    data_train_ind  = []
    data_test_ind  = []
    data_train_moe   = []
    data_test_moe = []
    
    labels_train_ind  = []
    labels_test_ind  = []
    labels_train_moe   = []
    labels_test_moe = []
    
    
    for class_i in class_set:
        data_inds = np.where(y==class_i)
        data_i = X[data_inds[0], ...]
        y_i = y[data_inds[0],...]
        N_i = len(data_inds[0])
        
        N_i_train_ind = int(N_i*train_ind)
        N_i_test_ind = int(N_i*test_ind)
        N_i_train_moe = int(N_i*train_moe)
        N_i_test_moe = int(N_i*test_moe)

        data_train_ind.append(data_i[:N_i_train_ind])
        data_test_ind.append(data_i[N_i_train_ind:N_i_train_ind+N_i_test_ind])
        data_train_moe.append(data_i[N_i_train_ind+N_i_test_ind:N_i_train_ind+N_i_test_ind+N_i_train_moe])
        data_test_moe.append(data_i[N_i_train_ind+N_i_test_ind+N_i_train_moe:])

        labels_train_ind.append(y_i[:N_i_train_ind])
        labels_test_ind.append(y_i[N_i_train_ind:N_i_train_ind+N_i_test_ind])
        labels_train_moe.append(y_i[N_i_train_ind+N_i_test_ind:N_i_train_ind+N_i_test_ind+N_i_train_moe])
        labels_test_moe.append(y_i[N_i_train_ind+N_i_test_ind+N_i_train_moe:])

    data_train_ind = np.concatenate(data_train_ind)
    data_test_ind = np.concatenate(data_test_ind)
    data_train_moe = np.concatenate(data_train_moe)
    data_test_moe = np.concatenate(data_test_moe)
    
    
    labels_train_ind = np.concatenate(labels_train_ind)
    labels_test_ind = np.concatenate(labels_test_ind)
    labels_train_moe = np.concatenate(labels_train_moe)
    labels_test_moe = np.concatenate(labels_test_moe)
    
    return data_train_ind,data_test_ind,data_train_moe,data_test_moe,labels_train_ind,labels_test_ind,labels_train_moe,labels_test_moe

class TSDataset(T.utils.data.Dataset):
    
    def __init__(self, X,y):
        
        self.X = T.tensor(X,dtype=T.float32).to(device)
        self.y = T.tensor(y,dtype=T.int64).to(device)
        
    def __len__(self):
        return len(self.X)#,len(self.X_valid),len(self.X_test)
    
    def __getitem__(self, idx):
        if T.is_tensor(idx):
          idx = idx.tolist()
        features = self.X[idx]
        labels = self.y[idx]
        sample = \
          { 'features' : features, 'labels' : labels }
        return sample
    
class Strategy(ABC):
    
    @abstractmethod
    def load_data(self,data):
        pass


class LoadStrategyA(Strategy):
    
    def load_data(self,data,seq_length,overlap,batch_size):
        tsp_obj = tsp.ts_processor(seq_length, overlap)
        X_train = data['X_train']
        X_valid = data['X_valid']
        X_test = data['X_test']
        y_train = data['y_train'].reshape(-1)
        y_valid = data['y_valid'].reshape(-1)
        y_test = data['y_test'].reshape(-1)
        
        X_train_processed, y_train_processed = tsp_obj.process_standard_ts(X_train, y_train)
        X_valid_processed, y_valid_processed = tsp_obj.process_standard_ts(X_valid, y_valid)
        X_test_processed, y_test_processed = tsp_obj.process_standard_ts(X_test, y_test)
        
        y_train = y_train_processed#pd.get_dummies( y_train_processed , prefix='labels' )
        y_valid = y_valid_processed#pd.get_dummies( y_valid_processed , prefix='labels' )
        y_test = y_test_processed#pd.get_dummies( y_test_processed , prefix='labels' )
        
        return X_train_processed,X_valid_processed,X_test_processed,y_train,y_valid,y_test

class LoadStrategySingle(Strategy):

    def load(self,data,seq_length,overlap,batch_size):
        # self.batch_size = batch_size
        tsp_obj = tsp.ts_processor(seq_length, overlap)
        X = data['X']
        y = data['y']
        print(X.shape,y.shape)
        X_processed,y = tsp_obj.process_standard_ts(X,y)

        return X_processed,None,None,y,None,None

class LoadStrategyD(Strategy):
    
        
    def load_data(self,data,seq_length,overlap,batch_size):
        # self.batch_size = batch_size
        tsp_obj = tsp.ts_processor(seq_length, overlap)
        X = data['X']
        y = data['y']
        X_processed,y = tsp_obj.process_standard_ts(X,y)
    
        # data_train_ind,data_test_ind,data_train_moe,data_test_moe,labels_train_ind,labels_test_ind,labels_train_moe,labels_test_moe=stratified_split_fn(X_processed,y)

        print(X_processed.shape, y.shape)

        return X_processed,None,None,y,None,None

class LoadStrategyBlank(Strategy):
    def load_data(self,data,seq_length,overlap,batch_size):
        # self.batch_size = batch_size
        X = data['X']
        y = data['y']
        print(X.shape, y.shape)

        return X,None,None,y,None,None

class LoadStrategyCNN(Strategy):
    def load_data(self,data,seq_length,overlap,batch_size):
        # self.batch_size = batch_size
        X = data['X']
        y = data['y']
        x_shape = X.shape
        X = np.reshape(X.astype(float), [x_shape[0], 1, x_shape[1], x_shape[2]])
        print(X.shape, y.shape)

        return X,None,None,y,None,None


class LoadDatasets:
    
    def __init__(self,data,seq_length, overlap,load_data_strategy: Strategy) -> None:
        

        # self.data = scipy.io.loadmat(src)
        # self.batch_size=batch_size
        self.data = data
        self.seq_length = seq_length
        self.overlap = overlap
        self._load_data_strategy = load_data_strategy
        
    @property
    def load_data_strategy(self) -> Strategy:
        
        return self._load_data_strategy
    
    @load_data_strategy.setter
    def strategy(self, load_data_strategy: Strategy) -> None:
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._load_data_strategy = load_data_strategy
     
        
    def load_data_logic(self,batch) -> None:
        
        self.batch_size=batch
        print(self.batch_size,self._load_data_strategy)
        self.X_train_processed,self.X_valid_processed,self.X_test_processed,self.y_train,self.y_valid,self.y_test = self._load_data_strategy.load_data(self.data,self.seq_length,self.overlap,self.batch_size)
        

    def prepare_teacher_loaders(self,number_of_teachers,batch_size):
            
            teacher_loaders = []
            self.load_data_logic(batch_size)
            X_train, y_train = unison_shuffled_copies(self.X_train_processed,self.y_train)
            data_size = len(X_train) // number_of_teachers
            
            for i in range(number_of_teachers):
                indices = list(range(i*data_size, (i+1)*data_size))
                subset_train_data = X_train[indices]
                subset_train_labels = y_train[indices]
                ds_train_obj = TSDataset(subset_train_data,subset_train_labels)
                train_data_loader = T.utils.data.DataLoader(ds_train_obj,batch_size=batch_size, shuffle=True)
                teacher_loaders.append(train_data_loader)
            return teacher_loaders
    
    def prepare_train_data_loader(self,batch_size):
        
        self.load_data_logic(batch_size)
        print("Training Shapes",self.X_train_processed.shape,self.y_train.shape)
        ds_train_obj = TSDataset(self.X_train_processed,self.y_train)
        train_data_loader = T.utils.data.DataLoader(ds_train_obj,batch_size=batch_size, shuffle=True)
        return train_data_loader
    
    def prepare_valid_data_loader(self,batch_size):
        # self.load_data()
        # self.load_data_logic(batch_size)
        print("Valid Shapes",self.X_valid_processed.shape,self.y_valid.shape)

        ds_valid_obj = TSDataset(self.X_valid_processed,self.y_valid)    
        valid_data_loader = T.utils.data.DataLoader(ds_valid_obj,batch_size=batch_size, shuffle=True)
        return valid_data_loader
    
    def prepare_test_data_loader(self,batch_size):
        # self.load_data_logic(batch_size)
        print("Test Shapes",self.X_test_processed.shape,self.y_test.shape)
        ds_test_obj = TSDataset(self.X_test_processed,self.y_test)
        test_data_loader = T.utils.data.DataLoader(ds_test_obj,batch_size=batch_size, shuffle=True)
        return test_data_loader
    
    def prepare_ensembles_test_loader(self,batch_size):
        self.load_data_logic(batch_size)
        ensemble_test_data_loaders = []
        for seq_data in self.X_train_processed:
            ds_ensemble_test_obj = TSDataset(seq_data,self.y_train)
            ensemble_test_data_loaders.append(T.utils.data.DataLoader(ds_ensemble_test_obj,batch_size=batch_size, shuffle=False))

        return ensemble_test_data_loaders