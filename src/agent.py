from torch import nn
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# from sklearn.metrics import f1_score
import random
import sklearn
import os
import utils.helper
from utils.helper import get_device_id
from utils.metrics import AverageMeter, AverageMeterList, calc_accuracy,f1_score
from utils.datasets import *
from utils.losses import *
from src.network import * 
from tqdm import tqdm


train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    device_id = get_device_id(torch.cuda.is_available())
device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")

def init_weights(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            param.data = param.data
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class agent:
    
    def __init__(self,train_config_params, network_config_params, model_id=None, verbose=True, model_dir=None):
        
        #Training params
        self.epochs = train_config_params["num_epochs"]
        self.reg_coef = train_config_params["reg_coeff"]
        self.lr =  train_config_params["learning_rate"]
        self.model_id = model_id
        self.model_dir = model_dir
        
        self.loss = nn.CrossEntropyLoss()#CrossEntropyLoss2d()
        self.set_device()
        
        self.model = Model(network_config_params)
        self.model.apply(init_weights)
        print(count_parameters(self.model))
        # self.model = CNN()
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.lr,weight_decay=self.reg_coef)
        
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_acc = 0
        self.best_valid_f1 = 0
        
        self.writer = SummaryWriter(log_dir="log_dir/experiment"+"/", comment='moe_dte')
        
        # for name, param in self.model.named_parameters():
        #     if 'bias' in name:
        #         nn.init.normal_(param)
        #     elif 'weight' in name:
        #         nn.init.normal_(param)

    def set_device(self):
        device_id = utils.helper.get_device_id(torch.cuda.is_available())
        self.device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")
        
    
    def save_model(self, id=None):
        if id is None:
            file_name = "model"+".pt"# if id is None else f"dte_moe.id.{id}.pt"
        else:
            file_name = f"teacher_{id}"+"/model.pt"
        path = os.path.join(self.model_dir, file_name)
        # self.logger.log(f"Saving the current state of the model to {path}")
        print("Saving model in: ",path)
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        # self.logger.log(f"Loading model from {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        
    
    def train(self,train_data_loader, test_data_loader):
        """
        Main training function, with per-epoch model saving
        """
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.model.train()
        for epoch in tqdm(range(self.current_epoch, self.epochs)):
            h = self.model.init_hidden(self.train_data_loader.batch_size)
            self.current_epoch = epoch
            self.train_one_epoch(h)

            valid_acc,valid_f1 =  self.validate()
            self.writer.add_scalar("validation_acc/epoch", valid_acc, self.current_epoch)
            is_best = valid_f1 > self.best_valid_f1
            if is_best:
                print("Best Accuracy so far",valid_acc," in epoch",epoch)
                print("Best F1 so far",valid_f1," in epoch",epoch)
                self.best_valid_acc = valid_acc
                self.best_valid_f1 = valid_f1
                self.save_model()
                
                
        self.writer.flush()
        self.writer.close()
        return self.best_valid_acc
        
    
    def train_one_epoch(self,h):
        
        epoch_loss = AverageMeter()
        epoch_acc = AverageMeter()
        epoch_f1 = AverageMeter()
        current_batch = 0
        
        for (batch_idx, batch) in enumerate(self.train_data_loader):
            
            X = batch['features']
            y = batch['labels'].flatten()

            #Avoiding unequal batch sizes
            if(X.shape[0]==self.train_data_loader.batch_size):
                h = tuple([each.data for each in h])
                self.optimizer.zero_grad()
                pred,h = self.model(X,h,self.train_data_loader.batch_size)
                # l1 = 0
                # for p in self.model.parameters():
                #     l1 = l1 + p.abs().sum()
                cur_loss = self.loss(pred, y)
                
                if np.isnan(float(cur_loss.item())):
                    raise ValueError('Loss is nan during training...')
                
                acc = calc_accuracy(pred.data,y.data)
                f1_macro = f1_score(pred.data,y.data)
                epoch_loss.update(cur_loss.item())
                epoch_acc.update(acc,y.size(0))
                epoch_f1.update(f1_macro,y.size(0))
                self.current_iteration += 1
                current_batch += 1

                cur_loss.backward()
                self.optimizer.step()

                # if random.random() < 0.5:
                h = self.model.init_hidden(self.train_data_loader.batch_size)

        # self.writer.add_graph(self.model,X)
        self.writer.add_scalar("training_acc/epoch", epoch_acc.value, self.current_epoch)
        self.writer.add_scalar("training_loss/epoch", epoch_loss.value, self.current_epoch)
        # self.writer.close()
        # print(epoch_loss.avg)
        # print("Epoch Accuracy", epoch_acc.avg)

        print("Epoch F1 score", epoch_f1.avg)
        print("Epoch Loss", epoch_loss.avg)
        
    def validate(self):
        self.model.eval()
        valid_loss_epoch = AverageMeter()
        valid_acc_epoch = AverageMeter()
        valid_epoch_f1 = AverageMeter()
        count = 0
        
        for (batch_idx, batch) in enumerate(self.test_data_loader):
            test_h = self.model.init_hidden(self.test_data_loader.batch_size)
            X = batch['features']#.permute(1,0,2)
            y = batch['labels'].flatten()
            if X.shape[0]==self.test_data_loader.batch_size:
                test_h = tuple([each.data for each in test_h])
                pred,test_h = self.model(X, test_h, 5000)
                # print(torch.unique(torch.max(pred,dim=1)[1],return_counts=True))
                cur_loss = self.loss(pred, y)
                acc = calc_accuracy(pred.data,y.data)
                f1_macro = f1_score(pred.data,y.data)
                valid_loss_epoch.update(cur_loss.item())
                valid_acc_epoch.update(acc,X.size(0))
                valid_epoch_f1.update(f1_macro,X.size(0))
                count+=1
        print("Valid Loss",valid_loss_epoch.avg)
        # print("Valid Accuracy",valid_acc_epoch.avg)
        print("Valid F1",valid_epoch_f1.avg)
        self.model.train()
        return valid_acc_epoch.avg,valid_epoch_f1.avg
    
    
    def predict(self,test_data):
        self.model.eval()
        prediction = self.model(test_data)
        return prediction