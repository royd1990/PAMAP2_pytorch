from src.periodic_activations import SineActivation, CosineActivation
from torch import ne, nn
import torch

class Model(nn.Module):
    def __init__(self, network_config_params):
        super(Model, self).__init__()

        self.activation = network_config_params["temporal_activation"]
        self.hiddem_dim_time = network_config_params["hidden_dim_time"]
        self.n_layers = network_config_params["num_layers"]
        self.n_hidden = network_config_params["num_hidden"]
        self.n_classes = network_config_params["n_classes"]
        self.drop_prob = network_config_params["keep_prob"]
        self.n_channels = network_config_params["n_channels"]
        self._max_norm_val = network_config_params["max_norm"]

        if self.activation == "sin":
            self.l1 = SineActivation(10, self.hiddem_dim_time, self.n_channels)
        elif self.activation == "cos":
            self.l1 = CosineActivation(10, self.hiddem_dim_time)
        
        if self.n_layers > 1:
            self.lstm  = nn.LSTM(self.n_channels, self.n_hidden, self.n_layers, batch_first=True, dropout=self.drop_prob)
        else:
            self.lstm  = nn.LSTM(self.n_channels, self.n_hidden, batch_first=True)
    
        self.fc = nn.Linear(self.n_hidden, self.n_classes)
        self.dropout = nn.Dropout(self.drop_prob)
        # self.fc1 = nn.Linear(self.hiddem_dim_time, 2)
    
    def init_hidden(self, batch_size):
        ''' Initialize hidden state'''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # if (train_on_gpu):
        if (torch.cuda.is_available() ):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden
    
    def max_norm(self,w):
        with torch.no_grad():
            norm = w.norm(2, dim=0, keepdim=True).clamp(min=self._max_norm_val / 2)
            desired = torch.clamp(norm, max=self._max_norm_val)
            w *= (desired / norm)
        
    def forward(self, x):
            # x = x.permute(0,2,1)#x.view(-1, x.shape[2], x.shape[1])
            # x = self.l1(x)
            # x = x.permute(0,2,1)#x.view(-1, x.shape[2], x.shape[1])
            x, _ = self.lstm(x)
            x = self.dropout(x)
            self.max_norm(self.fc.weight)
            out = self.fc(x)
            x_op = out[:,-1,:]
            x_op = torch.softmax(x_op,dim=1)
            return x_op
