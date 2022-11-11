from src.periodic_activations import SineActivation, CosineActivation
from torch import ne, nn
import torch

class Model(nn.Module):
    def __init__(self, network_config_params):
        super(Model, self).__init__()

        self.n_layers = network_config_params["num_layers"]
        self.n_hidden = network_config_params["num_hidden"]
        self.n_classes = network_config_params["n_classes"]
        self.drop_prob = network_config_params["keep_prob"]
        self.n_channels = network_config_params["n_channels"]
        self._max_norm_val = network_config_params["max_norm"]
        
        if self.n_layers > 1:
            self.lstm  = nn.LSTM(self.n_channels, self.n_hidden, self.n_layers, dropout=self.drop_prob)
        else:
            self.lstm  = nn.LSTM(self.n_channels, self.n_hidden, dropout=self.drop_prob, bidirectional=True)
        self.fc = nn.Linear(self.n_hidden*2, self.n_classes)
        self.dropout = nn.Dropout(self.drop_prob)
    
    def init_hidden(self, batch_size):
        ''' Initialize hidden state'''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # if (train_on_gpu):
        if (torch.cuda.is_available() ):
            hidden = (weight.new(self.n_layers*2, batch_size, self.n_hidden).zero_().cuda(),
                weight.new(self.n_layers*2, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers*2, batch_size, self.n_hidden).zero_(),
                weight.new(self.n_layers*2, batch_size, self.n_hidden).zero_())

        return hidden
    
    def max_norm(self,w):
        with torch.no_grad():
            norm = w.norm(2, dim=0, keepdim=True).clamp(min=self._max_norm_val / 2)
            desired = torch.clamp(norm, max=self._max_norm_val)
            w *= (desired / norm)
        
    def forward(self, x, hidden, batch_size):
        
        x = x.permute(1, 0, 2)
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)    
        # x = x.contiguous().view(-1, self.n_hidden)
        self.max_norm(self.fc.weight)
        out = self.fc(x)
        x_op = out[-1,:,:]
        return x_op, hidden