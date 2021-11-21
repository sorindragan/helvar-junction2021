import torch
from torch import nn
import torch.optim as optim


class SequenceLearner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_size = config.latent_dim
        self.hidden_size = config.hidden_dim
        self.num_layers = config.layer_dim
        self.output_dim = config.output_dim
    

        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.output_dim)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        x = self.encoder(x).reshape(config.batch_size,-1,self.input_size)
        out, _ = self.gru(x, h0.detach())
    
        out = out[:, -1, :]
        
        return self.fc(out)
    
    def set_coordinates(self, coordinates):
        self.encoder = Encoder(config, coordinates)


class Encoder(nn.Module):
    def __init__(self, config, coordinates):
        super(Encoder, self).__init__()
        self.register_buffer('coordinates', vertice.clone())
        self.mlp = nn.Linear(config.output_dim, config.latent_dim)
        self.transformer = nn.TransformerEncoderLayer(d_model=config.latent_dim, nhead=config.nheads)

    def forward(self, x):
        x = 
        output, _ = self.transformer(self.mlp(x)).max(axis=1)
        return output
