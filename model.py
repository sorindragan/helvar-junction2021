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
    
        self.transformer_encdoder = Encoder(config)
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.output_dim)
        self.blow_up_mlp = nn.Linear(2,self.input_size*self.num_layers)



    def forward(self, sequence, coordinates):
        
        h0 = self.blow_up_mlp(coordinates)
        h0 = self.transformer_encdoder(h0)
        h0 = h0.max(dim=1).reshape(self.config.num_layers, self.config['batch_size'], self.config.hidden_size)

        x = self.transformer_encdoder(sequence).reshape(self.config.batch_size,-1,self.input_size)
        out, _ = self.gru(x, h0)
    
        out = out[:, -1, :]
        
        return self.fc(out)
    



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.mlp = nn.Linear(config.output_dim, config.latent_dim)
        self.transformer = nn.TransformerEncoderLayer(d_model=config.latent_dim, nhead=config.nheads)

    def forward(self, x):
        output, _ = self.transformer(self.mlp(x)).max(axis=1)
        return output
