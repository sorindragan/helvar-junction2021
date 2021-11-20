import torch
from torch import nn
import torch.optim as optim


class SequenceLearner(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dim, config):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.config = config
    
        self.encoder = Encoder(input_size, config)

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.output_dim)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        x = self.encoder(x).reshape(config.batch_size,-1,self.input_size)
        out, _ = self.gru(x, h0.detach())
    
        out = out[:, -1, :]
        
        return self.fc(out)

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.mlp = nn.Linear(config.output_dim, config.latent_dim)
        self.transformer = nn.TransformerEncoderLayer(d_model=config.latent_dim, nhead=config.nheads10)

    def forward(self, x):
        output, _ = self.transformer(self.mlp(x)).max(axis=1)
        return output
