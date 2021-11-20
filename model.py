import torch
from torch import nn
import torch.optim as optim


class SequenceLearner(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dim):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
    
        self.encoder = Encoder(input_size)

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.output_dim)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        x = self.encoder(x).reshape(10,1,self.input_size)
        out, _ = self.gru(x, h0.detach())
    
        out = out[:, -1, :]
        
        return self.fc(out)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.mlp = nn.Linear(2, latent_dim)
        self.transformer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=10)

    def forward(self, x):
        output, _ = self.transformer(self.mlp(x)).max(axis=1)
        return output

if __name__ == "__main__":
    devices = 5
    coordinates = torch.range(1,100).reshape(-1,devices,2)

    latent_dim = 30

    seq_learner = SequenceLearner(latent_dim, 2, 2, output_dim=2)

    optimizer = optim.Adam(seq_learner.parameters(),
                           lr=float(0.004))

    epochs = range(10)
    for e in epochs:
        seq_learner.zero_grad()
        yhat = seq_learner(coordinates)

        print()