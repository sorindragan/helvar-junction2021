import torch
import torch.optim as optim
from model import SequenceLearner

latent_dim = 30
input_dim = latent_dim
hidden_dim = 2
layer_dim = 2
output_dim = 2 #for each coordinate

seq_learner = SequenceLearner(input_dim, hidden_dim, layer_dim, output_dim)

optimizer = optim.Adam(seq_learner.parameters(),
                       lr=float(0.004))
devices = 5
coordinates = torch.randint(10,(16,5,3)).repeat(16,1,1,1) 

epochs = range(10)
for e in epochs:
    for batch in coordinates:
        seq_learner.zero_grad()
        yhat = seq_learner(batch.float())
        
