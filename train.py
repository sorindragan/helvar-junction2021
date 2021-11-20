import torch
import torch.optim as optim
from model import SequenceLearner

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run = wandb.init(project="junction", entity="mvalente", config=r'config/train.yaml')
config = wandb.config

seq_learner = SequenceLearner(config.latent_dim,
                              config.hidden_dim,
                              config.layer_dim,
                              config.output_dim,
                              config.batch_size).to(device)

optimizer = optim.Adam(seq_learner.parameters(),
                       lr=config.lr)
loss_fn = torch.nn.MSELoss()

# Dummy Data
devices = 5
coordinates = torch.randint(10,(16,5,3)).repeat(16,1,1,1) 

epochs = range(10)
for e in config.epochs:
    for batch in coordinates:
        optimizer.zero_grad(())

        yhat = seq_learner(batch.float().to(device))
        y = batch[:,0,:2].shape
        
        loss = loss_fn(yhat,y)
        loss.backward()
        optimizer.step()

        run.log({"loss": loss.item()})
