import torch
import torch.nn.functional as F
import torch.optim as optim
from model import SequenceLearner
import os
import pickle
import numpy as np

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['WANDB_MODE'] = 'dryrun'
run = wandb.init(project="junction", entity="mvalente", config=r'configs/config.yaml')
config = wandb.config

seq_learner = SequenceLearner(config).to(device)

optimizer = optim.Adam(seq_learner.parameters(),
                       lr=config.lr)
loss_fn = torch.nn.MSELoss()

with open("data/data_list.pkl","rb") as f:
    data = pickle.load(f)
    data = np.array(data)
with open("data/coordinates.pkl","rb") as f:
    coordinates = pickle.load(f)
    coordinates = torch.tensor(coordinates).squeeze()

seq_learner.set_coordinates(coordinates)
# Dummy Data
complete_data = []
for i in range(80,15080):
    if isinstance(data[i][0],np.ndarray):
        complete_data.append(data[i][0])

complete_data = np.array(complete_data)
coordinates = torch.randint(10,(16,5,3)).repeat(16,1,1,1) 

for epoch in range(1, config.epochs+1):
    for batch in coordinates:
        optimizer.zero_grad(())

        yhat = seq_learner(batch.float().to(device))
        y = batch[:,0,:2].shape
        
        loss = loss_fn(yhat,y)
        loss.backward()
        optimizer.step()

        run.log({"loss": loss.item()})
