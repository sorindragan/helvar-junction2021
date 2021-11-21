import torch
import torch.nn.functional as F
import torch.optim as optim
from model import SequenceLearner
import os
import pickle
import numpy as np
from data_utils import Dataset,DataLoader,SequenceDataset
import wandb
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['WANDB_MODE'] = 'dryrun'
run = wandb.init(project="junction", entity="mvalente", config=r'configs/config.yaml')
config = wandb.config
parameters = []
seq_learner = SequenceLearner(config).to(device)

parameters += seq_learner.parameters()

optimizer = optim.Adam(parameters,
                       lr=config.lr)
loss_fn = torch.nn.MSELoss()


dataset = SequenceDataset(config,preprocess=False)
dataloader = DataLoader(dataset,batch_size=config['batch_size'])

for epoch in range(0, config.epochs):
    for index, batch in enumerate(tqdm(dataloader)):
        sequence, coordinates = batch
        sequence = sequence.to(device)
        coordinates = coordinates.to(device)
        optimizer.zero_grad(())
        yhat = seq_learner(sequence,coordinates)
        y = batch[:,0,:2].shape
        
        loss = loss_fn(yhat,y)
        loss.backward()
        optimizer.step()

        run.log({"loss": loss.item()})
