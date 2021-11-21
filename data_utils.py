import enum
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
import yaml
from datetime import datetime 
from datetime import timezone
from dateutil import parser
from utils import yaml_loader, str_to_seconds
import numpy as np
from tqdm import tqdm
from collections import namedtuple
import os
from utils import *
import random

class SequenceDataset(Dataset):
    def __init__(self,config,preprocess=True):
        super().__init__()
        self.config = config
        

        def agg(x):
            unique = x.unique()
            if len(unique)>1:
                return unique
        if preprocess:
            print(f"Preprocessing")
            for site_number in [1,2,3,4,5]:
                print(f"Site number: " + str(site_number))
                data_list = []

                
                path_data_list = f"data/site_{site_number}/data_list_{site_number}.pkl"
                path_df = f"data/site_{site_number}/timestamp_df_{site_number}.pkl"
                if not os.path.exists(path_df):
                    df_events = pd.read_pickle(f'./data/site_{site_number}/site_{site_number}.pkl', compression='gzip')
                    string_lambda = lambda x: str_to_seconds(x,True)
                    df_events['timestamp'] = df_events['timestamp'].apply(string_lambda)
                    pickle_dump(path_df,df_events)
                else:
                    
                    df_events = pickle_load(path_df)
    
                
                


                df_events["seconds"] = df_events['timestamp'].apply(lambda x : x.floor("1S"))

                
                data_list = df_events.groupby("seconds")['deviceid'].apply(agg).dropna().values.tolist()
                pickle_dump(path_data_list,data_list)

        self.data_lists = []
        self.df_coords = []
        for site_number in [1,2,3,4,5]:
            path_data_list = f"data/site_{site_number}/data_list_{site_number}.pkl"
            self.df_coords.append(df_coords = pd.read_json(f'./data/site_{site_number}/site_{site_number}.json'))
            self.data_lists.append(pickle_load(path_data_list))
        
        self.list_lengths = np.array([len(x) for x in self.data_lists])
    def __len__(self):
        return sum(self.list_lengths-self.config['sequence_length_train']) - 3
    def __getitem__(self,idx):

        ranges = []

        initial = 0
        s = self.config['sequence_length_train']
        for length in self.list_lengths:
            ranges.append((initial,initial+length-s))
            initial = initial + length + 1
        i = -1
        while True:
            i+=1
            range = ranges[i]
            if idx in range:
                inner_index = idx - range[0]
                site_number = i
            break

        data_list = self.data_lists[site_number][inner_index:inner_index+s]
        device_df = self.df_coords[site_number]
        n_devices = device_df.shape[0]
        rows = []
        device_indexes = range(0,n_devices)
        random.shuffle(device_indexes)
        train_known_range = random.randrange(config["train_known_range"][0],config['train_known_range'][1],0.05)

        to_predict = device_indexes[:int((1- train_known_range)*n_devices)]
        for x in data_list:
            row = np.zeros(n_devices)
            row[x] = 1
            rows.append(row)
        sequence = np.array(rows)
        
        coords = device_df[["x","y"]].values

        coords[to_predict,:] = -1

        return sequence , coords








            





                    



            




if __name__ == "__main__":
    config =  yaml_loader("configs/config.yaml")
    dataset = SequenceDataset(config)

