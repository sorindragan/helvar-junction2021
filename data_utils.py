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

class SequenceDataset(Dataset):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.device_dict = {}
        self.site_data_dict = {}

        for site_number in [1,2,3,4,5]:
            print(f"Site number: " + str(site_number))
            data_list = []



            float_path = f'./data/site_{site_number}/site_{site_number}_float.pkl'
   
            #if not os.path.exists(float_path):
            df_events = pd.read_pickle(f'./data/site_{site_number}/site_{site_number}.pkl', compression='gzip')
            string_lambda = lambda x: str_to_seconds(x,True)
            df_events['timestamp'] = df_events['timestamp'].apply(string_lambda)
            #     with open(float_path,"wb") as f:
            #         pickle.dump(df_events,f)
            # else:
            #     with open(float_path,"rb") as f:
            #         df_events = pickle.load(f)
            def agg(x):
                    return set([i for i in x])

           # a = df_events.groupby[]['deviceid'].apply(agg)
            df_events["seconds"] = df_events['timestamp'].apply(lambda x : x.floor("1S"))
            df_events.groupby("seconds")['deviceid'].apply(agg)
            df_events.groupby("timestamp")['device_id']
            

            with open("data/data_list.pkl","wb") as f:
                pickle.dump(data_list,f)


            



        with open("data/data_list.pkl","wb") as f:
            pickle.dump(self.site_data_dict,f)


                    



            




if __name__ == "__main__":
    config =  yaml_loader("configs/config.yaml")
    dataset = SequenceDataset(config)

