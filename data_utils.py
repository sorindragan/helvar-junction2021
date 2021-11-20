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

        Data_point = namedtuple("data_point",["boolean_array","site_number"])
        for site_number in [1,2,3,4,5]:
            print(f"Site number: " + str(site_number))
            data_list = []



            float_path = f'./data/site_{site_number}/site_{site_number}_float.pkl'
   
            if not os.path.exists(float_path):
                df_events = pd.read_pickle(f'./data/site_{site_number}/site_{site_number}.pkl', compression='gzip')

                df_events['timestamp'] = df_events['timestamp'].apply(str_to_seconds)
                with open(float_path,"wb") as f:
                    pickle.dump(df_events,f)
            else:
                with open(float_path,"rb") as f:
                    df_events = pickle.load(f)

            
            df_devices = pd.read_json(f'./data/site_{site_number}/site_{site_number}.json')
            n_devices = df_devices.shape[0]
            self.device_dict[site_number] = df_devices
            total_time = df_events['timestamp'].max() - df_events['timestamp'].min()

            df_events = df_events.sort_values(by=['timestamp']).reset_index()
            bucket_index = 0
            
            start = df_events["timestamp"][0]
            current_device_set = set()
            
            for index,row in tqdm(df_events.iterrows()):
                
                device_id = row['deviceid']
                time = row['timestamp']
                new_bucket_index = int((time - start)//config['bin_width'])
                if new_bucket_index!=bucket_index:
                    boolean_array = np.zeros(n_devices)
                    bucket_dif = new_bucket_index - bucket_index 
                    if bucket_dif>1:
                        data_point = (boolean_array,site_number)
                        fill_in_zeros = [data_point]*(bucket_dif-1)
                        data_list.extend(fill_in_zeros)

                    boolean_array[np.array(list(current_device_set))] = 1
                    current_device_set = set()
                    bucket_index = new_bucket_index
                    data_point = (boolean_array,site_number)
                    data_list.append(data_point)
                current_device_set.add(int(device_id))

            self.site_data_dict[site_number] = data_list

            with open("data/data_list.pkl","wb") as f:
                pickle.dump(data_list,f)


            



        with open("data/data_list.pkl","wb") as f:
            pickle.dump(self.site_data_dict,f)


                    



            




if __name__ == "__main__":
    config =  yaml_loader("configs/config.yaml")
    dataset = SequenceDataset(config)

