from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
import yaml
import datetime 
from datetime import timezone
from dateutil import parser
from utils import yaml_loader, str_to_seconds
import numpy as np
from tqdm import tqdm
from collections import namedtuple

class SequenceDataset(Dataset):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.device_dict = {}
        self.data_list = []
        Data_point = namedtuple("data_point",["sequence","site_number"])
        for site_number in [1,2,3,4,5]:
            print(f"Site number: " + str(site_number))
            with open(f'./data/site_{site_number}/site_{site_number}_floats.pkl',"rb") as f:
                df_events = pickle.load(f)
            #df_events = pd.read_pickle(f'./data/site_{site_number}/site_{site_number}.pkl', compression='gzip')
            

            #df_events['timestamp'] = df_events['timestamp'].apply(str_to_seconds)
            
            df_devices = pd.read_json(f'./data/site_{site_number}/site_{site_number}.json')
            n_devices = df_devices.shape[0]
            self.device_dict[site_number] = df_devices
            total_time = df_events['timestamp'].max() - df_events['timestamp'].min()
            n_bins_overall = int(total_time//(config['total_sequence_time']* 3600))
            n_bins_fine = int(config['total_sequence_time']* 3600//config['bin_width'])
            binned_overall = pd.cut(df_events['timestamp'],bins= n_bins_overall,labels = False)
            for overall_index in tqdm(np.unique(binned_overall)):
                row_list = []
                overall_mask = binned_overall==overall_index
                overall_df  = df_events.iloc[np.where(overall_mask)[0],:]
                binned_fine = pd.cut(overall_df['timestamp'],bins= n_bins_fine,labels = False)
                overall_df['fine_index'] = binned_fine
                for fine_index in binned_fine:
                    filtered_fine = overall_df[overall_df['fine_index'] == fine_index]
                    present_devices = filtered_fine['deviceid'].unique()
                    boolean_array = np.zeros(n_devices)
                    boolean_array[present_devices] = 1
                    row_list.append(boolean_array)
                sequence = np.array(row_list)
                data_point = Data_point(sequence,site_number)
                self.data_list.append(data_point)
        with open("data/data_list.pkl","rb") as f:
            pickle.dump(self.data_list,f)


                    



            




if __name__ == "__main__":
    config =  yaml_loader("configs/config.yaml")
    dataset = SequenceDataset(config)

