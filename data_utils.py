from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
import yaml
import datetime 
from datetime import timezone
from dateutil import parser
from utils import yaml_loader, str_to_seconds

class SequenceDataset(Dataset):
    def __init__(self,config):
        super().__init__()
        self.config = config
        

        for site_number in [1,2,3,4,5]:
            # with open(f'./data/site_{site_number}/site_{site_number}_floats.pkl',"rb") as f:
            #     df_events = pickle.load(f)
            df_events = pd.read_pickle(f'./data/site_{site_number}/site_{site_number}.pkl', compression='gzip')
            

            df_events['timestamp'] = df_events['timestamp'].apply(lambda x : parser.parse(x, tzinfos=[timezone.utc]).utcnow().timestamp())
            # df_devices = pd.read_json(f'./data/site_{site_number}/site_{site_number}.json')
            total_time = df_events['timestamp'].max() - df_events['timestamp'].min()
            print(total_time)
            # n_bins = total_time//(config['total_sequence_time']* 3600)
            # binned = pd.cut(df_events['timestamp'],bins= n_bins)
            pass




if __name__ == "__main__":
    config =  yaml_loader("configs/config.yaml")
    dataset = SequenceDataset(config)

