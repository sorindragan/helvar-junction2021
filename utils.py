import yaml
from datetime import datetime
import numpy as np

def yaml_loader(path):
    config_dict = {}
    with open(path) as f:
        config = yaml.load(f, yaml.FullLoader)
        for key,val in config.items():
            config_dict[key] = val['value']
    return config_dict

def str_to_seconds(strg):
    try:
        date_time_obj = datetime.strptime(strg, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
    except:
        date_time_obj = datetime.strptime(strg, "%Y-%m-%dT%H:%M:%SZ").timestamp()
    
      


    return date_time_obj





def find_position_by_polygon(points,distances):
    # points: Nx2 distances: Nx1
    weights = 1 - np.expand_dims(distances,1)/distances.sum()
    weights /= weights.sum()
    middle = (points*weights).sum(axis=0)
    return middle

