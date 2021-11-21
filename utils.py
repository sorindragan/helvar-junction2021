import yaml
from datetime import datetime
import numpy as np
import pickle
import os

def yaml_loader(path):
    config_dict = {}
    with open(path) as f:
        config = yaml.load(f, yaml.FullLoader)
        for key,val in config.items():
            config_dict[key] = val['value']
    return config_dict

def str_to_seconds(strg,return_date = False):
    try:
        date_time_obj = datetime.strptime(strg, "%Y-%m-%dT%H:%M:%S.%fZ")
    except:
        date_time_obj = datetime.strptime(strg, "%Y-%m-%dT%H:%M:%SZ")
    
      
    if not return_date:
        date_time_obj = date_time_obj.timestamp()

        

    return date_time_obj

def pickle_load(path):
    with open(path,"rb") as f:
        return pickle.load(f)
def pickle_dump(path,thing):
    with open(path,"wb") as f:
        pickle.dump(thing,f)


def find_position_by_polygon(points, distances):
    # points: Nx2 distances: Nx1
    if distances.sum() == 0:
        weights = np.expand_dims(np.array([1/len(distances) for _ in range(len(distances))]),1)
    else:
        weights = 1 - np.expand_dims(distances,1)/distances.sum()
        weights /= weights.sum()
    middle = (points*weights).sum(axis=0)
    return middle

