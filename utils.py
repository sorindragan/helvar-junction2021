import yaml
from datetime import datetime

def yaml_loader(path):
    config_dict = {}
    with open(path) as f:
        config = yaml.load(f, yaml.FullLoader)
        for key,val in config.items():
            config_dict[key] = val['value']
    return config_dict

def str_to_seconds(strg):
    date_time_obj = datetime.strptime(strg, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
    return date_time_obj

