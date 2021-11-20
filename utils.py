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

def find_position_by_triangle(p1, d1, p2, d2, p3, d3):
    s = d1 + d2 + d3
    # can be vectorized but whatever
    p1_w = 1 - (d1 / s)
    p2_w = 1 - (d2 / s)
    p3_w = 1 - (d3 / s)

    new_point = (p1_w * p1[0] + p2_w * p2[0] + p3_w * p3[0], p1_w * p1[1] + p2_w * p2[1] + p3_w * p3[1])
    return new_point