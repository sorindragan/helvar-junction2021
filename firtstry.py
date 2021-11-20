import pandas as pd
import numpy as np
import base64
import imageio as iio
from plotting import Plotting
import plotly.graph_objects as go
from random import shuffle
from pprint import pprint
from datetime import datetime
from datetime import timezone
from dateutil import parser
import pickle
from tqdm import tqdm
from copy import deepcopy


site = 'site_1'
df_solution = pd.read_json(f'./data/{site}/{site}.json')
df_events = pd.read_pickle(f'./data/{site}/{site}.pkl', compression='gzip')

print(df_events.shape)

df_events = df_events.sort_values(by=['timestamp'])

pct = 0.5

ids = list(set(df_events['deviceid']))
shuffle(ids)
known_ids = ids[:int(pct*len(ids))]
unknown_ids = [i for i in ids if i not in known_ids]

# comment after pkl generation
# tuple_dict = {}
# last_t, last_d = -1, -1

# for i, row in tqdm(df_events.iterrows()):
#     t = row['timestamp']
#     d = row['deviceid']
    
#     if last_d == -1 or last_d == d:
#         last_d = d
#         last_t = t
#         continue
    
#     if (last_d, d) not in tuple_dict: 
#         tuple_dict[(last_d, d)] = []
    
#     toint = lambda x: parser.parse(x, tzinfos=[timezone.utc]).utcnow().timestamp()
#     deltat = abs(toint(last_t) - toint(t))

#     if (d, last_d) in tuple_dict:
#         tuple_dict[(d, last_d)].append(deltat)
#     else:
#         tuple_dict[(last_d, d)].append(deltat)
    
#     last_t = t
#     last_d = d


# with open(f"./data/{site}/tuplepairs.pkl", 'wb') as f:
#     pickle.dump(tuple_dict, f)

# up to here

with open(f"./data/{site}/tuplepairs.pkl", 'rb') as f:
    tuple_dict = pickle.load(f)


neighbours_dict = {}

for k, v in tuple_dict.items():
    avg = sum(v) / (len(v) + 1)
    # fire at the same time - prodbably different locations
    if avg == 0:
        continue
    neighbours_dict[k] = avg

neighbours_dict = sorted(neighbours_dict.items(), key=lambda x: x[0][0])

closest_neighbours = {k:[(-1, 999), (-1, 999), (-1, 999)] for k in ids}
for k, v in neighbours_dict:
    if v < closest_neighbours[k[0]][0][1]:
        closest_neighbours[k[0]][0] = (k[1], v)
        closest_neighbours[k[0]].sort(key=lambda t: t[1], reverse=True)

# pprint(closest_neighbours)

available_for_computation = []

for k, v in closest_neighbours.items():
    if sum([1 if i in known_ids else 0 for i in [p[0] for p in v]]) == 3:
        available_for_computation.append(k)

print(available_for_computation)



    





# first_timestep =  parser.parse(df_events['timestamp'][], tzinfos=[timezone.utc]).utcnow().timestamp()






# print(len(unknown_ids))
# print(len(known_ids))

# # devices = {id: np.array(sorted(list(map(lambda x: parser.parse(x, tzinfos=[timezone.utc]).utcnow().timestamp(), 
# #                         df_events[df_events['deviceid']==id]['timestamp'].tolist()))))
# #                     for id in ids
# #                     }

# # with open(f"./data/{site}/timestamps.pkl", 'wb') as f:
# #     pickle.dump(devices, f)


# with open(f"./data/{site}/timestamps.pkl", 'rb') as f:
#     devices = pickle.load(f)

# maxlen = max([len(x) for x in devices.values()])

# rows = []
# known_rows = []

# # 86400 intervals

# for k, v in tqdm(sorted(devices.items(), key=lambda x: x[0])):
#     rows.append(np.append(v, np.array([np.inf]*(maxlen-len(v)))))

# # print(rows[0:3])
# alldevices = np.array(rows)
# known_devices = alldevices[known_ids, :]

# known_devices = np.expand_dims(known_devices, 1)
# differences = np.abs(known_devices - alldevices)

# pass