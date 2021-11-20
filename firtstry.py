import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from random import shuffle
from pprint import pprint
from tqdm import tqdm
from copy import deepcopy

from utils import str_to_seconds, find_position_by_polygon


site = 'site_1'
dict_generation = False
df_coords = pd.read_json(f'./data/{site}/{site}.json')
df_events = pd.read_pickle(f'./data/{site}/{site}.pkl', compression='gzip')

df_solution = deepcopy(df_coords)

print(df_events.shape)

df_events = df_events.sort_values(by=['timestamp'])
if dict_generation:
    df_events['timestamp'] = df_events['timestamp'].apply(lambda x : str_to_seconds(x))


# pctage of known ids
pct = 0.9
ids = list(set(df_events['deviceid']))
shuffle(ids)

known_ids = ids[:int(pct*len(ids))]
initial_known_inds = deepcopy(known_ids)
unknown_ids = [i for i in ids if i not in known_ids]

# tuple_dict pkl generation
if dict_generation:
    tuple_dict = {}
    last_t, last_d = -1, -1

    for i, row in tqdm(df_events.iterrows()):
        t = row['timestamp']
        d = row['deviceid']
        
        if last_d == -1 or last_d == d:
            last_d = d
            last_t = t
            continue
        
        if (last_d, d) not in tuple_dict: 
            tuple_dict[(last_d, d)] = []
        
        deltat = abs(last_t - t)

        if (d, last_d) in tuple_dict:
            tuple_dict[(d, last_d)].append(deltat)
        else:
            tuple_dict[(last_d, d)].append(deltat)
        
        last_t = t
        last_d = d


    with open(f"./data/{site}/tuplepairs.pkl", 'wb') as f:
        pickle.dump(tuple_dict, f)

# tuple_dict load
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

def_neighbour = -1
max_distance = 9999
polygon_edges = 3
neighbours_number = polygon_edges* 2
closest_neighbours = {k:[(def_neighbour, max_distance)] * neighbours_number for k in ids}
for k, v in neighbours_dict:
    if v < closest_neighbours[k[0]][0][1]:
        closest_neighbours[k[0]][0] = (k[1], v)
        closest_neighbours[k[0]].sort(key=lambda t: t[1], reverse=True)

# pprint(closest_neighbours)

# iteratively compute coordinates for the points with 3 known neighbours
approximated_devices = {}
# stupid init
available_for_computation = [-1]
while len(available_for_computation) > 0:
    available_for_computation = []
    for k, v in closest_neighbours.items():
        if k in unknown_ids and sum([1 if i in known_ids else 0 for i in [p[0] for p in v]]) >= polygon_edges:
            available_for_computation.append(k)
            corner_list = [(
                            (float(df_coords.iloc[int(p[0])]['x']), 
                             float(df_coords.iloc[int(p[0])]['y'])),
                             p[1]
                            ) for p in v
                            if p[0] in known_ids
                        ][:polygon_edges]
          
            points = np.array([x[0] for x in corner_list])
            distances = np.array([x[1] for x in corner_list])
            approximated_devices[k] = find_position_by_polygon(points,distances)
            # use approximations instead of actual values for each newly added point
            df_coords.iloc[int(k), df_coords.columns == 'x'] = approximated_devices[k][0]
            df_coords.iloc[int(k), df_coords.columns == 'y'] = approximated_devices[k][1]
            
    unknown_ids = list(set(unknown_ids) - set(available_for_computation))
    known_ids += available_for_computation
    print(available_for_computation)

# pprint(approximated_devices)
print(len(ids) - len(known_ids))

# check against real solution

# approximated points
x = np.array([v[0] for v in approximated_devices.values()])
y = np.array([v[1] for v in approximated_devices.values()]) 
plt.scatter(x, y, c='b')

# real solution
x = np.array([float(df_solution.iloc[int(k)]['x']) for k in approximated_devices.keys()])
y = np.array([float(df_solution.iloc[int(k)]['y']) for k in approximated_devices.keys()])
plt.scatter(x, y, c='r')

# initial known points
x = np.array([float(df_solution.iloc[int(k)]['x']) for k in initial_known_inds])
y = np.array([float(df_solution.iloc[int(k)]['y']) for k in initial_known_inds])
plt.scatter(x, y, c='k')

plt.show()
