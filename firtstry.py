import pickle
import itertools
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
discard_simultaniously = False
random_points = True
df_coords = pd.read_json(f'./data/{site}/{site}.json')
df_events = pd.read_pickle(f'./data/{site}/{site}.pkl', compression='gzip')

df_solution = deepcopy(df_coords)

print(df_events.shape)

if dict_generation:
    df_events['timestamp'] = df_events['timestamp'].apply(lambda x : str_to_seconds(x))
    df_events = df_events.sort_values(by=['timestamp']).reset_index()

# constants
def_neighbour = -1
max_distance = -1
polygon_corners = 2
neighbours_number = int(polygon_corners * 1.5)
secs_between_events = 1.5
window_size = 5

# pctage of known ids
pct = 0.9
ids = sorted(list(set(df_events['deviceid'])))
if random_points:
    shuffle(ids)

known_ids = ids[:int(pct*len(ids))]
initial_known_inds = deepcopy(known_ids)
unknown_ids = [i for i in ids if i not in known_ids]

corr = np.zeros([len(ids), len(ids)])
# tuple_dict = {(d1, d2): [] for d1, d2 in itertools.product(len(ids), len(ids))}
tuple_dict = {(d1, d2): 0 for d1, d2 in itertools.product(range(len(ids)), range(len(ids))) if d1 != d2}


# tuple_dict pkl generation
if dict_generation:
    rows = list(df_events.iterrows())
    length = len(rows)
    ws = window_size
    queue = [r[1] for r in rows[:ws]]

    for i, row in tqdm(rows[ws//2+1:length-ws-1]):
        t = row['timestamp']
        d = int(row['deviceid'])

        try: 
            queue.append(rows[i+ws][1])
        except: 
            print("Index out of bounds")
            print(i+ws)
            print(length)
        queue.pop(0)

        for n in queue:
            nt = n['timestamp']
            nd = int(n['deviceid'])
            if d == nd:
                continue
            
            deltat = abs(t - nt)
            if deltat > secs_between_events:
                continue

            # tuple_dict[(d, nd)].append(deltat)
            tuple_dict[(d, nd)] += 1
            tuple_dict[(nd, d)] += 1
            corr[int(d), int(nd)] += 1
            corr[int(nd), int(d)] += 1
            # tuple_dict[(nd, d)].append(deltat)

    with open(f"./data/{site}/tuplepairs.pkl", 'wb') as f:
        pickle.dump(tuple_dict, f)

# pprint(corr)

# tuple_dict load
with open(f"./data/{site}/tuplepairs.pkl", 'rb') as f:
    tuple_dict = pickle.load(f)

# pprint(tuple_dict)
neighbours_dict = {}

for k, v in tuple_dict.items():
    # avg = sum(v) / (len(v) + 1)
    # # fire at the same time - prodbably different locations
    # if avg == 0 and discard_simultaniously:
    #     continue
    # neighbours_dict[k] = avg
    neighbours_dict[k] = v

neighbours_dict = sorted(neighbours_dict.items(), key=lambda x: x[0][0])

closest_neighbours = {k:[(def_neighbour, max_distance)] * neighbours_number for k in ids}
for k, v in neighbours_dict:
    # if v < closest_neighbours[k[0]][0][1]:
    #     closest_neighbours[k[0]][0] = (k[1], v)
    #     closest_neighbours[k[0]].sort(key=lambda t: t[1], reverse=True)
    if v > closest_neighbours[k[0]][0][1]:
        closest_neighbours[k[0]][0] = (k[1], v)
        closest_neighbours[k[0]].sort(key=lambda t: t[1])

# pprint(closest_neighbours)

# iteratively compute coordinates for the points with <polygon_corners> known neighbours
approximated_devices = {}
# stupid init
available_for_computation = [-1]
while len(available_for_computation) > 0:
    available_for_computation = []
    for k, v in closest_neighbours.items():
        if k in unknown_ids and sum([1 if i in known_ids else 0 for i in [p[0] for p in v]]) >= polygon_corners:
            available_for_computation.append(k)
            corner_list = [(
                            (float(df_coords.iloc[int(p[0])]['x']), 
                             float(df_coords.iloc[int(p[0])]['y'])),
                             p[1]
                            ) for p in v
                            if p[0] in known_ids
                        ][:polygon_corners]
          
            points = np.array([x[0] for x in corner_list])
            distances = np.array([x[1] for x in corner_list])
            approximated_devices[k] = find_position_by_polygon(points,distances)
            # use approximations instead of actual values for each newly added point
            df_coords.iloc[int(k), df_coords.columns == 'x'] = approximated_devices[k][0]
            df_coords.iloc[int(k), df_coords.columns == 'y'] = approximated_devices[k][1]
            
    unknown_ids = list(set(unknown_ids) - set(available_for_computation))
    known_ids += available_for_computation
    print(available_for_computation)

pprint(approximated_devices)
print(len(ids) - len(known_ids))

# check against real solution

# approximated points
x = np.array([v[0] for v in approximated_devices.values()])
y = np.array([v[1] for v in approximated_devices.values()]) 
plt.scatter(x, y, c='b')

# real solution
x = np.array([float(df_solution.iloc[int(k)]['x']) for k in sorted(approximated_devices.keys())])
y = np.array([float(df_solution.iloc[int(k)]['y']) for k in sorted(approximated_devices.keys())])
plt.scatter(x, y, c='r')

# TODO: code a grid search
# mean diference
points_ = np.array([c[1] for c in sorted(list(approximated_devices.items()), key=lambda x: x[0])])
points = np.array([[x, y] for x, y in zip(x, y)])
avg_diff = np.linalg.norm(points - points_, axis=1).mean()
pprint("Error")
pprint(avg_diff)
pprint(points)
pprint(points_)

# initial known points
x = np.array([float(df_solution.iloc[int(k)]['x']) for k in initial_known_inds])
y = np.array([float(df_solution.iloc[int(k)]['y']) for k in initial_known_inds])
plt.scatter(x, y, c='k')

plt.show()
