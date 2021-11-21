import itertools
import matplotlib.pyplot as plt
from copy import deepcopy
from pprint import pprint
import pickle
from random import shuffle

import pandas as pd
import plotly.express as px
from tqdm import tqdm

from utils import *


def agg(x):
    return set([i for i in x])


construct_dict = True

site_number = 2
seconds = 1

df_events = pd.read_pickle(f'./data/site_{site_number}/site_{site_number}.pkl', compression='gzip')
df_coords = pd.read_json(f'./data/site_{site_number}/site_{site_number}.json')
df_solution = deepcopy(df_coords)
numdevs = df_coords.shape[0]

if construct_dict:
    string_lambda = lambda x: str_to_seconds(x, True)
    df_events['timestamp'] = df_events['timestamp'].apply(string_lambda)


    df_events["seconds"] = df_events['timestamp'].apply(lambda x : x.floor(f"{seconds}S"))
    df_events  = df_events.groupby("seconds")["deviceid"].apply(agg)
    # df_events.groupby("timestamp")['deviceid']

    with open(f"./data/site_{site_number}/bins_{seconds}.pkl", 'wb') as f:
        pickle.dump(df_events, f)

with open(f"./data/site_{site_number}/bins_{seconds}.pkl", 'rb') as f:
    df_events = pickle.load(f)

bins = df_events.values\

corr = np.zeros([numdevs, numdevs])

for bin in tqdm(bins):
    if len(bin) < 2:
        continue

    for d1, d2 in itertools.combinations(set(bin), r=2):
        corr[d1][d2] += 1
        corr[d2][d1] += 1

fig = px.imshow(corr)
fig.show()


pct = 0.9
ids = list(range(numdevs))
shuffle(ids)

known_ids = ids[:int(pct*len(ids))]
initial_known_inds = deepcopy(known_ids)

unknown_ids = ids[int(pct*len(ids)):]

top_n = 4
polygon_corners = 3

neighbours = {i: [] for i in ids}
for idx, row in enumerate(corr):
    neighbours[idx] = row.argsort()[-top_n:][::-1]

approximated_devices = {}
available_for_computation = [-1]

while len(available_for_computation) > 0:
    available_for_computation = []
    for k, v in neighbours.items():
        if k in unknown_ids and sum([1 if i in known_ids else 0 for i in v]) >= polygon_corners:
            available_for_computation.append(k)
            points = np.array([df_coords.iloc[p2].values[1:] for p2 in v[:polygon_corners]])
            distances = np.array([corr[k,p2] for p2 in v[:polygon_corners]])

            approximated_devices[k] = find_position_by_polygon(points,distances)
            # use approximations instead of actual values for each newly added point
            df_coords.iloc[int(k), df_coords.columns == 'x'] = approximated_devices[k][0]
            df_coords.iloc[int(k), df_coords.columns == 'y'] = approximated_devices[k][1]

    unknown_ids = list(set(unknown_ids) - set(available_for_computation))
    known_ids += available_for_computation

# pprint(approximated_devices)
pprint(len(ids) - len(known_ids))

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
pass
