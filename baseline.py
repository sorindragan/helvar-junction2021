import pandas as pd

import plotly.express as px

import pickle
import itertools

from tqdm import tqdm

from utils import *

def agg(x):
    return set([i for i in x])


construct_dict = True

site_number = 1
seconds = 1

df_events = pd.read_pickle(f'./data/site_{site_number}/site_{site_number}.pkl', compression='gzip')
df_coords = pd.read_json(f'./data/site_{site_number}/site_{site_number}.json')
numdevs = df_coords.shape[0]

if construct_dict:
    string_lambda = lambda x: str_to_seconds(x, True)
    df_events['timestamp'] = df_events['timestamp'].apply(string_lambda)


    df_events["seconds"] = df_events['timestamp'].apply(lambda x : x.floor(f"{seconds}S"))
    df_events["deviceid"] = df_events.groupby("seconds")["deviceid"].apply(agg)
    # df_events.groupby("timestamp")['deviceid']

    with open(f"./data/site_{site_number}/bins_{seconds}.pkl", 'wb') as f:
        pickle.dump(df_events, f)

with open(f"./data/site_{site_number}/bins_{seconds}.pkl", 'rb') as f:
    df_events = pickle.load(f)

corr = np.zeros([numdevs, numdevs])

for bin in tqdm(df_events['deviceid']):
    if type(bin) != type(set):
        continue

    for d1, d2 in itertools.combinations(set(bin), r=2):
        corr[d1][d2] += 1
        corr[d2][d1] += 1

fig = px.imshow(corr)
fig.show()



