import pandas as pd
import numpy as np
import base64
import imageio as iio
from plotting import Plotting
import plotly.graph_objects as go
from tqdm import tqdm

site = 'site_1'
df_events = pd.read_pickle(f'./data/{site}/{site}.pkl', compression='gzip')
df_events.loc[:, 'timestamp'] = (pd.to_datetime(df_events['timestamp'], utc=True)
                                 .dt.tz_convert('Europe/Helsinki')
                                 .dt.tz_localize(None))

df_devices = pd.read_json(f'./data/{site}/{site}.json')
# df_events_day = df_events.copy()
# df_events_day.loc[:, 'timestamp'] = df_events_day['timestamp'].dt.floor('1h')
# df_events_day.loc[:, 'value'] = 1.0
# df_events_day = df_events_day.groupby('timestamp').sum()
# df_events_day = df_events_day.drop(['deviceid'], axis=1)
# df_events_day = df_events_day.reindex(pd.date_range(df_events_day.index.min(), df_events_day.index.max(), freq='1h')).fillna(0)
# 

frac = 100000
window_size = int(df_events.shape[0]/frac)

values = []
for f in tqdm(range(1,frac)):
    df_events_s = df_events.copy()[window_size*f - window_size:window_size*f]
    df_events_s.timestamp = df_events_s.timestamp.dt.floor('500ms')
    df_events_s.loc[:, 'b'] = 1
    df_events_s = df_events_s.groupby(['deviceid', 'timestamp']).sum()
    df_events_s = df_events_s.pivot_table(index='timestamp', columns='deviceid', values='b')
    df_events_s = df_events_s.reindex(pd.date_range(df_events_s.index[0], df_events_s.index[-1], freq='500ms', closed='left')).fillna(0)
    if df_events_s.values.sum()
    values.append(df_events_s)
# df_events_s = df_events_s.reindex(pd.date_range(df_events_s.index.min().floor('1D'), df_events_s.index.max().ceil('1D'), freq='1min', closed='left')).fillna(0)

