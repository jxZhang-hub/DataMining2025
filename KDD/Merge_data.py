import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

weather = pd.read_csv('./cleaned_data/cleaned_meo.csv')
air_quality = pd.read_csv('./cleaned_data/cleaned_aq.csv')

air_quality = air_quality.rename(columns = {'stationId': 'station_id'})
weather['station_id'] = weather['station_id'].map(lambda x : str(x)[:-4])
air_quality['station_id'] = air_quality['station_id'].map(lambda x : str(x)[:-3])
weather_set = set(weather['station_id'].value_counts().to_dict().keys())
air_quality_set = set(air_quality['station_id'].value_counts().to_dict().keys())

stations = weather_set & air_quality_set

def merge_dataframe(weather, air_quality, station):
    w_s = weather[weather['station_id'] == station]
    aq_s = air_quality[air_quality['station_id'] == station].drop(['station_id'], axis=1)
    station_data = pd.merge(w_s, aq_s, on='utc_time')
    station_data.to_csv('./' + station + '.csv', index=False)
    print(station, len(station_data))

for station in stations:
    merge_dataframe(weather, air_quality, station)