import numpy
import pandas
import random

weather_data = pandas.read_csv('./beijing_17_18_meo.csv')
weather_data.describe()
weather_data.isnull().any()
weather_data.drop_duplicates()
weather_data = weather_data.fillna(999999)

print(weather_data.loc[weather_data['wind_speed'] > 15, 'wind_speed'].value_counts())
def clean_direction(d):
    if d > 360:
        return random.randint(0, 360)
    else:
        return d

weather_data['wind_direction'] = weather_data.wind_direction.apply(clean_direction)

def clean_speed(s):
    if s > 16:
        return random.randint(0, 16)
    else:
        return s

weather_data['wind_speed'] = weather_data.wind_speed.apply(clean_speed)

def clean_temperature(t):
    if not (-20 <= t <= 40):
        return random.randint(-20, 40)
    else:
        return t

weather_data['temperature'] = weather_data.temperature.apply(clean_temperature)

def clean_pressure(p):
    if not (990 <= p <= 1040):
        return random.randint(990, 1040)
    else:
        return p

weather_data['pressure'] = weather_data.pressure.apply(clean_pressure)

def clean_humidity(h):
    if not (0 <= h <= 100):
        return random.randint(0, 100)
    else:
        return h

weather_data['humidity'] = weather_data.humidity.apply(clean_humidity)

weather_data.describe()

weather_data.to_csv('./cleaned_meo.csv', index=False)