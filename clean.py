import pandas as pd
import numpy as np


# Shared interpolation logic for numeric sensor types
def interpolate_sensor(sensor_df, value_cols, sensor_type, freq='1min'):
    sensor_df = sensor_df[sensor_df['type'] == sensor_type].copy()
    sensor_df['datetime'] = pd.to_datetime(sensor_df['datetime'])
    sensor_df[value_cols] = sensor_df[value_cols].apply(pd.to_numeric, errors='coerce')

    imputed_logs = []
    cleaned_dfs = []

    for uid, group in sensor_df.groupby('uid'):
        group = group.set_index('datetime').sort_index()
        resampled = group[value_cols].resample(freq).mean()

        nan_before = resampled.isna().sum()
        resampled = resampled.interpolate(method='linear', limit_direction='both')
        resampled = resampled.ffill().bfill()
        nan_after = resampled.isna().sum()
        imputed_count = (nan_before - nan_after).clip(lower=0)

        for col in value_cols:
            imputed_logs.append({
                'uid': uid,
                'sensor': sensor_type,
                'column': col,
                'imputed': imputed_count[col]
            })

        resampled['uid'] = uid
        resampled['type'] = sensor_type
        resampled = resampled.reset_index()
        cleaned_dfs.append(resampled)

    cleaned_all = pd.concat(cleaned_dfs, ignore_index=True)
    return cleaned_all, pd.DataFrame(imputed_logs)


def clean_accel(sensor_df):
    return interpolate_sensor(sensor_df, ['x', 'y', 'z'], 'accelerometer')

def clean_light(sensor_df):
    return interpolate_sensor(sensor_df, ['value'], 'light')

def clean_wifi(sensor_df):
    return interpolate_sensor(sensor_df, ['level'], 'wifi')

def clean_location(sensor_df):
    location_df = sensor_df[sensor_df['type'] == 'location'].copy()

    # Convert 'value' to numeric (represents distance)
    location_df['value'] = pd.to_numeric(location_df.get('value', np.nan), errors='coerce')

    # Interpolate x/y/z if they exist
    for col in ['x', 'y', 'z']:
        if col in location_df.columns:
            location_df[col] = pd.to_numeric(location_df[col], errors='coerce')
            location_df[col] = location_df[col].interpolate(limit_direction='both')

    # Drop rows where datetime is missing
    location_df = location_df.dropna(subset=['datetime'])

    return location_df

# Event-driven sensors: no interpolation

def clean_screen(sensor_df):
    screen_df = sensor_df[sensor_df['type'] == 'screen'].copy()
    return screen_df[['uid', 'datetime', 'type', 'value']], pd.DataFrame()

def clean_calls(sensor_df):
    calls_df = sensor_df[sensor_df['type'] == 'calls'].copy()
    return calls_df[['uid', 'datetime', 'type', 'sub_type', 'suuid', 'value', 'level', 'sensor_status']], pd.DataFrame()
