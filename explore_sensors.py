import pandas as pd
def explore_sensors(sensor_df, session_name):
    print(f"\n--- Exploring sensors in {session_name} ---")
    sensor_types = sensor_df['type'].unique()
    print(f"Found sensor types: {sensor_types}\n")

    for sensor_type in sensor_types:
        count = sensor_df[sensor_df['type'] == sensor_type].shape[0]
        print(f"Sensor '{sensor_type}': {count} samples")
