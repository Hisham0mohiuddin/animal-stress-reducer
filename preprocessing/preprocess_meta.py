import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

if __name__ =="__main__":
    df = pd.read_csv("metadata.csv")
    print(df.head())

    df.drop(columns=['english_name','location','length','temp','sample_rate','quality'], axis=1, inplace=True)
    print(df.head())
    # df['time']= pd.to_datetime(df['time'])
    df['date']= pd.to_datetime(df['date'])
    # df['hour'] = df['time'].dt.hour
    df['month'] = df['date'].dt.month
    df['date']= df['date'].dt.day
    print(df.head())

    
    df.loc[df['type'].astype(str).str.contains('alarm call', case=False, na=False), 'type'] = 'alarm call'
    mask_call = (df['type'].astype(str).str.contains('call', case=False, na=False)) & (df['type'] != 'alarm call')
    df.loc[mask_call, 'type'] = 'call'
    df.loc[df['type'].astype(str).str.contains('song', case=False, na=False), 'type'] = 'song'

    print(df.head())
    df['time'] = pd.to_datetime(df['time'], format='%H:%M',errors='coerce')
    df['hour'] = df['time'].dt.hour
    
    df.drop(columns=['time'], axis=1, inplace=True)
    print(df.head())

    df.to_csv("metadata_processed.csv")

    print(len(np.unique_values(df['country'])))

