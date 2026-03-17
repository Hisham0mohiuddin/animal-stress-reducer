import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    print(np.unique_values(df['type']))