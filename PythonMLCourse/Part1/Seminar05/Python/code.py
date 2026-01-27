# -*- coding: utf-8 -*-

import pandas as pd

df = pd.read_csv('/content/data.csv', float_precision ="high") # START
df.dropna(inplace=True) #1
df = df[~df["OBJECT_TYPE"].isin(["AV", "AGENT"])] #2
df_count = df.groupby('TRACK_ID').count() #3 
df_tags = df_count[df_count.TIMESTAMP > 10].index
df = df[df['TRACK_ID'].isin(df_tags)]
df_min_max = df.groupby('TRACK_ID').agg({'X': ['min', 'max'], 'Y':['min', 'max']}) #4
df_min_max_tags = df_min_max[((df_min_max['X']['max'] - df_min_max['X']['min']) <= 1) & ((df_min_max['Y']['max'] - df_min_max['Y']['min']) <= 1)].index
df = df[~df['TRACK_ID'].isin(df_min_max_tags)]
df = df.sort_values(by=['TRACK_ID', 'TIMESTAMP']) #5
df.to_csv('result.csv', index=False) #END.

df = pd.read_csv('/content/result.csv')
print(df)