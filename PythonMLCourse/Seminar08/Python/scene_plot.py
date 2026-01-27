# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import random
import pandas as pd
import time

df = pd.read_csv("data.csv")
df = df.sort_values(by=['TRACK_ID', 'TIMESTAMP'])

df.head()

df['TRACK_ID'] = df['TRACK_ID'].str[-12:]
df["TRACK_ID"] = df["TRACK_ID"] +' (' +df["OBJECT_TYPE"]+')'
df.head()

df_count = df.groupby('TRACK_ID').count()
print(df_count)
df_tags = df_count.index
print(df_tags.values)

plt.figure(figsize=(17, 17))

linestyles=['-','--','-.',':','.',',','o','v','^','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','|','_']     

n = len(df_tags.values)
colors = plt.cm.jet(np.linspace(0,1,n))

for idx, i in enumerate(df_tags.values):
  data_x = df[df['TRACK_ID']==i][['X']].values
  data_y = df[df['TRACK_ID']==i][['Y']].values
  plt.plot(data_x, data_y, linestyles[idx], label=i.format(linestyles[idx]), color = colors[idx])

plt.grid() # Отобразим сетку
plt.xlabel("X") # Название оси X
plt.xticks(rotation=90) # Поворот значений по оси X
plt.ylabel("Y") # Название оси Y
plt.title("Визуализация дорожной сцены") # Название графика
plt.legend() # Отобразить легенду
plt.savefig("scene.png") # Любой график можно сохранить в файл
plt.show()