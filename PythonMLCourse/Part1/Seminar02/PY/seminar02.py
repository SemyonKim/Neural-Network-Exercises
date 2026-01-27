# -*- coding: utf-8 -*-

import pandas as pd

data = pd.read_csv('141.csv') 
print(data)

objects_in_the_script = set(data['TRACK_ID'])
print("the number of objects in the script: ",len(objects_in_the_script))

def distance_eu(p,q):
  return math.sqrt((q[0]-p[0])**2 + (q[1]-p[1])**2)

import math

longest_trajectory={'length': -1, 'track': 'noname'}

for track in objects_in_the_script:
  track_trajectory = 0
  track_coord = data.loc[data['TRACK_ID']==track]
    
  pre = [0,0] #x y
  cur = [0,0]
  skip_first_iter = True
  for index, row in track_coord.iterrows():
    pre = cur.copy()
    cur[0]=row['X']
    cur[1]=row['Y']
     
    if skip_first_iter:
      skip_first_iter = False
      continue 

    dist = distance_eu(pre, cur)
    track_trajectory = track_trajectory + dist 

  #print(track,track_trajectory, longest_trajectory['length'])
  if track_trajectory > longest_trajectory['length']:
     longest_trajectory['track']=track
     longest_trajectory['length']=track_trajectory
print(longest_trajectory)
longest_trajectory['length']=round(longest_trajectory['length'],2)
print('after round down to 2 decimals a float: ', longest_trajectory)