# -*- coding: utf-8 -*-

def MyReaderFunc(filename_):
    with open(filename_) as f: 
        for line in f:
          line = line.strip()
          line = line.split(',')
          yield line 

def pure_euclidean_distance(Ax,Ay,Bx,By):
  if not (isinstance(Ax, (int, float)) and isinstance(Ay, (int, float)) and isinstance(Bx, (int, float)) and isinstance(By, (int, float))):
    raise ValueError('Given values are not numeric')
  return ((Bx-Ax)**2+(By-Ay)**2)**(1/2)

def MyGenerator(filename):
  speed_limit = 40 * 5 / 18
  res_dict ={}
  violators = []
  skip_first = True
  
  for tup in MyReaderFunc(filename):

      if skip_first:
        skip_first = False
        continue

      if tup[1] in res_dict:
        distance = pure_euclidean_distance(res_dict[tup[1]]['x'],res_dict[tup[1]]['y'],float(tup[3]),float(tup[4]))
        if speed_limit < distance/(float(tup[0])-res_dict[tup[1]]['time']):
          res_dict[tup[1]]['during']=res_dict[tup[1]]['during'] + (float(tup[0])-res_dict[tup[1]]['time'])
        else:
          res_dict[tup[1]]['during']=0.    
      else: 
        res_dict.setdefault(tup[1], {})['during'] = 0.
      
      res_dict.setdefault(tup[1], {})['time'] = float(tup[0]) 
      res_dict.setdefault(tup[1], {})['x'] = float(tup[3]) 
      res_dict.setdefault(tup[1], {})['y'] = float(tup[4])
  
      if res_dict[tup[1]]['during'] >= 1.:
        violators.append(tup[1]) 

  return set(violators)

MyGenerator('data.csv')