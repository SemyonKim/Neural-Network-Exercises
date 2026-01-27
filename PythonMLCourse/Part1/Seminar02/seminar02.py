# -*- coding: utf-8 -*-

def a1(x):
  if x > 2.165910593024557:
    return -1
  else:
    return 1
    
def R():
  return 0.4654157689276084
    
def a2(x): 
  a = (((x[0]-0.7174210335926431)**2)/0.4993618181610397 + ((x[1]-0.8051975503768627)**2)/4.48524294354885 - ((x[0]-0.17993739697070765)**2)/1.1661549489331444 - ((x[1]-0.9654650474585901)**2)/2.385011087123411)/2 - 0.3472121797418795 + 0.7503963310295383 - 0.0768559842036564 - 0.4346018865092804 + 0.5 - 0.5

  if a >= 0:
    return -1
  else:
    return 1
    
if __name__ == '__main__':
  print(a1(0.0), R(), a2([0.0, 0.0]))
    
  with open('seminar02_task1.txt', 'w') as f:
    for i in range(-50, 50):
      x = i/10.0
      y = a1(x)
      f.write('%d ' % y)
            
  with open('seminar02_task2.txt', 'w') as f:
    f.write('%.3f' % R())   
    
  with open('seminar02_task3.txt', 'w') as f:
    for i in range(-50, 50):
      x1 = i/10.0
      for j in range(-50, 50):
        x2 = j/10.0
        y = a2([x1, x2])
        f.write('%d ' % y)