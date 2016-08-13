import numpy as np
import csv
from random import randint
import random
import matplotlib.pyplot as plt
#generate two random numbers within range
a = randint(-100,100)
b = randint(-100,100)
#generate linear array y=ax+b with 1000 samples
start =-1
stop = 1
divStep = 1000
x = [ (1.0/divStep)*x for x in range(start*divStep, stop*divStep)]
y = [x1*a+b for x1 in x]
print y
#generate random points using x and y
newCoordsX=[]
newCoordsY=[]
deviationFromPoint = 1
for i in range(0,len(x)):
  newCoordsX.append(x[i] + random.gauss(1,0.15) * deviationFromPoint)
  newCoordsY.append(y[i] + random.gauss(1,0.15)* deviationFromPoint)
newPoint = zip(newCoordsX,newCoordsY)
#print newPoint
plt.plot(newCoordsX,newCoordsY,"*")
#plt.plot(x,y,'.r')
plt.show()
rand = zip(newCoordsX, newCoordsY, x)
#save the data into cvs file
inf = open('Data/best4linreg.csv','wb')
writer = csv.writer(inf)
writer.writerows(rand)
inf.close()