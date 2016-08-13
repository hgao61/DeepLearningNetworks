import numpy as np
import csv
import random
from random import randint
rand = np.random.rand(1000,3)

inf = open('Data/best4KNN.csv','wb')
writer = csv.writer(inf)

writer.writerows(rand)
inf.close()