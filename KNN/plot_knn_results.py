import numpy as np
import csv
from random import randint
import random
import matplotlib.pyplot as plt


ISRMSE=[0.474452933563,0.262752734131,0.136590187312,0.250286457124,0.430180373541,0.607782800242,0.789919670626,0.961440272706,1.12919986106,1.27859977717]
ISCORR=[1.0,0.985971678512,0.981360326901,0.975383222588,0.973427600347,0.97161540002,0.967528179895,0.964988623322,0.959418573483,0.954402050655]
OSRMSE=[0.483481725151,0.301746836891,0.207762150054,0.278516588212,0.42817402625,0.580325766777,0.727981198392,0.898930614199,1.05049512469,1.19067064467]
OSCORR=[0.944111176194,0.952952269598,0.955537498166,0.954609910672,0.951772965431,0.946508234451,0.944268967869,0.938383482968,0.93599266378,0.934828370822]
#print ISRMSE
K=[1,2,3,4,5,6,7,8,9,10]
#print K
#plt.plot(K,'*')
ISRMSE,=plt.plot(K,ISRMSE, label='In Sample RMSE')
OSRMSE,=plt.plot(K,OSRMSE, label='Out of Sample RMSE')
plt.ylabel('RMSE')
plt.xlabel('K')
plt.legend(handles=[ISRMSE,  OSRMSE],bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show
plt.savefig('ISRMSE.jpg')
plt.close()

ISCORR,=plt.plot(K,ISCORR, label='In Sample CORR')
OSCORR,=plt.plot(K,OSCORR, label='Out of Sample CORR')
plt.ylabel('CORR')
plt.xlabel('K')
plt.legend(handles=[ ISCORR,  OSCORR],bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show
plt.savefig('ISCORR.jpg')