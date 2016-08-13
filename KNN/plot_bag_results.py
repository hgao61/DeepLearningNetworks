import numpy as np
import csv
from random import randint
import random
import matplotlib.pyplot as plt


ISRMSE=[0.158923802536,0.1545686864,0.151138058055,0.152562661933,0.152274192925,0.151617854795,0.15199460099,0.152252918529,0.151423565086]
ISCORR=[0.976827802758,0.978577334861,0.979732770328,0.97932226814,0.979465423666,0.979650285527,0.979502457149,0.979409144809,0.979783666763]
OSRMSE=[0.221170885886,0.212821653225,0.208435104237,0.212714592755,0.210646477943,0.210868996627,0.212564253547,0.209769637671,0.211007042167]
OSCORR=[0.953305595814,0.958019318507,0.959791584154,0.958022525508,0.958931479604,0.95897537901,0.957951146865,0.959319540658,0.958829661479]
#print ISRMSE
bags=[10,20,40,60,80,100,120,140,200]
#print K
#plt.plot(K,'*')
ISRMSE,=plt.plot(bags,ISRMSE, label='In Sample RMSE')
OSRMSE,=plt.plot(bags,OSRMSE, label='Out of Sample RMSE')

plt.ylabel('RMSE')
plt.xlabel('bags')
plt.legend(handles=[ISRMSE,  OSRMSE],bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show
plt.savefig('BAGRMSE0.jpg')
plt.close()
plt.ylabel('CORR')
plt.xlabel('bags')
ISCORR,=plt.plot(bags,ISCORR, label='In Sample CORR')
OSCORR,=plt.plot(bags,OSCORR, label='Out of Sample CORR')
plt.legend(handles=[ISCORR,  OSCORR],bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig('BAGCORR0.jpg')