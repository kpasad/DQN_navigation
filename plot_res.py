import matplotlib.pyplot as plt
import pickle as pk
import glob
import sys
sys.path.insert(0, '../Value_methods')
from paramutils import *
import numpy as np

basepath = r'C:\Users\kpasad\Dropbox\ML\project\deep-reinforcement-learning-master\DQN_navigation\\'
all_pk = glob.glob(basepath+'*.pk')

ma_length = 100
for pk_file in all_pk:
    filename=pk_file.rsplit('\\')[-1]
    #legend = filename.rsplit('.')[0]
    [scores, params] = pk.load(open(pk_file,'rb'))
    legend = ('dbl' if params.double_dqn=='enable' else 'none')+"_"+params.network
    ma= np.convolve(scores, np.ones(ma_length), 'valid') / ma_length
    plt.plot(ma,label=legend)

plt.legend(loc="lower right")
plt.grid(b=True, which='both', color='0.65', linestyle='-')
plt.xlabel('Episodes')
plt.ylabel('Moving Average Scores (window ='+str(ma_length)+')')
plt.show()