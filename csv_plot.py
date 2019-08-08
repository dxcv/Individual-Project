# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:12:40 2019

@author: dell
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:43:50 2019

@author: oggi
"""

import numpy
import pandas as pd
import matplotlib.pyplot as plt


filepath = 'records/csv_20_150_proba3_500'
savepath = filepath
df=pd.read_csv(filepath+'.csv', sep=',',header=0)



plt.plot(df['episode_number'],df['accuracy_train'])
plt.plot(df['episode_number'],df['accuracy_test']) 
plt.plot(df['episode_number'],df['f1_train'])
plt.plot(df['episode_number'],df['f1_test'])
 
plt.legend(('accuracy_train', 'accuracy_test', 'f1_train', 'f1_test'),
           loc='bottom right', shadow=True)
plt.xlabel('episode-number')
plt.ylim((0.2,0.7))
plt.title('20 training, 150 budget')
plt.savefig(filepath+'.png')