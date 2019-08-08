import csv
import pandas as pd
import numpy as np
import regex as re
import glob
import matplotlib.pyplot as plt
import itertools as it 

#path = "records_conf/*.csv"
path = "/media/oggi2/DATA3/EngageMEProject/MRL/v10/EXP_3/records/*.csv"

#colnames = ['accuracy_train', 'accuracy_test', 'f1_train', 'f1_test', 'accuracy_majority_train','f1_majority_train','accuracy_majority_test','f1_majority_test', 'conf_train','conf_test']
#colnames2 = ['accuracy_train', 'accuracy_test', 'f1_train', 'f1_test', 'conf_train','conf_test']

colnames = ['reward_all', 'count_0', 'count_1']
#colnames2 = ['f1_train', 'f1_test', 'conf_train','conf_test']

bud = 50
con = 1

output = '/media/oggi2/DATA3/EngageMEProject/MRL/v10/reward_summary.csv'

dlist = list()

#model_it_10_budget_5_content_0_cum_0_feature_0_0_0_0_poly_0
flist = ['budget_','content_', 'cum_', 'poly_']

for fname, m in zip(glob.glob(path),it.cycle(['>','^','+','*','d','x','8','s','p'])):

    print(fname)
    fopt = re.findall('\d',fname)
    print(fopt)

    ntype = int(fopt[5])
    print(fopt, ntype)
    poly = int(fopt[-1])
    cum = int(fopt[-6])
    feature = fopt[-5:-1]
    content = int(fopt[-7])
    
    fname2 = fname.split('budget_', maxsplit=1)
    budget = int(fname2[1].split('_content')[0])



    #print('poly {0}, cum {1}, feature {2}. content {3}. budget {4}'.format(poly, cum, feature, content, budget))
    
    # create a filter 
    if 1:#content==0 and cum==0: # and content==con and budget==bud:
    
      fname2 = fname.split('content_', maxsplit=1)
    
      fopt = re.findall('\d',fname2[1])
      
      data = pd.read_csv(fname)
      pp= []

      for k in colnames:

          # if k in colnames2:

          #     pp = data[k].str.slice(1, -1)
          #     pp = pp.str.split(", ")
             
          #     cols = []
          #     for i in range(len(pp[0])):
          #         cols.append([])

          #     for i in pp:
          #         print(k)
          #         for j in range(len(i)):
          #             print(j)
          #             cols[j].append(np.float16(i[j]))

          #     res = '['
          #     for j in range(len(i)):
          #         if j == len(i)-1:
          #             res = res+str(np.mean(np.array(cols[j][-20:])))
          #         else:
          #             res = res+str(np.mean(np.array(cols[j][-20:])))+','                    
          #     res += ']'

          #     #print(res)
          #     fopt.append(res)

          # else: 

          pp.append(np.array(data[k], dtype=np.float16).reshape(-1))



      pp = np.array(pp)
      print(pp.shape)
      pp = pp[0]/(pp[1]+pp[2])
      print(len(pp))

      pp2 = []
      for i in range(1,len(pp)-1):
          if i < 15:
             pp2.append(np.mean(pp[1:i+1]))
          else:
             pp2.append(np.mean(pp[i-15:i+1]))

      #if k == 'reward_all':
      label = 'f: {0}, c: {1}, p: {2}, b: {3}, n: {4}'.format(feature, content, poly, budget, ntype)
      plt.step(range(len(pp2)),pp2[:], label = label, marker = m )



          # pp = pp.as_matrix()
          # pp = pp.reshape((pp.shape[0],-1))
          # res = np.mean(pp.astype(np.float16)[-20:])
          # fopt.append(res.astype(str))

      # plt.scatter(range(len(pp2)),pp2[:], c='blue')
      # plt.show()
plt.legend()
plt.show()


