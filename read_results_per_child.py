import csv
import pandas as pd
import numpy as np
import regex as re
import glob

#path = "records_conf/*.csv"
path = "tests/latest_all/*.csv"
bud = '100'

#colnames = ['accuracy_train', 'accuracy_test', 'f1_train', 'f1_test', 'accuracy_majority_train','f1_majority_train','accuracy_majority_test','f1_majority_test', 'count_0', 'count_1', 'conf_train','conf_test']
#colnames2 = ['accuracy_train', 'accuracy_test', 'f1_train', 'f1_test', 'conf_train','conf_test']

colnames = ['count_0_B', 'count_1_B', 'conf_test_A', 'conf_test_B', 'accuracy_majority_test_A',  'accuracy_majority_test_B', 'f1_majority_test_A', 'f1_majority_test_B']
colnames2 = ['conf_test_A','conf_test_B']


#output = 'results_summary_conf_5.csv'
output = 'results_summary_pc_majority_content.csv'
#output = 'results_summary_majority_conf_20.csv'

def write_csv(dats):

	f = open(output, "a")
	writer = csv.DictWriter(
		f, fieldnames=["content", 
						 "cum", 
						 "feature_a",
						 "feature_b",
						 "feature_c",
						 "feature_d",
						 "poly",
						 "countB_0",
						 "countB_1",
						 "confA_test",
						 "confB_test",
						 'accA',
						 'accB',
						 'f1A',
						 'f1B'])

	writer.writeheader()

	with f as csvFile:
			writer = csv.writer(csvFile)
			writer.writerows(dats)
			writer.writerows('\n')

	csvFile.close()
	print('csv saved')


dlist = list()

#model_it_10_budget_5_content_0_cum_0_feature_0_0_0_0_poly_0
flist = ['budget_','content_', 'cum_', 'poly_']

for fname in glob.glob(path):

		print(fname)
		fopt = re.findall('\d',fname)

		poly = int(fopt[-1])
		cum = int(fopt[-6])
		feature = fopt[-5:-1]
		content = int(fopt[-7])
		
		fname2 = fname.split('budget_', maxsplit=1)
		budget = fname2[1].split('_content')[0]



		print('poly {0}, cum {1}, feature {2}. content {3}. budget {4}'.format(poly, cum, feature, content, budget))
		
		# create a filter 
		if poly==0 and cum==0 and content==1 and budget==bud:
		
			fname2 = fname.split('content_', maxsplit=1)
		
			fopt = re.findall('\d',fname2[1])
			
			data = pd.read_csv(fname)

			for k in colnames:

					if k in colnames2:

							pp = data[k].str.slice(1, -1)
							pp = pp.str.split(", ")
						 
							cols = []
							for i in range(len(pp[0])):
									cols.append([])

							for i in pp:
									print(k)
									for j in range(len(i)):
											print(j)
											cols[j].append(np.float16(i[j]))

							res = '['
							for j in range(len(i)):
									if j == len(i)-1:
											res = res+str(np.mean(np.array(cols[j])))
									else:
											res = res+str(np.mean(np.array(cols[j])))+','                    
							res += ']'

							print(res)
							fopt.append(res)

					else: 

							pp = data[k]
							pp = pp.as_matrix()
							pp = pp.reshape((pp.shape[0],-1))
							res = np.mean(pp.astype(np.float16))
							fopt.append(res.astype(str))

			dlist.append(fopt)

print(dlist)
write_csv(dlist)

 # write the results summary to a text file

