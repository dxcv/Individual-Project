import glob
import numpy.random as random

path = '/data/orudovic/v5/'


file_list = glob.glob('/data/orudovic/CompiledFeaturesNew/*pkl')

test_child = random.choice(file_list, size=15, replace=False)

for i in test_child:
	file_list.remove(i)

with open(path + 'train_child.txt', 'w') as out:
    for i in file_list:
        out.write(i + '\n')


with open(path + 'test_child.txt', 'w') as out:
    for i in test_child:
        out.write(i + '\n')