import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class diversitySampling():

	def __init__(self, X, pool = [], budget = 0):

		self.X = X
		self.X2 = np.copy(X)
		self.k = budget
		self.m = X.shape[0]
		self.d = X.shape[1]
		self.C = list()
		for i in range(pool.shape[0]):
			self.C.append(pool[i])
		self.ind2 =list(range(self.m))
		self.newind = list()


	def updateCplus(self):

		if len(self.C) == 0:
			tmp = np.random.randint(self.m)
			xel = self.X[tmp]
			self.C.append(xel)
			self.newind.append(self.ind2[tmp])
			self.X2 = np.delete(self.X2,tmp,0)
			del self.ind2[tmp]


		cnt = 1
		while cnt < self.k:

			if len(self.ind2) == 0:				
				print(len(self.ind2))
				return self.C

			D = self.computeD()
			#print(len(D))
			D = D/(sum(D,0)+np.array(sum(D,0)==0).astype(int))
			cumprobs = D.cumsum()
			r = np.random.random()
			rind = np.where(cumprobs >= r)[0][0]
			self.C.append(self.X2[rind])
			self.X2 = np.delete(self.X2,rind,0)
			self.newind.append(self.ind2[rind])
			del self.ind2[rind]

			cnt+=1

		return self.C

	def computeD(self):
		
		D=[]
		for i in range(self.X2.shape[0]):
			tmpd = []
			for j in range(len(self.C)):
				d = np.sqrt(np.sum((self.X2[i]-self.C[j])**2/self.d))
				tmpd.append(d)				
			D.append(np.amin(tmpd))
		return D


if __name__ == "__main__":

	np.random.seed(0)
	dataset=pd.read_csv('Mall_Customers.csv')

	# data to select from
	data = dataset.iloc[:20, [3, 4]].values
	
	# if pool exists, otherwise pool = []
	pool = dataset.iloc[195:, [3, 4]].values 


	budget = 50

   
	s = diversitySampling(data, pool = pool, budget = budget)
	s.updateCplus()
	# returns indices of # budget most diverse examples in data; if budget > size(data), returns only budget 
	diversity = s.newind
	print(diversity)

   

	# print(np.sort(diversity))
	# plt.scatter(data[:,0],data[:,1], c='blue')
	# for k in range(len(s.newind)):
	# 	plt.scatter(data[s.newind[k],0],data[s.newind[k],1],c='red')

	# for k in range(len(s.C)-budget):
	# 	plt.scatter(s.C[k][0],s.C[k][1],c='green')
	# plt.show()
	
