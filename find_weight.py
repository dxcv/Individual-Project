import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss
import os


### finding the optimum weights



def log_loss_func(weights, predictions, test_y):
	''' scipy minimize will pass the weights as a numpy array '''
	final_prediction = 0
	for weight, prediction in zip(weights, predictions):
			final_prediction += weight*prediction

	return log_loss(test_y, final_prediction)
   

def find_weight(models, test_x, test_y):
	#the algorithms need a starting value, right not we chose 0.5 for all weights
	#its better to choose many random starting points and run minimize a few times
	predictions = []
	for i, model in enumerate(models):
		predictions.append(model.get_marginal(test_x[i]))

	starting_values = [0.5]*len(predictions)

	#adding constraints  and a different solver as suggested by user 
	cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
	#our weights are bound between 0 and 1
	bounds = [(0,1)]*len(predictions)

	res = minimize(log_loss_func, starting_values, args=(predictions, test_y), method='SLSQP', bounds=bounds, constraints=cons)

	print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
	print('Best Weights: {weights}'.format(weights=res['x']))
	return res['x']
