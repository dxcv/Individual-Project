#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:45:05 2019

@author: beryl
"""
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
from game_ner_new import Env
from robotDQN_new import RobotLSTMQ
import numpy as np
import helper_new as helper
import tensorflow as tf
import random
from tagger_new import Tagger
import argparse
from joblib import Parallel, delayed
import multiprocessing
import itertools

#parser = argparse.ArgumentParser(description='')
#parser.add_argument('exp_num', type=int, nargs=1, help='Experiment number')
#parser.add_argument('model_type', type=str, nargs=1, help='Q-learning strategy')
#args = parser.parse_args()
EXPNUM = 23 #args.exp_num[0]
NTYPE  = 'd1qn' #args.model_type[0] #'d3qn' #'d1qn', 'd2qn', 'd3qn', 'd4qn']


num_cores = multiprocessing.cpu_count()
print(num_cores)

"""
https://github.com/jasimpson/ez2ec2
"""


AGENT = "LSTMQ"
MAX_EPISODE = 100

##################################################################
# changable variables
BUDGETS = [5, 10, 20, 50, 100]
NITERS = [10] # number of epochs 
POLYS = [False]
CUMS = [False] #cumulative


# test on only content true, running
# test on only fcls true, running
# test on only logits true, running
# test on only the softmax outputs
#CONTENTS = [True]
CONTENTS = [False]
#FCLS = True
FCLS = False
#LOGIT = True
LOGIT = False
FEATURES =[[1,2,3,4]]
#################################################################
#feature shape
FEATURE_SHAPE = [[378,10],[257,10],[70,10],[27,10],[24,10]]
METHODS = ['auto']


def initialise_game(model, budget, niter, feature_number, method):

	train_x_all = []
	test_x_all  = []
	
	for i in range(4):
		n_shape = FEATURE_SHAPE[i+1][0]
		train_x, train_y = helper.load_traindata(i+1,n_shape, seed=0)
		train_x_all.append(train_x)
		train_y_all = train_y
		test_x, test_y = helper.load_testdata(i+1,n_shape, seed=0)
		test_x_all.append(test_x)
		test_y_all = test_y
	dev_x_all = test_x_all
	dev_y_all = test_y_all
	
	story = [train_x_all, train_y_all]
	dev = [dev_x_all, dev_y_all]
	test = [test_x_all, test_y_all]
	print('shape data ', len(train_x_all))
	
	# load game
	game = Env(story, test, dev, budget, MODEL_VER, model,feature_number, CUM, EXPNUM, 0, method)
	return game


def play_ner(feature_now, model_ver, poly, niter, logit, fcls,  method):
	actions = 2
	global BUDGET
	
	tf.reset_default_graph()
	if AGENT == "LSTMQ":
		robot = RobotLSTMQ(actions, FEATURE, content = CONTENT, poly = poly, logit = logit, fcls=fcls, ntype = NTYPE, expnum = EXPNUM)
	else:
		print("** There is no robot.")
		raise SystemExit

	############NEW###############################
	model_selected = []

	with tf.name_scope(model_ver):
		model = Tagger(model_file=model_ver+'/feature_5',
							n_input=FEATURE_SHAPE[0][0],n_steps=FEATURE_SHAPE[0][1],feature_number=5, epochs=niter, expnum = EXPNUM)
	model.train([],[],feature_number = 5)
	model_selected.append(model)


	game = initialise_game(model_selected,BUDGET,NITER,FEATURE, method)
	
	
 
	###############################################
	
	# initialise a decision robot
	
	# play game
	episode = 1

	rAll = []
	while episode <= MAX_EPISODE:

		observation = game.get_frame(model_selected)
		action = robot.get_action(observation)

		reward, observation2, terminal = game.feedback(action, model_selected)
		game.rAll.append(reward)
		rAll.append(reward)

		robot.update(observation, action, reward, observation2, terminal)

		if terminal == True:
			print("> Episodes finished: ", float("%.3f" % (episode/MAX_EPISODE)), "> Reward: ", float("%.3f" % np.mean(rAll)))
			episode += 1
			rAll = []
			if episode == MAX_EPISODE:
				robot.save_Q_network(MODEL_VER)   
	return robot


def main(in_iter):

	global AGENT, MAX_EPISODE, MODEL_VER, FEATURE_SHAPE, CONTENT, CUM, NITER, POLYS, NTYPE, BUDGET, FEATURE

	
	#for budget in BUDGETS:
	BUDGET=in_iter[0]
	FEATURE = in_iter[1]
	method = in_iter[2]

	for NITER in NITERS:
		for CONTENT in CONTENTS:
			for CUM in CUMS:
				MODEL_VER_0 = 'model_{0}_it_{1}_budget_{2}_content_{3}_cum_{4}_logits_{5}_fcls_{6}_{7}'.format(NTYPE, NITER, BUDGET, int(CONTENT), int(CUM), int(LOGIT), int(FCLS), method)

				#s=[0,0,0,0]
				'''
				fvar = '_feature'
				for i in range(np.shape(FEATURE)[0]):
					if FEATURE[i]:
						s[FEATURE[i]-1]=1
				for i in range(np.shape(s)[0]):
					fvar = fvar+'_{0}'.format(s[i])
				'''
				fvar = '_feature_2_0_0_0'
				MODEL_VER_0 += str(fvar)

				if CONTENT: 

					POLY=False
					MODEL_VER = MODEL_VER_0 + '_poly_{0}'.format(int(POLY))		

				else:

					for POLY in POLYS:
						MODEL_VER = MODEL_VER_0 + '_poly_{0}'.format(int(POLY))

				robot = play_ner(FEATURE, MODEL_VER, POLY, NITER, LOGIT, FCLS, method)
				tf.reset_default_graph()


if __name__ == '__main__':
	#main([5,[5],'auto'])
	Parallel(n_jobs=num_cores-2)(delayed(main)(i) for i in itertools.product(BUDGETS,FEATURES, METHODS))

