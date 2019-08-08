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
from game_ner import Env
from robotDQN import RobotLSTMQ
import numpy as np
import helper
import tensorflow as tf
import random
from tagger import Tagger
import argparse
from joblib import Parallel, delayed
import multiprocessing
import itertools
import find_weight

#parser = argparse.ArgumentParser(description='')
#parser.add_argument('exp_num', type=int, nargs=1, help='Experiment number')
#parser.add_argument('model_type', type=str, nargs=1, help='Q-learning strategy')
#args = parser.parse_args()
EXPNUM = 11 #args.exp_num[0]
NTYPES  = ['d1qn'] #args.model_type[0] #'d3qn' #'d1qn', 'd2qn', 'd3qn', 'd4qn']

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
CONTENTS = [False]
CUMS = [False] #cumulative
FEATURES =[[0],[1],[2],[3],[4],[1,2,3,4]]
#################################################################
#feature shape
FEATURE_SHAPE = [[378,10],[257,10],[70,10],[27,10],[24,10]]
METHODS = ['maj']
#FEATURE = 'ALL','FACE', 'BODY', 'PHY', 'AUDIO']


def initialise_game(model, budget, niter, feature_number, method):

    train_x_all = []
    test_x_all  = []
    dev_x_all = []
    for i in range(len(model)):
        train_x, train_y = helper.load_traindata(feature_number[i],model[i], seed=0)
        train_x_all.append(train_x)
        train_y_all = train_y
        test_x, test_y = helper.load_testdata(feature_number[i],model[i], seed=0)
        test_x_all.append(test_x)
        test_y_all = test_y
        #dev_x, dev_y = helper.load_testdata(feature_number[i], model[i], seed=3)
        #dev_x_all.append(dev_x)
        #dev_y_all = dev_y
    dev_x_all = test_x_all
    dev_y_all = test_y_all
    
    
    story = [train_x_all, train_y_all]
    dev = [dev_x_all, dev_y_all]
    test = [test_x_all, test_y_all]
    
    # load game
    game = Env(story, test, dev, budget, MODEL_VER, model,feature_number, CUM, EXPNUM, 0, method)
    return game


def play_ner(feature_now, model_ver, poly, niter, logit, method):
    actions = 2
    global BUDGET
    
    tf.reset_default_graph()
    if AGENT == "LSTMQ":
        robot = RobotLSTMQ(actions, FEATURE, content = CONTENT, poly = poly, logit = logit, ntype = NTYPE, expnum = EXPNUM)
    else:
        print("** There is no robot.")
        raise SystemExit

    ############NEW###############################
    model_selected = []

    for i in feature_now:
        with tf.name_scope(model_ver+'/feature_{0}'.format(i)):
            model = Tagger(model_file=model_ver+'/feature_{0}'.format(i),
                                n_input=FEATURE_SHAPE[i][0],n_steps=FEATURE_SHAPE[i][1],feature_number=i, epochs=niter, expnum = EXPNUM)
        model.train([],[],feature_number = i)
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
                print('in')
                robot.save_Q_network(MODEL_VER)
                weights = find_weight.find_weight(model_selected, game.dev_x_all, game.dev_y_all)
                np.save(model_ver+'.npy', weights)
                print(weights)
    return robot


def main(in_iter):

    global AGENT, MAX_EPISODE, MODEL_VER, FEATURE_SHAPE, CONTENT, CUM, NITER, POLYS, NTYPE, BUDGET, FEATURE

    
    #for budget in BUDGETS:
    BUDGET=in_iter[0]
    FEATURE = in_iter[1]
    method = in_iter[2]
    NTYPE = in_iter[3]
    logit = False
    for NITER in NITERS:
        for CONTENT in CONTENTS:
            for CUM in CUMS:
                MODEL_VER_0 = 'model_{0}_it_{1}_budget_{2}_content_{3}_cum_{4}_logits_{5}_{6}'.format(NTYPE, NITER, BUDGET, int(CONTENT), int(CUM), int(logit), method)

                s=[0,0,0,0]

                fvar = '_feature'
                for i in range(np.shape(FEATURE)[0]):
                    if FEATURE[i]:
                        s[FEATURE[i]-1]=1
                for i in range(np.shape(s)[0]):
                    fvar = fvar+'_{0}'.format(s[i])

                MODEL_VER_0 += str(fvar)

                if CONTENT: 

                    POLY=False
                    MODEL_VER = MODEL_VER_0 + '_poly_{0}'.format(int(POLY))        

                else:

                    for POLY in POLYS:
                        MODEL_VER = MODEL_VER_0 + '_poly_{0}'.format(int(POLY))

                robot = play_ner(FEATURE, MODEL_VER, POLY, NITER, logit, method)
                tf.reset_default_graph()


if __name__ == '__main__':
    #main([20,[1,2,3,4],'maj','d1qn'])
    Parallel(n_jobs=num_cores-2)(delayed(main)(i) for i in itertools.product(BUDGETS,FEATURES, METHODS, NTYPES))