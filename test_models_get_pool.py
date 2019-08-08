#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 18:54:23 2019

@author: Beryl
"""

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle as pkl
from robotDQN_filter import RobotLSTMQ
from tagger_filter import Tagger
from game_ner_new2 import Env
from sklearn.metrics import f1_score
from collections import Counter
import csv
from joblib import Parallel, delayed
from apply_filter import random_sampling, uncertainty_sampling, diversity_sampling, conservative_sampling, least_confident
from time import sleep

import multiprocessing
import itertools
import helper

AGENT = "LSTMQ"
MAX_EPISODE = 100

##################################################################
# changable variables
#parser = argparse.ArgumentParser(description='')
#parser.add_argument('exp_num', type=int, nargs=1, help='Experiment number')
#parser.add_argument('model_type', type=str, nargs=1, help='Q-learning strategy')
#args = parser.parse_args()

# exp23: content 0, only softmax
EXPNUM = 23
NTYPE  = 'd1qn' #'d3qn' #'d1qn', 'd2qn', 'd3qn', 'd4qn']

NITER = 10

num_cores = multiprocessing.cpu_count()
print(num_cores)

BUDGETS = [20, 50, 100]
#NSTATES = [0, 32, 64, 128]
NSTATES = [0]
#BUDGETS = [100, 300]
POLYS = [False]
CUMS = [False] #cumulative


# test on only content true, running
# test on only fcls true, running
# test on only logits true, running
# test on only the softmax outputs
CONTENTS = [False]
#CONTENTS = [False]
#FCLS = True
FCLS = False
LOGIT = False
#LOGIT = False
FEATURES =[[1,2,3,4]]
METHOD = 'maj'
################################################################
#feature shape
FEATURE_SHAPE = [[378,10],[257,10],[70,10],[27,10],[24,10]]
#FEATURE = 'ALL','FACE', 'BODY', 'PHY', 'AUDIO']

def initialise_game(model, budget, niter, feature_number, student, cvit):
    
    train_x, train_y = helper.test_data(model, feature_number, student)
    test_x = dev_x = train_x
    print(len(train_y))
    story = [train_x, train_y]
    dev = [dev_x, train_y]
    test = [test_x, train_y]

    print("Loading environment..")
    game = Env(story, test, dev, budget, MODEL_VER, model,feature_number, CUM, EXPNUM, cvit, METHOD)
    return game

def play_ner(feature_now, model_ver, poly, cvit, logit, fcls, method, n_states):
    actions = 2
    global BUDGET
    tf.reset_default_graph()
    if AGENT == "LSTMQ":
        robot = RobotLSTMQ(actions, FEATURE, content = CONTENT, poly = poly, logit = logit, fcls = fcls, ntype = NTYPE, expnum = EXPNUM, n_states=n_states)
    else:
        print("** There is no robot.")
        raise SystemExit
        
   

    ############NEW###############################
    model_selected = []

    for i in feature_now:
        with tf.name_scope(model_ver+'/feature_{0}'.format(i)):
            model = Tagger(model_file=model_ver+'/feature_{0}'.format(i),
                           n_input=FEATURE_SHAPE[i][0],n_steps=FEATURE_SHAPE[i][1],feature_number=i, expnum = EXPNUM, cvit=cvit)
        model_selected.append(model)
        
    ###############################################
    
    # initialise a decision robot
    #TODO:
    
    # play game
    ID_student = 0
    test_child = []
    with open('test_child.txt') as f:
        for line in f.readlines():
            l = line.split()[0]
            test_child.append(l)

        #student is the individual test kid
    print(">>>>>> Playing game ..")
    test_game = initialise_game(model_selected,BUDGET,NITER,FEATURE, test_child[ID_student], cvit)
    
    conf_action = []
    while ID_student < len(test_child):
        observation = test_game.get_frame(model_selected)
        action, qvalue = robot.test_get_action(model_ver,observation, BUDGET)
        test_game.current_frame = test_game.current_frame + 1
        #if action[0] == 1:
        #print('> Action', action)
        if action[1] == 1:
            test_game.count_action1 += 1
            test_game.make_query =True
            test_game.query()
            conf_action.append(qvalue)
        else:
            test_game.count_action0 += 1
        ################ test mode A################
        if test_game.current_frame == len(test_game.train_y_all)-1:
            ##################################### pool data ################################
            if len(model_selected) == 1:
                pool_x = test_game.queried_set_x
            else:
                queried_set_xx = test_game.queried_set_x
                indexs = [0]
                a = 0
                for i in range(len(model_selected)):
                    a = a + model_selected[i].n_input
                    indexs.append(a)
                for i in range(len(queried_set_xx)):
                    queried_set_xx[i] = np.concatenate((test_game.queried_set_x[i]),axis=1)
                queried_set_xx = np.array(queried_set_xx)
                #for i in range(4):
                #    pool_x.append(queried_set_xx[:,:,indexs[i]:indexs[i+1]])
            pool_y = test_game.queried_set_y
            # pool_x in shape ([[N*F1], [N*F2] , [N*F3], [N*F4]])
            #pool = [pool_x, pool_y]
            if not os.path.exists('Filter_again_07_29_exp_{0}/test_{1}'.format(EXPNUM, cvit) + os.sep):
                os.makedirs('Filter_again_07_29_exp_{0}/test_{1}'.format(EXPNUM, cvit) + os.sep)
            np.save('Filter_again_07_29_exp_{0}/test_{1}/pool_x_budget_{2}_ID_{3}.npy'.format(EXPNUM, cvit,BUDGET, ID_student),queried_set_xx)
            np.save('Filter_again_07_29_exp_{0}/test_{1}/pool_y_budget_{2}_ID_{3}.npy'.format(EXPNUM, cvit,BUDGET, ID_student),pool_y)
            ################################################################################
            
            print('ID ', ID_student)
            print('> Terminal this child<')
            np.save('Filter_again_07_29_exp_{0}/test_{1}/budget_{2}_student_{3}.npy'.format(EXPNUM, cvit,BUDGET, ID_student), conf_action)
            #conf_action = []
            ID_student += 1
            if ID_student < len(test_child):
                test_game = initialise_game(model_selected,BUDGET,NITER,FEATURE, test_child[ID_student], cvit)
        
def main(in_iter):
    global AGENT, MAX_EPISODE, BUDGET, MODEL_VER, FEATURE, FEATURE_SHAPE, CONTENT, CUM, NITER, POLYS, NTYPE, EXPNUM
    

    BUDGET=in_iter[0]
    FEATURE = in_iter[1]
    cvit = in_iter[2]
    method = 'maj'
    n_states = in_iter[3]
    print(BUDGET, FEATURE)
    for CONTENT in CONTENTS:
        for CUM in CUMS:
            MODEL_VER_0 = 'model_hidden_{0}_it_{1}_budget_{2}_content_{3}_cum_{4}_logits_{5}_fcls_{6}_{7}'.format(n_states, NITER, BUDGET, int(CONTENT), int(CUM), int(LOGIT), int(FCLS), method)

            s=[0,0,0,0]
            fvar = '_feature'
            for i in range(np.shape(FEATURE)[0]):
                if FEATURE[i]:
                    s[FEATURE[i]-1]=1
            for i in range(np.shape(s)[0]):
                fvar = fvar+'_{0}'.format(s[i])

            MODEL_VER_0 = MODEL_VER_0 +str(fvar)
            POLY=False
            #The same model_ver
            ####################### test mode A #############################
            MODEL_VER = MODEL_VER_0 + '_poly_{0}'.format(int(POLY))
            print('test on model ', MODEL_VER)
            robot = play_ner(FEATURE, MODEL_VER, POLY, cvit, LOGIT, FCLS, method, n_states)
            tf.reset_default_graph()

                               
 
if __name__ == '__main__':
    #cvits = np.linspace(5,9,5).astype(int)
    cvits = [4]
    Parallel(n_jobs=num_cores)(delayed(main)(i) for i in itertools.product(BUDGETS,FEATURES,cvits, NSTATES))
    #main([5,[1,2,3,4],10,0])                           