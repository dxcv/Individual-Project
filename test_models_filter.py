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
from apply_filter import random_sampling, uncertainty_sampling, diversity_sampling
from apply_filter import conservative_sampling, least_confident
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

BUDGETS = [5, 10, 20, 50, 100]
ALMETHODS = ['rs','us','ds','cs','lc']
fndict = {"rs": lambda feature_now, model_selected, budget, train_data: random_sampling(feature_now, model_selected, budget, train_data), 
          "us": lambda feature_now, model_selected, budget, train_data: uncertainty_sampling(feature_now, model_selected, budget, train_data), 
          "ds": lambda feature_now, model_selected, budget, train_data: diversity_sampling(feature_now, model_selected, budget, train_data), 
          "cs": lambda feature_now, model_selected, budget, train_data: conservative_sampling(feature_now, model_selected, budget, train_data), 
          "lc": lambda feature_now, model_selected, budget, train_data: least_confident(feature_now, model_selected, budget, train_data)}
BUDGET_100 = [100,16,100,100,67,76,100,100,41,31,54,19,68,100]

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
    
    story = [train_x, train_y]
    dev = [dev_x, train_y]
    test = [test_x, train_y]

    print("Loading environment..")
    game = Env(story, test, dev, budget, MODEL_VER, model,feature_number, CUM, EXPNUM, cvit, METHOD)
    return game

def play_ner(feature_now, model_ver, poly, cvit, logit, fcls, method, n_states, al_method):
    global BUDGET
    tf.reset_default_graph()        
   

    ############NEW###############################
    model_selected = []

    for i in feature_now:
        with tf.name_scope(model_ver+'/feature_{0}'.format(i)):
            model = Tagger(model_file=model_ver+'/feature_{0}'.format(i),
                           n_input=FEATURE_SHAPE[i][0],n_steps=FEATURE_SHAPE[i][1],
                           feature_number=i, expnum = EXPNUM, cvit=cvit)
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
    #conf_action = []
    while ID_student < len(test_child):
        ##################################### pool data ################################
        if len(model_selected) == 1:
           pool_x = test_game.queried_set_x
        queried_set_xx = np.load('Filter_again_07_29_exp_{0}/test_{1}/pool_x_budget_{2}_ID_{3}.npy'.format(EXPNUM, cvit, BUDGET, ID_student))                
        indexs = [0, 257, 327, 354, 378]
        pool_x = []
        for i in range(4):
            pool_x.append(queried_set_xx[:,:,indexs[i]:indexs[i+1]])
        pool_y = np.load('Filter_again_07_29_exp_{0}/test_{1}/pool_y_budget_{2}_ID_{3}.npy'.format(EXPNUM, cvit, BUDGET, ID_student))
        # pool_x in shape ([[N*F1], [N*F2] , [N*F3], [N*F4]])
        pool = [pool_x, pool_y]
        print('pool loaded')

        ################################################################################
        ############################ baseline active learning###########################
        budget = BUDGET
        if len(pool_y) < budget:
            budget = len(pool_y)
        if al_method == 'rs':
            queried_indexes = random_sampling(feature_now, model_selected, budget, pool, cvit)
        else:
            queried_indexes = fndict[al_method](feature_now, model_selected, budget, pool)


        for i in range(len(model_selected)):
            ################################# select required indexes #####################################
            # model_selected[i].train_mode_B(np.array(test_game.train_x_all[i])[queried_indexes], np.array(test_game.train_y_all)[queried_indexes], test_game.feature[i], mode=al_method)
            model_selected[i].train_mode_B(np.array(pool_x[i])[queried_indexes], np.array(pool_y)[queried_indexes], test_game.feature[i], mode=al_method)
        write_test_csv(test_game, model_selected, ID_student, model_ver, cvit, al_method)
        
        print('ID ', ID_student)
        print('> Terminal this child<')
        #np.save('Filter_07_29_exp_{0}/test_{1}/budget_{2}_student_{3}.npy'.format(EXPNUM, cvit,BUDGET, ID_student), conf_action)
        #conf_action = []
        ID_student += 1
        if ID_student < len(test_child):
            test_game = initialise_game(model_selected,BUDGET,NITER,FEATURE, test_child[ID_student], cvit)
            
            

def write_test_csv(test_game, model, student_ID, model_ver, cvit, al_method):
    csv_name = 'Filpool_base_exp_{0}/test_{1}/results/{2}_'.format(EXPNUM, cvit, al_method)+str(model_ver)+'.csv'
    if not os.path.exists('Filpool_base_exp_{0}/test_{1}/results'.format(EXPNUM, cvit) + os.sep):
            os.makedirs('Filpool_base_exp_{0}/test_{1}/results'.format(EXPNUM, cvit) + os.sep)

    f = open(csv_name, "a")
    
    writer = csv.DictWriter(
        f, fieldnames=["student_ID", 
                       #"accuracy_test_A",
                       #"f1_test_A",
                       "accuracy_majority_test_A",
                       "f1_A",
                       "f1_majority_test_A",
                       "conf_test_A",
                       #"count_0_B",
                       #"count_1_B",
                       "accuracy_majority_test_{0}".format(al_method),
                       "f1_{0}".format(al_method),
                       "f1_majority_test_{0}".format(al_method),
                       "conf_test_{0}".format(al_method)
                       ])
    
    if student_ID == 0:
        writer.writeheader()
    
    accuracy_test_A = []
    f1_test_A = []
    accuracy_majority_test_A = []
    f1_majority_test_A = []
    conf_test_A = []
    predd_test_A = []
    y_majority_test_A = []
    
    accuracy_test_B = []
    f1_test_B = []
    accuracy_majority_test_B = []
    f1_majority_test_B = []
    conf_test_B = []
    predd_test_B = []
    y_majority_test_B = []


    for i in range(len(model)):
        accuracy_test_A.append(float("%.3f" % model[i].test(test_game.test_x_all[i],test_game.test_y_all)))
        f1_test_A.append(float("%.3f" % model[i].get_f1_score(test_game.test_x_all[i],test_game.test_y_all)))
                
        accuracy_test_B.append(float("%.3f" % model[i].test_B(test_game.test_x_all[i],test_game.test_y_all, mode=al_method)))
        f1_test_B.append(float("%.3f" % model[i].get_f1_score_B(test_game.test_x_all[i],test_game.test_y_all, mode=al_method)))
        
        
        if len(model) ==1:
            conf_test_A.append(float("%.3f" % np.mean(model[i].get_confidence(test_game.test_x_all[i]))))
            conf_test_B.append(float("%.3f" % np.mean(model[i].get_confidence_B(test_game.test_x_all[i]))))
            predd_test_A.append(model[i].get_predictions(np.squeeze(test_game.test_x_all[i],axis=0)))
            predd_test_B.append(model[i].get_predictions_B(np.squeeze(test_game.test_x_all[i],axis=0)))
        else:
            conf_test_A.append(float("%.3f" % np.mean(model[i].get_confidence(list(test_game.test_x_all[i])))))
            conf_test_B.append(float("%.3f" % np.mean(model[i].get_confidence_B(list(test_game.test_x_all[i]), mode=al_method))))
            predd_test_A.append(model[i].get_predictions(test_game.test_x_all[i]))
            predd_test_B.append(model[i].get_predictions_B(test_game.test_x_all[i], mode=al_method))
    
    predd_test_A = np.array(predd_test_A).T.tolist()
    predd_test_B = np.array(predd_test_B).T.tolist()
    
    for i in range(len(predd_test_A)):
        preds_test_A = Counter(predd_test_A[i])
        preds_test_B = Counter(predd_test_B[i])
        y_majority_test_A.append(preds_test_A.most_common(1)[0][0])
        y_majority_test_B.append(preds_test_B.most_common(1)[0][0])

    f1_A = f1_score(test_game.train_y_all, y_majority_test_A, average=None)
    f1_majority_test_A = f1_score(test_game.train_y_all, y_majority_test_A, average='macro')
    accuracy_majority_test_A = sum(np.equal(test_game.test_y_all, y_majority_test_A))/len(test_game.test_y_all)
    f1_B = f1_score(test_game.test_y_all, y_majority_test_B, average=None)
    f1_majority_test_B = f1_score(test_game.train_y_all, y_majority_test_B, average='macro')
    accuracy_majority_test_B = sum(np.equal(test_game.test_y_all, y_majority_test_B))/len(test_game.test_y_all)

    writer.writerow({"student_ID": student_ID, 
                     #"accuracy_test_A": accuracy_test_A,
                     #"f1_test_A": f1_test_A,
                     "accuracy_majority_test_A": float("%.3f" % accuracy_majority_test_A),
                     "f1_A": f1_A,
                     "f1_majority_test_A": float("%.3f" % f1_majority_test_A),
                     "conf_test_A": conf_test_A,
                     #"count_0_B": test_game.count_action0,
                     #"count_1_B": test_game.count_action1,
                     #"accuracy_test_{0}".format(al_method): accuracy_test_B,
                     #"f1_test_{0}".format(al_method): f1_test_B,
                     "accuracy_majority_test_{0}".format(al_method): float("%.3f" % accuracy_majority_test_B),
                     "f1_{0}".format(al_method): f1_B,
                     "f1_majority_test_{0}".format(al_method): float("%.3f" % f1_majority_test_B),
                     "conf_test_{0}".format(al_method): conf_test_B
                     })

    print('csv saved')
    f.close()
        
def main(in_iter):
    global AGENT, MAX_EPISODE, BUDGET, MODEL_VER, FEATURE, FEATURE_SHAPE, CONTENT, CUM, NITER, POLYS, NTYPE, EXPNUM
    

    BUDGET=in_iter[0]
    FEATURE = in_iter[1]
    cvit = in_iter[2]
    al_method = in_iter[3]
    method = 'maj'
    n_states = 0
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
            robot = play_ner(FEATURE, MODEL_VER, POLY, cvit, LOGIT, FCLS, method, n_states, al_method)
            tf.reset_default_graph()

                               
 
if __name__ == '__main__':
    #cvits = np.linspace(0,3,4).astype(int)
    cvits = [4]
    #### iterate over budget, al_method, cvit
    Parallel(n_jobs=num_cores)(delayed(main)(i) for i in itertools.product(BUDGETS,FEATURES,cvits, ALMETHODS))
    #main([5,[1,2,3,4],9,'rs'])                           