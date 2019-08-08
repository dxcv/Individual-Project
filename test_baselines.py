# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:57:20 2019
UPDATED: Sat May 25
ADDED FUNCTIONS "conservative sampling", "Jaccord similarity"
@author: Beryl
"""

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle as pkl
from robot import RobotLSTMQ
from tagger import Tagger
from game_ner import Env
from sklearn.metrics import f1_score
from collections import Counter
from diversity import diversitySampling
import csv
import random

AGENT = "LSTMQ"
MAX_EPISODE = 100

##################################################################
# changable variables
BUDGETS = [5, 10, 20, 50, 100]
#BUDGETS = [100, 300]
NITERS = [10] # number of epochs 
POLYS = [False]
CONTENTS = [False]
CUMS = [False] #cumulative
FEATURES = [[0],[1],[2],[3],[4],[1,2,3,4]] # WHAT ABOUT THE OTHER FEATURE COMBINATIONS?
#FEATURES = [[0],[1]] # comp 1 (this)
#FEATURES = [[2],[3]]
#FEATURES = [[1,2,3,4]]
#FEATURES = [[4]]
################################################################
#feature shape
FEATURE_SHAPE = [[378,10],[257,10],[70,10],[27,10],[24,10]]
#FEATURE = 'ALL','FACE', 'BODY', 'PHY', 'AUDIO']
SAMPLE_METHOD = ['rs','us','ds','cs','lc']
VOTE_METHOD = ['maj', 'wei', 'conf']

###############################################################################
############ function returns the data in shape of N*n_step*n_input############
def test_data(tagger, features_selected, student):
    print('loading data')
    field_inds = {'FACE': (0, 257), 
                'BODY': (257, 327),
                'PHY': (327, 354),
                'AUDIO': (354, 378),
                'CARS': (378, 393)} # we dont use this
    
    id_offset = 6
    fields = ('FACE', 'BODY', 'PHY', 'AUDIO', 'CARS')

    
    print(student)
    with open(student, 'rb') as f:
        raw = pkl.load(f)
        data_test = pd.DataFrame(raw)
        data_test.sort_values(5, ascending=True, inplace=True)
        data_test = np.array(data_test)

    raw_data_test = []
    label_test = []
    
    uniques = np.unique(data_test[:,3])
    df = pd.DataFrame(data_test)
    for unique in uniques:
        df_spec = df.loc[df[3] == unique]
        array_spec = np.array(df_spec)
        window_index = 0
        while window_index < len(array_spec)-30:
            sum_eng = 0
            if array_spec[window_index+30][5] - array_spec[window_index][5] == 30:
                for frame_raw in range(window_index, window_index+30, 3):
                    raw_data_test.append(array_spec[frame_raw])
                    sum_eng = sum_eng + array_spec[frame_raw][-1]
                #assign label based on 30 frames 
                label_indicator = sum_eng/10.0
                if label_indicator < 0.5:
                    label_test.append(0)
                elif label_indicator >= 0.5 and label_indicator < 0.8:
                    label_test.append(1)
                else:
                    label_test.append(2)
                                        
                window_index = window_index + 5
            else:
                window_index += 1

    raw_data_test = np.array(raw_data_test).reshape((len(raw_data_test),402))

    #shape of 384
    test_feature_total = []
    test_feature_total.append(np.array(raw_data_test[:,6:378+6]).reshape(-1,10,378))
    for field in fields:
        start_col, end_col = field_inds[field]
        test_feature_total.append(np.array(raw_data_test[:, start_col + id_offset:end_col + id_offset]).reshape(-1,10,end_col-start_col))
    if len(features_selected) == 1:
        return [list(test_feature_total[features_selected[0]])], list(label_test)
    else:
        return list(test_feature_total[1:-1]), label_test
###############################################################################



########################### RANDOM SAMPLING ###################################
def random_sampling(feature_now, model_ver, budget_test, cvit):    
    test_child = []
    
    with open('test_child.txt') as f:
        for line in f.readlines():
            l = line.split()[0]
            test_child.append(l)

        #student is the individual test kid
    print(">>>>>> Playing game ..")
    model_selected = []
    for i in feature_now:
        with tf.name_scope(model_ver+'/feature_{0}'.format(i)):
            model = Tagger(model_file=model_ver+'/feature_{0}'.format(i),
                                n_input=FEATURE_SHAPE[i][0],n_steps=FEATURE_SHAPE[i][1],feature_number=i)
        model_selected.append(model)
        
    ID_student = 0
    # once ID_student > 14, break_loop
    while ID_student < len(test_child):
        train_x_all, train_y_all = test_data(model_selected, feature_now, test_child[ID_student])
        test_x_all, test_y_all = train_x_all, train_y_all


        sample_N = min(budget_test*4,len(train_y_all))
        order = np.linspace(0, sample_N-1, sample_N)
        order = order.astype(int)
        
        budget_test = min(budget_test,len(order))
        random.seed(0)
        random.shuffle(order)
        queried_indexs = random.sample(list(order), budget_test)
        for i in range(len(model_selected)):
            model_selected[i].train_mode_B(np.array(train_x_all[i])[queried_indexs], np.array(train_y_all)[queried_indexs], feature_now[i])
        print("training of mode B finished")
        write_test_csv(model_selected, ID_student, model_ver, cvit, test_x_all, test_y_all)
        ID_student = ID_student+1
###############################################################################
        
        

########################## UNCERTAINTY SAMPLING ##############################
def uncertainty_sampling(feature_now, model_ver, budget_test, cvit):
    test_child = []
    with open('test_child.txt') as f:
        for line in f.readlines():
            l = line.split()[0]
            test_child.append(l)

        #student is the individual test kid
    print(">>>>>> Playing game ..")
    model_selected = []
    for i in feature_now:
        with tf.name_scope(model_ver+'/feature_{0}'.format(i)):
            model = Tagger(model_file=model_ver+'/feature_{0}'.format(i),
                                n_input=FEATURE_SHAPE[i][0],n_steps=FEATURE_SHAPE[i][1],feature_number=i)
        model_selected.append(model)
        
    ID_student = 0
    # once ID_student > 14, break_loop
    while ID_student < len(test_child):
        train_x_all, train_y_all = test_data(model_selected, feature_now, test_child[ID_student])
        test_x_all, test_y_all = train_x_all, train_y_all
        
        sample_N = min(budget_test*4,len(train_y_all))
        
        N = len(train_y_all)
        budget_test = min(budget_test,N)
        uncertainty = np.zeros((sample_N,))
        ones = np.ones((sample_N,))
        for i in range(len(model_selected)):
            uncertainty = uncertainty + (ones - model_selected[i].get_confidence(list(train_x_all[i][:sample_N])))
        queried_indexs = sorted(range(len(uncertainty)), key=lambda i: uncertainty[i])[-budget_test:]

        print(queried_indexs)
        for i in range(len(model_selected)):
            model_selected[i].train_mode_B(np.array(train_x_all[i])[queried_indexs], np.array(train_y_all)[queried_indexs], feature_now[i])
        print("training of mode B finished")
        write_test_csv(model_selected, ID_student, model_ver, cvit, test_x_all, test_y_all)
        ID_student = ID_student+1
############################################################################### 

       

######################### COSERVATIVE SAMPLING ################################
def conservative_sampling(feature_now, model_ver, budget_test, cvit):    
    test_child = []
    with open('test_child.txt') as f:
        for line in f.readlines():
            l = line.split()[0]
            test_child.append(l)

        #student is the individual test kid
    print(">>>>>> Playing game ..")
    model_selected = []
    for i in feature_now:
        with tf.name_scope(model_ver+'/feature_{0}'.format(i)):
            model = Tagger(model_file=model_ver+'/feature_{0}'.format(i),
                                n_input=FEATURE_SHAPE[i][0],n_steps=FEATURE_SHAPE[i][1],feature_number=i)
        model_selected.append(model)
        
    ID_student = 0
    # once ID_student > 14, break_loop
    while ID_student < len(test_child):
        train_x_all, train_y_all = test_data(model_selected, feature_now, test_child[ID_student])
        test_x_all, test_y_all = train_x_all, train_y_all
        
        sample_N = min(budget_test*4,len(train_y_all))
        
        N = len(train_y_all)
        budget_test = min(budget_test,N)
        confidence = []
        conf_diff = np.zeros((sample_N,))
        
        for i in range(len(model_selected)):
            confidence.append(model_selected[i].get_confidence(list(train_x_all[i][:sample_N])))
        # the max indecies
        ind_max = np.argmax(confidence, axis=0)
        # the min indecies
        ind_min = np.argmin(confidence, axis=0)
        for i in range(sample_N):
            conf_diff[i] = confidence[ind_max[i]][i]-confidence[ind_min[i]][i]
        queried_indexs = sorted(range(len(conf_diff)), key=lambda i: conf_diff[i])[:budget_test]

        for i in range(len(model_selected)):
            model_selected[i].train_mode_B(np.array(train_x_all[i])[queried_indexs], np.array(train_y_all)[queried_indexs], feature_now[i])
        print("training of mode B finished")
        write_test_csv(model_selected, ID_student, model_ver, cvit, test_x_all, test_y_all)
        ID_student = ID_student+1
###############################################################################
        
        
        
########################### DIVERSITY SAMPLING ################################
def diversity_sampling(feature_now, model_ver, budget_test, cvit):
    test_child = []
    
    with open('test_child.txt') as f:
        for line in f.readlines():
            l = line.split()[0]
            test_child.append(l)

        #student is the individual test kid
    print(">>>>>> Playing game ..")
    model_selected = []
    for i in feature_now:
        with tf.name_scope(model_ver+'/feature_{0}'.format(i)):
            model = Tagger(model_file=model_ver+'/feature_{0}'.format(i),
                                n_input=FEATURE_SHAPE[i][0],n_steps=FEATURE_SHAPE[i][1],feature_number=i)
        model_selected.append(model)
        
    ID_student = 0
    while ID_student < len(test_child):
        train_x_all, train_y_all = test_data(model_selected, feature_now, test_child[ID_student])
        test_x_all, test_y_all = train_x_all, train_y_all

        sample_N = min(budget_test*4,len(train_y_all))
        order = np.linspace(0, sample_N-1, sample_N)
        order = order.astype(int)
        
        budget_test = min(budget_test,len(order))
        s = diversitySampling(train_x_all[:,:sample_N], pool = [], budget = budget_test)
        s.updateCplus()
        queried_indexs = s.newind
        for i in range(len(model_selected)):
            model_selected[i].train_mode_B(np.array(train_x_all[i])[queried_indexs], np.array(train_y_all)[queried_indexs], feature_now[i])
        print("training of mode B finished")
        write_test_csv(model_selected, ID_student, model_ver, cvit, test_x_all, test_y_all)
        ID_student = ID_student+1

###############################################################################
        
        

############################# LEAST CONFIDENT #################################
        
        
        
        
        
###############################################################################
        
        
        
######################### FUNCTION TO WRITE CSV ###############################
def write_test_csv(model, student_ID, model_ver, cvit, test_x_all, test_y_all):
    csv_header = 'D_test_rnd/test_{0}/'
    csv_name = csv_header.format(cvit)+str(model_ver)+'.csv'
    if not os.path.exists(csv_header.format(cvit) + os.sep):
            os.makedirs(csv_header.format(cvit) + os.sep)

    f = open(csv_name, "a")
    
    writer = csv.DictWriter(
        f, fieldnames=["student_ID", 
                       "accuracy_test_A", 
                       "f1_test_A",
                       "accuracy_majority_test_A",
                       "f1_majority_test_A",
                       "conf_test_A",                       
                       "accuracy_test_B", 
                       "f1_test_B",
                       "accuracy_majority_test_B",
                       "f1_majority_test_B",
                       "conf_test_B"])
    
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
        accuracy_test_A.append(float("%.3f" % model[i].test(test_x_all[i],test_y_all)))
        f1_test_A.append(float("%.3f" % model[i].get_f1_score(test_x_all[i],test_y_all)))
 
        accuracy_test_B.append(float("%.3f" % model[i].test_B(test_x_all[i],test_y_all)))
        f1_test_B.append(float("%.3f" % model[i].get_f1_score_B(test_x_all[i],test_y_all)))
        
        
        if len(model) ==1:
            conf_test_A.append(float("%.3f" % np.mean(model[i].get_confidence(test_x_all[i]))))
            conf_test_B.append(float("%.3f" % np.mean(model[i].get_confidence_B(test_x_all[i]))))
            predd_test_A.append(model[i].get_predictions(np.squeeze(test_x_all[i],axis=0)))
            predd_test_B.append(model[i].get_predictions_B(np.squeeze(test_x_all[i],axis=0)))
        else:
            conf_test_A.append(float("%.3f" % np.mean(model[i].get_confidence(list(test_x_all[i])))))
            conf_test_B.append(float("%.3f" % np.mean(model[i].get_confidence_B(list(test_x_all[i])))))
            predd_test_A.append(model[i].get_predictions(test_x_all[i]))
            predd_test_B.append(model[i].get_predictions_B(test_x_all[i]))
    
    predd_test_A = np.array(predd_test_A).T.tolist()
    predd_test_B = np.array(predd_test_B).T.tolist()
    
    for i in range(len(predd_test_A)):
        preds_test_A = Counter(predd_test_A[i])
        preds_test_B = Counter(predd_test_B[i])
        y_majority_test_A.append(preds_test_A.most_common(1)[0][0])
        y_majority_test_B.append(preds_test_B.most_common(1)[0][0])

    f1_majority_test_A = f1_score(test_y_all, y_majority_test_A, average='macro')
    accuracy_majority_test_A = sum(np.equal(test_y_all, y_majority_test_A))/len(test_y_all)
    f1_majority_test_B = f1_score(test_y_all, y_majority_test_B, average='macro')
    accuracy_majority_test_B = sum(np.equal(test_y_all, y_majority_test_B))/len(test_y_all)

    
        
    writer.writerow({"student_ID": student_ID, 
                     "accuracy_test_A": accuracy_test_A,
                     "f1_test_A": f1_test_A,
                     "accuracy_majority_test_A": float("%.3f" % accuracy_majority_test_A),
                     "f1_majority_test_A": float("%.3f" % f1_majority_test_A),
                     "conf_test_A": conf_test_A,
                     "accuracy_test_B": accuracy_test_B,
                     "f1_test_B": f1_test_B,
                     "accuracy_majority_test_B": float("%.3f" % accuracy_majority_test_B),
                     "f1_majority_test_B": float("%.3f" % f1_majority_test_B),
                     "conf_test_B": conf_test_B})

    print('csv saved')
    f.close()
###############################################################################
    
    
    
def main(cvit):
    global AGENT, MAX_EPISODE, BUDGET, MODEL_VER, FEATURE, FEATURE_SHAPE, CONTENT, CUM, NITER, POLYS
    
    for budget in BUDGETS:
        BUDGET=budget
        for niter in NITERS:
            NITER=niter
            for content in CONTENTS:
                CONTENT = content
                for cum in CUMS:
                    CUM = cum
                    for feature in FEATURES:
                        FEATURE = feature
                        MODEL_VER_0 = 'model_it_{0}_budget_{1}_content_{2}_cum_{3}'.format(NITER, BUDGET, int(CONTENT), int(CUM))

                        s=[0,0,0,0]
                        fvar = '_feature'
                        for i in range(np.shape(FEATURE)[0]):
                            if FEATURE[i]:
                                s[FEATURE[i]-1]=1
                        for i in range(np.shape(s)[0]):
                            fvar = fvar+'_{0}'.format(s[i])

                        MODEL_VER_0 = MODEL_VER_0 +str(fvar)


                        if CONTENT: 
                            POLY=False
                            #The same model_ver
                            ####################### test mode A #############################
                            MODEL_VER = MODEL_VER_0 + '_poly_{0}'.format(int(POLY))
                            print('test on model ', MODEL_VER)
                            random_sampling(feature, MODEL_VER, BUDGET, cvit)
                            #uncertainty_sampling(feature, MODEL_VER, BUDGET, cvit)
                            tf.reset_default_graph()
                        else:
                            for poly in POLYS:
                                POLY = poly
                                MODEL_VER = MODEL_VER_0 + '_poly_{0}'.format(int(POLY))
                                print('test on model', MODEL_VER)
                                random_sampling(feature, MODEL_VER, BUDGET, cvit)
                                #uncertainty_sampling(feature, MODEL_VER, BUDGET, cvit)
                                tf.reset_default_graph()
                                
 
if __name__ == '__main__':
	[main(i) for i in range(10)] 
                               