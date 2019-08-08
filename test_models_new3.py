import os
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle as pkl
from robotDQN_new3 import RobotLSTMQ
from tagger_new3 import Tagger
from game_ner_new2 import Env
from sklearn.metrics import f1_score
from collections import Counter
import csv
from joblib import Parallel, delayed

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
EXPNUM = 23
NTYPE  = 'd1qn' #'d3qn' #'d1qn', 'd2qn', 'd3qn', 'd4qn']

NITER = 10

num_cores = multiprocessing.cpu_count()
print(num_cores)

#BUDGETS = [5,10, 20, 50, 100]
BUDGETS = [5, 10, 20, 50, 100]
#NSTATES = [0, 32, 64, 128]
NSTATES = [0]
#BUDGETS = [100, 300]
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
    while ID_student < len(test_child):
        observation = test_game.get_frame(model_selected)
        action = robot.test_get_action(model_ver,observation, BUDGET)
        test_game.current_frame = test_game.current_frame + 1
        #if action[0] == 1:
        print('> Action', action)
        if action[1] == 1:
            test_game.count_action1 += 1
            
            test_game.make_query =True
            test_game.query()
        else:
            test_game.count_action0 += 1 
        ################ test mode A################
        if test_game.count_action1 == BUDGET or test_game.current_frame == len(test_game.train_y_all)-1:
            if len(model_selected) == 1:
                model_selected[0].train_mode_B(np.squeeze(test_game.queried_set_x,axis=1), test_game.queried_set_y,feature_now[0])
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
                for i in range(len(model_selected)):
                    model_selected[i].train_mode_B(queried_set_xx[:,:,indexs[i]:indexs[i+1]], test_game.queried_set_y, test_game.feature[i])
            write_test_csv(test_game, model_selected, ID_student, model_ver, cvit)
            print('ID ', ID_student)
            print('> Terminal this child<')
            ID_student += 1
            if ID_student < len(test_child):
                test_game = initialise_game(model_selected,BUDGET,NITER,FEATURE, test_child[ID_student], cvit)
                

def write_test_csv(test_game, model, student_ID, model_ver, cvit):
    csv_name = 'C_Q_{0}_content/test_{1}/'.format(EXPNUM, cvit)+str(model_ver)+'.csv'
    if not os.path.exists('C_Q_{0}_content/test_{1}'.format(EXPNUM, cvit) + os.sep):
            os.makedirs('C_Q_{0}_content/test_{1}'.format(EXPNUM, cvit) + os.sep)

    f = open(csv_name, "a")
    
    writer = csv.DictWriter(
        f, fieldnames=["student_ID", 
                       "accuracy_test_A", 
                       "f1_A",
                       "f1_test_A",
                       "accuracy_majority_test_A",
                       "f1_majority_test_A",
                       "conf_test_A",                       
                       "count_0_B",
                       "count_1_B",
                       "accuracy_test_B", 
                       "f1_B",
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
        accuracy_test_A.append(float("%.3f" % model[i].test(test_game.test_x_all[i],test_game.test_y_all)))
        f1_test_A.append(float("%.3f" % model[i].get_f1_score(test_game.test_x_all[i],test_game.test_y_all)))
        

        
        accuracy_test_B.append(float("%.3f" % model[i].test_B(test_game.test_x_all[i],test_game.test_y_all)))
        f1_test_B.append(float("%.3f" % model[i].get_f1_score_B(test_game.test_x_all[i],test_game.test_y_all)))
        
        
        if len(model) ==1:
            print(np.array(test_game.test_x_all[i]).shape)
            conf_test_A.append(float("%.3f" % np.mean(model[i].get_confidence(test_game.test_x_all[i]))))
            conf_test_B.append(float("%.3f" % np.mean(model[i].get_confidence_B(test_game.test_x_all[i]))))
            predd_test_A.append(model[i].get_predictions(np.squeeze(test_game.test_x_all[i],axis=0)))
            predd_test_B.append(model[i].get_predictions_B(np.squeeze(test_game.test_x_all[i],axis=0)))
        else:
            print(np.array(test_game.test_x_all[i]).shape)
            conf_test_A.append(float("%.3f" % np.mean(model[i].get_confidence(list(test_game.test_x_all[i])))))
            conf_test_B.append(float("%.3f" % np.mean(model[i].get_confidence_B(list(test_game.test_x_all[i])))))
            predd_test_A.append(model[i].get_predictions(test_game.test_x_all[i]))
            predd_test_B.append(model[i].get_predictions_B(test_game.test_x_all[i]))
    
    predd_test_A = np.array(predd_test_A).T.tolist()
    predd_test_B = np.array(predd_test_B).T.tolist()
    
    for i in range(len(predd_test_A)):
        preds_test_A = Counter(predd_test_A[i])
        preds_test_B = Counter(predd_test_B[i])
        y_majority_test_A.append(preds_test_A.most_common(1)[0][0])
        y_majority_test_B.append(preds_test_B.most_common(1)[0][0])

    f1_A = f1_score(test_game.train_y_all, y_majority_test_A, average=None)
    f1_B = f1_score(test_game.train_y_all, y_majority_test_B, average=None)

    f1_majority_test_A = f1_score(test_game.train_y_all, y_majority_test_A, average='macro')
    accuracy_majority_test_A = sum(np.equal(test_game.test_y_all, y_majority_test_A))/len(test_game.test_y_all)
    f1_majority_test_B = f1_score(test_game.test_y_all, y_majority_test_B, average='macro')
    accuracy_majority_test_B = sum(np.equal(test_game.test_y_all, y_majority_test_B))/len(test_game.test_y_all)

    
        
    writer.writerow({"student_ID": student_ID, 
                     "accuracy_test_A": accuracy_test_A,
                     "f1_A": f1_A,
                     "f1_test_A": f1_test_A,
                     "accuracy_majority_test_A": float("%.3f" % accuracy_majority_test_A),
                     "f1_majority_test_A": float("%.3f" % f1_majority_test_A),
                     "conf_test_A": conf_test_A,
                     "count_0_B": test_game.count_action0,
                     "count_1_B": test_game.count_action1,
                     "accuracy_test_B": accuracy_test_B,
                     "f1_B": f1_B,
                     "f1_test_B": f1_test_B,
                     "accuracy_majority_test_B": float("%.3f" % accuracy_majority_test_B),
                     "f1_majority_test_B": float("%.3f" % f1_majority_test_B),
                     "conf_test_B": conf_test_B})

    print('csv saved')
    f.close()
        
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
    cvits = np.linspace(2,9,8).astype(int)
    #cvits = [5,6,7,8,9]
    Parallel(n_jobs=num_cores)(delayed(main)(i) for i in itertools.product(BUDGETS,FEATURES,cvits, NSTATES))
    #main([5,[1,2,3,4],9,32])                           