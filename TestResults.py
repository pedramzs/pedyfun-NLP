"""
@author: Pedram ZS
"""


""" Libraries """
import pandas as pd
import numpy as np
import os.path


""" Configs """
models_names = ['bert-base-uncased', 'albert-base-v1', 'distilbert-base-uncased', 'roberta-base']

""" Functions """
def score_task_1(truth_loc, prediction_loc):
    truth = pd.read_csv(truth_loc, usecols=['id','meanGrade'])
    pred = pd.read_csv(prediction_loc, usecols=['id','pred'])
    
    assert(sorted(truth.id) == sorted(pred.id)),"ID mismatch between ground truth and prediction!"
    
    data = pd.merge(truth,pred)
    rmse = np.sqrt(np.mean((data['meanGrade'] - data['pred'])**2))
    
    print("RMSE = %.3f" % rmse)
    
    return rmse
    

def score_task_2(truth_loc, prediction_loc):
    truth = pd.read_csv(truth_loc, usecols=['id','label'])
    pred = pd.read_csv(prediction_loc, usecols=['id','pred'])
    
    assert(sorted(truth.id) == sorted(pred.id)),"ID mismatch between ground truth and prediction!"
    
    data = pd.merge(truth,pred)
    data = data[data.label != 0]
    accuracy = np.sum(data.label == data.pred)*100/len(data)
    
    print("Accuracy = %.3f %%\n" % accuracy)
    
    return accuracy


Rmse = dict()
Acc = dict()

for model_name in models_names:
    
    print('model:', model_name)
    
    fname1 = model_name + "_task1"
    fname2 = model_name + "_task2"
    
    SAVE_PATH_task1 = os.path.join("results/" + fname1 + "_prediction.csv");
    SAVE_PATH_task2 = os.path.join("results/" + fname2 + "_test.csv");
    
    Rmse[model_name] = score_task_1("semeval-2020-task-7-data-full/task-1/test.csv", SAVE_PATH_task1)
    
    Acc[model_name] = score_task_2("semeval-2020-task-7-data-full/task-2/test.csv", SAVE_PATH_task2)
    

print('\nRMSE for Task 1 with the best model of each BERT-based word embedding model:\n')
for key in Rmse:
    print(key, ' : ', Rmse[key])

print('\nAccuracy for Task 2 with the best model of each BERT-based word embedding model:\n')
for key in Acc:
    print(key, ' : ', Acc[key])