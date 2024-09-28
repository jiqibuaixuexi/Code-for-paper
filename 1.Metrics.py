import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir) 
        
csv_path = 'models_and_experts_output.csv'

data = pd.read_csv(csv_path)

print(data)


for i in range(data.shape[0]):
    
    if data.loc[i,'P_0.1_SRC_MT_pred_label'] ==0 or data.loc[i,'Expert_1'] == 0:
        data.loc[i,'Expert_1_with_model'] = 0
    else:
        data.loc[i,'Expert_1_with_model'] = 1
    if data.loc[i,'P_0.1_SRC_MT_pred_label'] ==0 or data.loc[i,'Expert_2'] == 0:
        data.loc[i,'Expert_2_with_model'] = 0
    else:
        data.loc[i,'Expert_2_with_model'] = 1   

print(data)
data.to_csv(csv_path,index=False)


expert_1_label = data.loc[:,'Expert_1']
expert_2_label = data.loc[:,'Expert_2']
expert_1_with_model = data.loc[:,'Expert_1_with_model']
expert_2_with_model = data.loc[:,'Expert_2_with_model']

true_label = data.loc[:,'true_label']

expert_label_dict = {'expert_1_label':expert_1_label,'expert_2_label':expert_2_label,'expert_1_with_model':expert_1_with_model,'expert_2_with_model':expert_2_with_model}

for n,key in enumerate(expert_label_dict):
    print(f'current eval:{key}')
    pred_label = expert_label_dict[key]
    # print(pred)
    # print(true_label)
    
    # define 0 as AS(positive) and 1 as nonAS(negative)
    # tp(true_label=0,pred_label=0)
    tp = 0
    for i,item in enumerate(true_label):
        if true_label[i] == 0 and pred_label[i] == 0:
            tp += 1
    print("tp:",tp) 

    # tn(true_label=1,pred_label=1)
    tn = 0
    for i,item in enumerate(true_label):
        if true_label[i] == 1 and pred_label[i] == 1:
            tn += 1
    print("tn:",tn) 

    # fp(true_label=1,pred_label=0)
    fp = 0
    for i,item in enumerate(true_label):
        if true_label[i] == 1 and pred_label[i] == 0:
            fp += 1
    print("fp:",fp) 

    # fn(true_label=0,pred_label=1)
    fn = 0
    for i,item in enumerate(true_label):
        if true_label[i] == 0 and pred_label[i] == 1:
            fn += 1
    print("fn:",fn) 
    
    # check if we are right
    print('tp,tn,fp,fn result:',tp+tn+fn+fp==len(true_label)) 

    # calculate ACC=(TP+TN)/(TP+FP+FN+TN)
    acc = (tp+tn)/(tp+fp+fn+tn)
    print("ACC:",acc) 

    # calculate TPR(true positive rate):TPR=TP/ (TP+ FN) (recall)
    tpr = tp/(tp+fn)
    print("TPR/RECALL:",tpr) 

    # calculate FPR(false positive rate):FPR= FP / (FP + TN)
    fpr = fp/(fp+tn)
    print("FPR:",fpr) 

    # calculate BER(balanced error rate):BER=1/2*(FPR+FN/(FN+TP))
    ber = 1/2*(fpr+fn/(fn+tp)) 
    print("BER:",ber) 

    # calculate TNR(true negative rate):TNR= TN / (FP + TN)
    tnr = tn/(fp+tn)
    print("TNR:",tnr) 

    # calculate PPV(positive predictive value):PPV=TP / (TP + FP) (precision)
    ppv = tp/(tp+fp)
    print("PPV/PRECISION:",ppv) 

    # calculate NPV(negative predictive value):NPV=TN / (FN + TN)
    npv = tn/(fn+tn)
    print("NPV:",npv) 

    # calculate Fβ score:Fβ score=(1+β**2)*（precision*recall）/((β**2*precision)+recall)
    # F1 score
    b=1
    B=b**2
    f1 = (1+B)*(ppv*tpr)/((B*ppv)+tpr)
    print("F1 score:",f1)
    # F2 score: weight of recall > precision
    b=2
    B=b**2
    f2 = (1+B)*(ppv*tpr)/((B*ppv)+tpr)
    print("F2 score:",f2) 
    # F0.5 score: weight of recall < precision
    b=0.5
    B=b**2
    f05 = (1+B)*(ppv*tpr)/((B*ppv)+tpr)
    print("F0.5 score:",f05) 

    # AUC:    
    y_true = np.array(true_label)
    y_scores = np.array(pred_label)
    auc = roc_auc_score(y_true, y_scores)
    print("AUC:",auc) 
    
    acc=np.around(acc,3)
    tpr=np.around(tpr,3)
    fpr=np.around(fpr,3)
    ber=np.around(ber,3)
    tnr=np.around(tnr,3)
    ppv=np.around(ppv,3)
    npv=np.around(npv,3)
    f1=np.around(f1,3)
    f2=np.around(f2,3)
    f05=np.around(f05,3)
    auc=np.around(auc,3)     
    # save metrics:
    df_metrics = pd.DataFrame()
    df_metrics['name'] = [key]
    df_metrics['ACC'] = [acc]
    df_metrics['TPR'] = [tpr]
    df_metrics['FPR'] = [fpr]
    df_metrics['BER'] = [ber]
    df_metrics['TNR'] = [tnr]
    df_metrics['PPV'] = [ppv]
    df_metrics['NPV'] = [npv]
    df_metrics['F1'] = [f1]
    df_metrics['F2'] = [f2]
    df_metrics['F0.5'] = [f05]
    df_metrics['AUC'] = [auc]
    df_metrics = df_metrics
    print(df_metrics)
    if n==0:
        final = df_metrics
    else:
        final = pd.concat([final,df_metrics],axis=0)
print(final)
final.to_csv('Experts_Metrics.csv',index=False)
    