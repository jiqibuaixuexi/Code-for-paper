import os
import sys
import shutil
import argparse
import logging
import time
import random
import numpy as np
import pandas as pd
from visualdl import LogWriter

import paddle

from paddle.vision import transforms
import paddle.nn.functional as F
from paddle.io import Dataset,DataLoader
# model
from model.resnet_model import resnet50
# from model.mobilevit import mobilevit_s
from model.poolformer_attention import poolformer_m48
from model.vip_model import vip_s14,vip_m7
from model.vit import VisionTransformer
from model.ConvNeXt import convnext_base
from model.swinT import swin_b

from utils import losses, ramps
from dataloaders import dataset
from dataloaders.dataset import TwoStreamBatchSampler

from sklearn.metrics import roc_auc_score

def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


parser = argparse.ArgumentParser()
# dataset
parser.add_argument('--txt_test', type=str, default='AS_test_list.txt', help='testing set txt file')

###################################################################################
# model weight
parser.add_argument('--model_weight_path', type=str,  default='Experiments_baseline_and_upperbound/transform_None/label_rate_1.0/drop_rate_0.2/resnet50/params_best.pdparams', help='model weight path')

### experiment
parser.add_argument('--model_name', type=str,  default='resnet50', help='model name')
parser.add_argument('--exp', type=str,  default='model_choose', help='experiment method')
### save path
parser.add_argument('--save_name', type=str, default='resnet50_1.0_label_baseline', help='model_save_name')
###################################################################################



parser.add_argument('--transform', type=str,  default='None', help='transform method')
parser.add_argument('--label_rate', type=float, default=0.2, help='rate of labeled')
parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
parser.add_argument('--ema_consistency', type=int, default=1, help='whether train baseline model') # Setting to other values will not employ semi-supervised training, and the model will be trained using the CE loss with labeled data only.

args = parser.parse_args()

networks = {
    'resnet50':resnet50(pretrained=False,num_classes=2,drop_rate=args.drop_rate),
    'resnet50_CBAM':resnet50(pretrained=False,num_classes=2,drop_rate=args.drop_rate,use_CBAM_Module=True),
    'vit':VisionTransformer(num_classes=2,dropout=args.drop_rate), 
    'poolformer':poolformer_m48(num_classes = 2,drop_rate=args.drop_rate,use_attention=False),
    'vip':vip_m7(num_classes = 2,drop_rate =args.drop_rate),
    'poolformer_attention':poolformer_m48(num_classes = 2,drop_rate=args.drop_rate,use_attention=True),
    'convnext':convnext_base(num_classes=2,drop_path_rate=args.drop_rate),
    'swinT':swin_b(class_dim=2,drop_rate=args.drop_rate),} # TODO


model_weight_path = args.model_weight_path



device = paddle.device.get_device()
paddle.device.set_device(device) 
print('Device is ：',device)

if __name__=="__main__":     
    # get model
    model = networks[args.model_name]
    # load weight
    param = paddle.load(model_weight_path)
    model.set_dict(param)
    print(f'Use trained weight for {args.model_name}')
    # dataset  
    test_dataset = dataset.TEST_Dataset(args.txt_test)
    val_dataloader = DataLoader(dataset=test_dataset, batch_size=1,
                                shuffle=False, num_workers=0)
    # loss function
    loss_fn = losses.cross_entropy_loss()

    # test
    val_step = 0
    print(f'Start test in {args.model_name} for weight {args.model_weight_path}')
    model.eval()   
    time1 = time.time()
    true_label = []
    pred_label = []
    
    eval_loss_list = []
    eval_acc_list = []
    for val_i,[image_batch,label_batch] in enumerate(val_dataloader):
        

        val_step += 1
        # time3 = time.time()
        inputs = image_batch
        with paddle.no_grad():
            activations,outputs = model(inputs)
        ## calculate the loss
        loss_classification = loss_fn(outputs, label_batch) 
        loss = loss_classification
        
        
        true_label.append(label_batch)
        pred_label.append(np.argmax(outputs,axis=1))

        ## calculate the acc
        # print('output shape',outputs.shape)
        acc = paddle.metric.accuracy(outputs, label_batch.reshape([outputs.shape[0],1])) 

        eval_loss_list.append(loss)
        eval_acc_list.append(acc)
    time2 = time.time()
    # loss,acc,time
    eval_average_loss = np.mean(eval_loss_list)
    eval_average_acc = np.mean(eval_acc_list)
    eval_time_one_epoch = str(int(time2-time1))+' s'

    # 0：AS
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

    # calculate AUC:ACC=(TP+TN)/(TP+FP+FN+TN)
    acc = (tp+tn)/(tp+fp+fn+tn)
    print('check acc:',acc == eval_average_acc) 
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
    # save path
    save_path = f'Test_output/{args.exp}/{args.save_name}/'
    make_dirs(save_path)
    # save true label and pred label
    df_true_pred = pd.DataFrame()
    df_true_pred['model_name'] = [args.save_name]
    for i,item in enumerate(np.array(true_label)):
        df_true_pred.loc[i,'true_label'] = int(item) 
    for i,item in enumerate(np.array(pred_label)):
        df_true_pred.loc[i,'pred_label'] = int(item)
    df_true_pred.to_csv(save_path+'True_and_Pred.csv',index=False) 

    # save metrics:
    df_metrics = pd.DataFrame()
    df_metrics['model_name'] = [args.save_name]
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
    df_metrics.to_csv(save_path+'Metrics.csv',index=False) 

    print("loss is: {}, acc is: {}, time cost is: {}".format(eval_average_loss, eval_average_acc,eval_time_one_epoch))


