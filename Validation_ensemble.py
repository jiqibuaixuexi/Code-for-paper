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
parser.add_argument('--positive', type=int, default=0, help='whether 1 or 0 is positive samples')
parser.add_argument('--use_softmax', type=int, default=1, help='whether use softmax for outputs')

###################################################################################
# model weight
parser.add_argument('--model1_weight_path', type=str,  default='Experiments_different_label_rate/transform_None/label_rate_0.1/drop_rate_0.2/poolformer/params_best.pdparams', help='model weight path')
parser.add_argument('--model2_weight_path', type=str,  default='Experiments_different_label_rate/transform_None/label_rate_0.1/drop_rate_0.2/convnext/0.0001/params_best.pdparams', help='model weight path')

### experiment
parser.add_argument('--model1_name', type=str,  default='poolformer', help='model name')
parser.add_argument('--model2_name', type=str,  default='convnext', help='model name')
parser.add_argument('--exp', type=str,  default='ensemble_model', help='experiment method')
### save path
parser.add_argument('--save_name', type=str, default='poolformer_and_convnext_use_softmax', help='model_save_name')
### AUC 
parser.add_argument('--auc_cal', type=str, default='pred_label',choices = ['pred_scores','pred_label'], help='How to calculate auc')
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


model1_weight_path = args.model1_weight_path
model2_weight_path = args.model2_weight_path


device = paddle.device.get_device()
paddle.device.set_device(device) 
print('Device is ：',device)

if __name__=="__main__":     
    
    # model1
    model1 = networks[args.model1_name] 
    
    param1 = paddle.load(model1_weight_path)
    model1.set_dict(param1)
    print(f'Use trained weight for {args.model1_name}')
    # model2
    model2 = networks[args.model2_name] 
    
    param2 = paddle.load(model2_weight_path)
    model2.set_dict(param2)
    print(f'Use trained weight for {args.model2_name}')
    
    
    # dataset  
    test_dataset = dataset.TEST_Dataset(args.txt_test)
    val_dataloader = DataLoader(dataset=test_dataset, batch_size=1,
                                shuffle=False, num_workers=0)
    
    loss_fn = losses.cross_entropy_loss()

    
    val_step = 0
    print(f'Start test in {args.model1_name} for weight {args.model1_weight_path}')
    print(f'Start test in {args.model2_name} for weight {args.model2_weight_path}')
    model1.eval()   
    model2.eval()
    
    time1 = time.time()
    true_label = []
    pred_label = []
    pred_scores = []
    
    eval_loss_list = []
    eval_acc_list = []
    for val_i,[image_batch,label_batch] in enumerate(val_dataloader):
        

        val_step += 1
        # time3 = time.time()
        inputs = image_batch
        with paddle.no_grad():
            activations1,outputs1 = model1(inputs)
            activations2,outputs2 = model2(inputs)
        ## calculate the loss
        ensemble_outputs = (outputs1+outputs2)/2
        loss_classification = loss_fn(ensemble_outputs, label_batch) 
        loss = loss_classification
        
        
        true_label.append(label_batch)
        if args.use_softmax==1:
            softmax_outputs1 = F.softmax(outputs1,axis=-1)
            # print("softmax_outputs1:",softmax_outputs1)
            softmax_outputs2 = F.softmax(outputs2,axis=-1)
            ensemble_softmax_outputs = (softmax_outputs1+softmax_outputs2)/2
            pred_label.append(np.argmax(ensemble_softmax_outputs,axis=-1))
            pred_scores.append(ensemble_softmax_outputs[0][1].item())
            outputs = ensemble_softmax_outputs
        else:                
            pred_label.append(np.argmax(ensemble_outputs,axis=1))
            pred_scores.append(F.softmax(ensemble_outputs,axis=-1)[0][1].item())
            outputs = ensemble_outputs
        # print(pred_scores)

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

    
    positive = args.positive 
    negative = 1-positive 

    # tp(true_label=positive,pred_label=positive)
    tp = 0
    for i,item in enumerate(true_label):
        if true_label[i] == positive and pred_label[i] == positive:
            tp += 1
    print("tp:",tp) 

    # tn(true_label=negative,pred_label=negative)
    tn = 0
    for i,item in enumerate(true_label):
        if true_label[i] == negative and pred_label[i] == negative:
            tn += 1
    print("tn:",tn) 

    # fp(true_label=negative,pred_label=positive)
    fp = 0
    for i,item in enumerate(true_label):
        if true_label[i] == negative and pred_label[i] == positive: 
            fp += 1
    print("fp:",fp) 

    # fn(true_label=positive,pred_label=negative)
    fn = 0
    for i,item in enumerate(true_label):
        if true_label[i] == positive and pred_label[i] == negative: 
            fn += 1
    print("fn:",fn) 
    
    # check if we are right
    print('tp,tn,fp,fn result:',tp+tn+fn+fp==len(true_label)) 

    # ACC:ACC=(TP+TN)/(TP+FP+FN+TN)
    acc = (tp+tn)/(tp+fp+fn+tn)
    print('check acc:',acc == eval_average_acc) 
    print("ACC:",acc) 

    # TPR(true positive rate):TPR=TP/ (TP+ FN) (recall)
    tpr = tp/(tp+fn)
    print("TPR/RECALL:",tpr) 

    # FPR(false positive rate):FPR= FP / (FP + TN)
    fpr = fp/(fp+tn)
    print("FPR:",fpr) 

    # BER(balanced error rate):BER=1/2*(FPR+FN/(FN+TP))
    ber = 1/2*(fpr+fn/(fn+tp)) 
    print("BER:",ber) 

    # TNR(true negative rate):TNR= TN / (FP + TN)
    tnr = tn/(fp+tn)
    print("TNR:",tnr) 

    # PPV(positive predictive value):PPV=TP / (TP + FP) (precision)
    ppv = tp/(tp+fp)
    print("PPV/PRECISION:",ppv) 

    # NPV(negative predictive value):NPV=TN / (FN + TN)
    npv = tn/(fn+tn)
    print("NPV:",npv) 

    # Fβ score:Fβ分数=(1+β**2)*（precision*recall）/((β**2*precision)+recall)
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
    
    if args.auc_cal == 'pred_label': ### Convert the output to 0s and 1s for computation.
        y_true = np.array(true_label)
        y_scores = np.array(pred_label)
        auc = roc_auc_score(y_true, y_scores)
        print("AUC from pred_label:",auc) 
     
    elif args.auc_cal == 'pred_scores':  ### Use the raw output passed through the softmax function for calculation.
        y_true = np.array(true_label)
        y_scores = np.array(pred_scores)
        auc = roc_auc_score(y_true, y_scores)
        print("AUC from pred_scores:",auc) 

    # Retain three decimal places.
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
    # save true label and pred label and pred score for CA
    df_true_pred = pd.DataFrame()
    df_true_pred['model_name'] = [args.save_name]
    for i,item in enumerate(np.array(true_label)):
        df_true_pred.loc[i,'true_label'] = int(item) 
    for i,item in enumerate(np.array(pred_label)):
        df_true_pred.loc[i,'pred_label'] = int(item)
    for i,item in enumerate(np.array(pred_scores)):
        df_true_pred.loc[i,'pred_scores'] = item   
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
