
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
from model.poolformer_attention import poolformer_m48
from model.vip_model import vip_s14,vip_m7
from model.vit import VisionTransformer

from utils import losses, ramps
from dataloaders import dataset
from dataloaders.dataset import TwoStreamBatchSampler

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import seaborn as sns

def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

parser = argparse.ArgumentParser()
# dataset
parser.add_argument('--txt_test', type=str, default='Relation_Matrix_samples.txt', help='testing set txt file')

###################################################################################
# model weight
parser.add_argument('--model_weight_path', type=str,  default='checkpoints\poolformer_SRC_MT_0.1\params_best.pdparams', help='model weight path')

# ### information of file
parser.add_argument('--model_name', type=str,  default='poolformer', help='model name')
parser.add_argument('--figure_name', type=str,  default='poolformer_SRC_MT_0.1', help='result of which model')

###################################################################################

args = parser.parse_args()

networks = {
    'resnet50':resnet50(pretrained=False,num_classes=2),
    'resnet50_CBAM':resnet50(pretrained=False,num_classes=2,use_CBAM_Module=True),
    'vit':VisionTransformer(num_classes=2), 
    'poolformer':poolformer_m48(num_classes = 2,use_attention=False),
    'poolformer_attention':poolformer_m48(num_classes = 2,use_attention=True),
    'vip':vip_m7(num_classes = 2),
    }

model_weight_path = args.model_weight_path


device = paddle.device.get_device()
paddle.device.set_device(device) 
print('Device is ：',device)

if __name__=="__main__":     
    
    model = networks[args.model_name] 
    
    param = paddle.load(model_weight_path)
    model.set_dict(param)
    print(f'Use trained weight for {args.model_name}')
    # dataset  
    test_dataset = dataset.TEST_Dataset(args.txt_test)
    val_dataloader = DataLoader(dataset=test_dataset, batch_size=64,
                                shuffle=False, num_workers=0)
    
    # loss_fn = losses.cross_entropy_loss()

    
    val_step = 0
    print(f'Start test in {args.model_name} for weight {args.model_weight_path}')
    model.eval()   
    time1 = time.time()
    true_label = []
    pred_label = []
    
    eval_loss_list = []
    eval_acc_list = []
    
    for val_i,[image_batch,label_batch] in enumerate(val_dataloader):
        
        line = []
        val_step += 1
        # time3 = time.time()
        inputs = image_batch
        labels = label_batch
        with paddle.no_grad():
            activations,outputs = model(inputs) 
        activations = paddle.flatten(activations,1) # <64,num_features>
        similarity = activations.mm(activations.t()) # similarity:relation matrix
        print('shape of similarity matrix\n',similarity)
        

        sns.set()
        sim_norm = (similarity-similarity.min()) / (similarity.max()-similarity.min())
        plt.figure(figsize=(20,20)) 
        sns.heatmap(sim_norm.numpy(),cmap="RdBu_r",vmin=0, vmax=1,center=0,square=True, linewidths=.5,xticklabels=False,yticklabels=False)
        
        plt.title('Relation Matrix',size=20)

        plt.savefig(args.figure_name+'_Similarity_Matrix.svg')