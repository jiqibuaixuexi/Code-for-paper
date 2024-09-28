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
# import paddle.backends.cudnn as cudnn
from paddle.io import Dataset,DataLoader
# from paddle.vision.utils import make_grid
# model
from model.resnet_model import resnet50
from model.mobilevit import mobilevit_s
from model.poolformer_attention import poolformer_m48
from model.vip_model import vip_s14,vip_m7
from model.vit import VisionTransformer
from model.ConvNeXt import convnext_base
from model.swinT import swin_b

from utils import losses, ramps
# from utils.metrics import compute_AUCs
# from utils.metric_logger import MetricLogger
from dataloaders import dataset
from dataloaders.dataset import TwoStreamBatchSampler
from dataloaders.Augmentations import RandAugment
# from utils.util import get_timestamp
# from validation import epochVal, epochVal_metrics

def makedir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


parser = argparse.ArgumentParser()
# dataset
# parser.add_argument('--root_path', type=str, default='/research/pheng4/qdliu/Semi/dataset/skin/training_data/', help='dataset root dir')
# parser.add_argument('--csv_file_train', type=str, default='/research/pheng4/qdliu/Semi/dataset/skin/training.csv', help='training set csv file')
# parser.add_argument('--csv_file_val', type=str, default='/research/pheng4/qdliu/Semi/dataset/skin/validation.csv', help='validation set csv file')
# parser.add_argument('--csv_file_test', type=str, default='/research/pheng4/qdliu/Semi/dataset/skin/testing.csv', help='testing set csv file')

### experiment
parser.add_argument('--model_name', type=str,  default='resnet50', help='model name')
parser.add_argument('--exp', type=str,  default='model_choose', help='experiment method')
parser.add_argument('--transform', type=str,  default='None', help='transform method')
parser.add_argument('--label_rate', type=float, default=0.2, help='rate of labeled')
parser.add_argument('--drop_rate', type=float, default=0.2, help='dropout rate')
parser.add_argument('--teacher_transform', type=int, default=1, help='whether teacher need aug,defaolt:use aug same as student')

parser.add_argument('--save_epoch', type=int, default=1, help='save weight in epoch')
parser.add_argument('--epochs', type=int,  default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=32, help='number of labeled data per batch')
parser.add_argument('--ema_consistency', type=int, default=1, help='whether train baseline model') 
parser.add_argument('--base_lr', type=float,  default=1e-4, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use') 

### tune
parser.add_argument('--resume', type=str,  default=None, help='use ckpt for train')
parser.add_argument('--start_epoch', type=int,  default=0, help='start_epoch')
parser.add_argument('--global_step', type=int,  default=0, help='global_step')
### costs
parser.add_argument('--label_uncertainty', type=str,  default='U-Ones', help='label type')
parser.add_argument('--consistency_relation_weight', type=int,  default=1, help='consistency relation weight') # weight of SRC loss,if == 0: use MT
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=1, help='consistency') 
parser.add_argument('--consistency_rampup', type=float,  default=30, help='consistency_rampup')



args = parser.parse_args()


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


pretrain_weight_path={'mobilevit':'pretrain_weight/mobilevit_s.pdparams',
'poolformer':'pretrain_weight/poolformer_m48.pdparams',
'poolformer_attention':'pretrain_weight/poolformer_m48.pdparams',
'vip':'pretrain_weight/vip_m7.pdparams',
'vit':'pretrain_weight/vit_base_patch16_224.pdparams',
'convnext':'pretrain_weight/convnext_base_1k_224_ema.pdparams',
'swinT':'pretrain_weight/swin_base_patch4_window7_224.pdparams'}


networks = {
    'resnet50':resnet50(pretrained=True,num_classes=2,drop_rate=args.drop_rate),
    'resnet50_CBAM':resnet50(pretrained=True,num_classes=2,drop_rate=args.drop_rate,use_CBAM_Module=True),
    'vit':VisionTransformer(num_classes=2,dropout=args.drop_rate), 
    'poolformer':poolformer_m48(num_classes = 2,drop_rate=args.drop_rate,use_attention=False),
    'vip':vip_m7(num_classes = 2,drop_rate =args.drop_rate),
    'poolformer_attention':poolformer_m48(num_classes = 2,drop_rate=args.drop_rate,use_attention=True),
    'convnext':convnext_base(num_classes=2,drop_path_rate=args.drop_rate),
    'swinT':swin_b(class_dim=2,drop_rate=args.drop_rate),} # TODO


train_RandAugment =transforms.Compose([
        transforms.Resize(size=(224, 224)), 
        transforms.Normalize(mean=[127.5, 127.5, 127.5],std=[127.5, 127.5, 127.5],data_format='HWC'), 
        transforms.Transpose()])
train_RandAugment.transforms.insert(1,RandAugment(2,10))


if args.teacher_transform == 0: 
    train_transforms = {
        'None':dataset.Transform_student_input(transforms.Compose([
            transforms.Resize(size=(224, 224)), 
            transforms.Normalize(mean=[127.5, 127.5, 127.5],std=[127.5, 127.5, 127.5],data_format='HWC'), 
            transforms.Transpose()])),
        'Base':dataset.Transform_student_input(transforms.Compose([
            transforms.Resize(size=(224, 224)), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.Normalize(mean=[127.5, 127.5, 127.5],std=[127.5, 127.5, 127.5],data_format='HWC'), 
            transforms.Transpose()])),
        'RandAug':dataset.Transform_student_input(train_RandAugment), 
        
    }
else: 
    train_transforms = {
        'None':dataset.TransformTwice(transforms.Compose([
            transforms.Resize(size=(224, 224)), 
            transforms.Normalize(mean=[127.5, 127.5, 127.5],std=[127.5, 127.5, 127.5],data_format='HWC'), 
            transforms.Transpose()])),
        'Base':dataset.TransformTwice(transforms.Compose([
            transforms.Resize(size=(224, 224)), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.Normalize(mean=[127.5, 127.5, 127.5],std=[127.5, 127.5, 127.5],data_format='HWC'), 
            transforms.Transpose()])),
        'RandAug':dataset.TransformTwice(train_RandAugment), 
        
    }



device = paddle.device.get_device()
paddle.device.set_device(device) 
print('Device is ：',device)


if args.ema_consistency == 1:
    print('SRC-MT training...')
else:
    print('fully supervised training...')


batch_size = args.batch_size
base_lr = args.base_lr
labeled_bs = args.labeled_bs


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242

    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

# EMA
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param = ema_param * alpha + param * (1 - alpha)

if __name__=="__main__": 
    
    
    def create_model(net,ema=False):
        # Network definition
        model = net
        if ema:
            for param in model.parameters():
                param.detach() 
        return model
    
    net = networks[args.model_name] 
    
    if args.resume is not None:
        param = paddle.load(args.resume) 
        net.set_dict(param)
        print(f'Use checkpoint weight for {args.model_name}')
    else:
        if args.model_name in ['poolformer','vip','vit','poolformer_attention','convnext','swinT']:
            param = paddle.load(pretrain_weight_path[args.model_name]) 
            net.set_dict(param)
            print(f'Use pretrained weight for {args.model_name}')
    model = create_model(net)
    ema_model = create_model(net,ema=True)

    train_log_dir = f'log/{args.exp}/transform_{args.transform}/label_rate_{args.label_rate}/drop_rate_{args.drop_rate}/{args.model_name}/train'
    val_log_dir = f'log/{args.exp}/transform_{args.transform}/label_rate_{args.label_rate}/drop_rate_{args.drop_rate}/{args.model_name}/val'
    make_dir(train_log_dir)
    make_dir(val_log_dir)    

    vdl_train=LogWriter(train_log_dir) 
    vdl_val=LogWriter(val_log_dir) 


    optimizer = paddle.optimizer.Adam(learning_rate=args.base_lr, beta1=0.9, beta2=0.999, epsilon=1e-08, parameters=model.parameters(), weight_decay=5e-4, grad_clip=None,name=None, lazy_mode=False)

    # dataset  

    train_dataset = dataset.ASDataset(mode='train',transform=train_transforms[args.transform])  
    val_dataset =dataset.ASDataset(mode='val',transform=transforms.Compose([
        transforms.Resize(size=(224, 224)), 
        transforms.Normalize(mean=[127.5, 127.5, 127.5],std=[127.5, 127.5, 127.5],data_format='HWC'), 
        transforms.Transpose()]))

    labeled_idxs = list(range(int(len(train_dataset)*args.label_rate))) 
    unlabeled_idxs = list(range(int(len(train_dataset)*args.label_rate), len(train_dataset)))
    print('labeled_idxs num',len(labeled_idxs),'unlabeled_idxs num:',len(unlabeled_idxs)) 
    
    if len(unlabeled_idxs)>0:
        print('exist unlabeled data')       
    else:
        print('no unlabeled data')

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler,
                                num_workers=0)

    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0)

    model.train()

    loss_fn = losses.cross_entropy_loss()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    iter_num = args.global_step
    lr_ = base_lr
    
    train_step = 0
    val_step = 0
    for epoch in range(args.start_epoch, args.epochs): 
        # Train 
        print('Start train in epoch:',epoch+1)
        model.train() 
        train_loss_list = []
        train_acc_list = []     
        time1 = time.time()
        
        for i,[[image_batch,ema_image_batch],label_batch] in enumerate(train_dataloader):
            train_step += 1
            
            
            # noise1 = torch.clamp(torch.randn_like(image_batch) * 0.1, -0.1, 0.1)
            # noise2 = torch.clamp(torch.randn_like(ema_image_batch) * 0.1, -0.1, 0.1)
            
            inputs = image_batch #+ noise1
            ema_inputs = ema_image_batch #+ noise2 

            activations,outputs = model(inputs)
            with paddle.no_grad():
                ema_activations, ema_output = ema_model(ema_inputs)
            ## calculate the loss
            if len(unlabeled_idxs)>0:
                loss_classification = loss_fn(outputs[:labeled_bs], label_batch[:labeled_bs]) 
                loss = loss_classification            
                
                acc = paddle.metric.accuracy(outputs[:labeled_bs], label_batch[:labeled_bs].reshape([labeled_bs,1])) # 只有带标签数据能够计算acc
            else:
                loss_classification = loss_fn(outputs, label_batch) 
                loss = loss_classification            
                
                acc = paddle.metric.accuracy(outputs, label_batch.reshape([outputs.shape[0],1])) # 全标签训练时,整个batch计算acc
            
            ## MT loss (have no effect in the beginneing)
            if args.ema_consistency == 1:
                consistency_weight = get_current_consistency_weight(epoch)
                # mse loss or KL loss
                consistency_dist = paddle.sum(consistency_criterion(outputs, ema_output)) / batch_size #/ dataset.N_CLASSES
                consistency_loss = consistency_weight * consistency_dist  

                consistency_relation_dist = paddle.sum(losses.relation_mse_loss(activations, ema_activations)) / batch_size
                consistency_relation_loss = consistency_weight * consistency_relation_dist * args.consistency_relation_weight 
            else:
                consistency_loss = 0.0
                consistency_relation_loss = 0.0
                consistency_weight = 0.0
                consistency_dist = 0.0
                #+ consistency_loss
            
            if (epoch > 20) and (args.ema_consistency == 1):
                loss = loss_classification + consistency_loss + consistency_relation_loss

            train_loss_list.append(loss)
            train_acc_list.append(acc)

            optimizer.clear_grad()
            loss.backward()
            if train_step % 10 == 0:
                vdl_train.add_scalar(tag="train/loss", step=train_step, value=loss)
                vdl_train.add_scalar(tag="train/acc", step=train_step, value=acc)
                

            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
        time2 = time.time()

        # calculate average acc and loss after one epoch
        train_average_loss = np.mean(train_loss_list)
        train_average_acc = np.mean(train_acc_list)
        train_time_one_epoch = str(int(time2-time1))+' s'
        print("train_epoch: {},  loss is: {}, acc is: {}, time cost is: {}".format(epoch+1, train_average_loss, train_average_acc, train_time_one_epoch))
        

        # eval stage 
        print('Start eval in epoch:', epoch+1)
        model.eval()   
        time1 = time.time()
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
            
            ## calculate the acc
            # print('output shape',outputs.shape)
            acc = paddle.metric.accuracy(outputs, label_batch.reshape([outputs.shape[0],1])) 

            eval_loss_list.append(loss)
            eval_acc_list.append(acc)

            if val_step % 10 == 0:
                vdl_val.add_scalar(tag="val/loss", step=val_step, value=loss)
                vdl_val.add_scalar(tag="val/acc", step=val_step, value=acc)
        time2 = time.time()
        # calculate average acc and loss after one epoch
        eval_average_loss = np.mean(eval_loss_list)
        eval_average_acc = np.mean(eval_acc_list)
        eval_time_one_epoch = str(int(time2-time1))+' s'
        print("val_epoch: {},  loss is: {}, acc is: {}, time cost is: {}".format(epoch+1, eval_average_loss, eval_average_acc,eval_time_one_epoch))

        # save model
        save_path_epoch = f'Experiments_{args.exp}/transform_{args.transform}/label_rate_{args.label_rate}/drop_rate_{args.drop_rate}/{args.model_name}/{args.base_lr}/params_in_epoch_{epoch+1}.pdparams' # TODO：修改模型保存路径名称
        save_path_best = f'Experiments_{args.exp}/transform_{args.transform}/label_rate_{args.label_rate}/drop_rate_{args.drop_rate}/{args.model_name}/{args.base_lr}/params_best.pdparams' # TODO：修改模型保存路径名称

        # save model at 1,20,50,100 epoch：
        if args.save_epoch:
            if epoch+1==1 or epoch+1==20 or epoch+1==50 or epoch+1==100:
                print(f'######Save model in epoch:{epoch+1}######')
                paddle.save(model.state_dict(),save_path_epoch)
        # save best model 
        if epoch+1 == 1:
            max_acc = 0.1 # initial a small value
        if eval_average_acc > max_acc:
            max_acc = eval_average_acc
            max_acc_epoch = epoch + 1
            print(f'******Max eval acc is {max_acc}. Save best model in epoch:{epoch+1}******')
            paddle.save(model.state_dict(),save_path_best)
    print(f"The best {args.model_name} model is in epoch {max_acc_epoch}. The best eval acc is {max_acc}")

        













