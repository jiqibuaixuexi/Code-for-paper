import os
import numpy as np
import paddle
import PIL
from paddle.io import Dataset,DataLoader,Sampler,BatchSampler
from paddle.vision.datasets import DatasetFolder, ImageFolder
from paddle.vision.transforms import Compose, Resize, Normalize, Transpose
from PIL import Image
import pandas as pd
import itertools

N_CLASSES = 2
CLASS_NAMES = [ 'AS', 'nonAS']


class MyDataset(Dataset): 
    """
    Step 1: Inherit the paddle.io.Dataset class.
    """
    def __init__(self, mode='train'):
        """
        Step 2: Implement the constructor, define the data reading method, and split the dataset into training and testing sets.
        """
        super(MyDataset, self).__init__()
        train_data = '/home/aistudio/train_dataset.txt' # Use a TXT file to define the dataset
        eval_data = '/home/aistudio/eval_dataset.txt' # Use a TXT file to define the dataset
        test_data = '/home/aistudio/test_label.txt'

        transform_train = Compose([Normalize(mean=[127.5, 127.5, 127.5],std=[127.5, 127.5, 127.5],data_format='HWC'), Transpose()]) # 大多数数据预处理是使用HWC格式的图片，而神经网络可能使用CHW模式输入张量。 输出的图片是numpy.ndarray的实例。Default: (2, 0, 1) <C,H,W>
        transform_eval = Compose([Normalize(mean=[127.5, 127.5, 127.5],std=[127.5, 127.5, 127.5],data_format='HWC'), Transpose()])
        
        with open(train_data,'r') as F_train:
            train_data = F_train.readlines()
        with open(eval_data,'r') as F_eval:
            eval_data = F_eval.readlines()
        with open(test_data,'r') as F_test:
            test_data = F_test.readlines()       

        self.mode = mode # Read different data based on different modes
        if self.mode  == 'train':
            self.data = train_data
            self.transform = transform_train
        elif self.mode  == 'eval':
            self.data = eval_data
            self.transform = transform_eval
        elif self.mode  == 'test':
            self.data = test_data
            self.transform = transform_eval


    def __getitem__(self, index):
        """
        Step 3: Implement the getitem method to define how to retrieve data for a specified index and return a single data instance (training data and its corresponding label).
        """
        data = self.data[index]
        img_path = data.split(' ')[0]
        img = Image.open(img_path)
        img = self.transform(img)
        
        

        # if self.mode  == 'test': # Please note that in some cases, testing datasets may not include labels, and it may be necessary to adapt the implementation to handle this situation accordingly.
        #     return img
        # else:
        label = data.split(' ')[1]
        label = float(label)            
        label = np.array([label]).astype('float32') # The label format should be in the form of [n].

        return img, label

    def __len__(self):
        """
        Step 4: Implement the len method to return the total number of instances in the dataset.
        """
        return len(self.data)

# define a AS dataset
class ASDataset(Dataset):
    def __init__(self, mode='train',transform=None):
        self.mode = mode
        # define transforms
        # transform_train = Compose([Resize(size=(224, 224)), Normalize(mean=[127.5, 127.5, 127.5],std=[127.5, 127.5, 127.5],data_format='HWC'), Transpose()]) # 大多数数据预处理是使用HWC格式的图片，而神经网络可能使用CHW模式输入张量。 输出的图片是numpy.ndarray的实例。Default: (2, 0, 1) <C,H,W>
        # transform_eval = Compose([Resize(size=(224, 224)), Normalize(mean=[127.5, 127.5, 127.5],std=[127.5, 127.5, 127.5],data_format='HWC'), Transpose()])
        self.transforms = transform
        train_txt_path = 'AS_train_train.txt'
        val_txt_path = 'AS_train_val.txt'

        with open(train_txt_path,'r') as F_train:
            train_data = F_train.readlines()
        with open(val_txt_path,'r') as F_val:
            val_data = F_val.readlines()

        if mode == 'train':
            self.data = train_data
            
        if mode == 'val':
            self.data = val_data                  

    def __getitem__(self, idx):
        img_path = self.data[idx].split(' ')[0]
        image = Image.open(img_path)
        image = image.convert('RGB') # Critical Step: Enhancing Code Robustness
        if self.transforms is not None:
            image = self.transforms(image)
        label = self.data[idx].split(' ')[1].strip()
        label = int(label) # label must be int
        # label = paddle.to_tensor(label)
        return image, label

    def __len__(self):
        return len(self.data)

# define a TEST dataset
class TEST_Dataset(Dataset):
    def __init__(self,test_txt_path = 'AS_test_list.txt'):        
        # define transforms
        # transform_train = Compose([Resize(size=(224, 224)), Normalize(mean=[127.5, 127.5, 127.5],std=[127.5, 127.5, 127.5],data_format='HWC'), Transpose()]) # 大多数数据预处理是使用HWC格式的图片，而神经网络可能使用CHW模式输入张量。 输出的图片是numpy.ndarray的实例。Default: (2, 0, 1) <C,H,W>
        self.transform_eval = Compose([Resize(size=(224, 224)), Normalize(mean=[127.5, 127.5, 127.5],std=[127.5, 127.5, 127.5],data_format='HWC'), Transpose()])
              
        val_txt_path = test_txt_path        
        with open(val_txt_path,'r') as F_val:
            val_data = F_val.readlines()       
            
        self.data = val_data                  

    def __getitem__(self, idx):
        img_path = self.data[idx].split(' ')[0]
        image = Image.open(img_path)
        image = image.convert('RGB') # Critical Step: Enhancing Code Robustness
        image = self.transform_eval(image)
        label = self.data[idx].split(' ')[1].strip()
        label = int(label) # label must be int
        # label = paddle.to_tensor(label)
        return image, label

    def __len__(self):
        return len(self.data)


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        # assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        if len(self.secondary_indices) >= self.secondary_batch_size > 0: # Sampling Method for Data with and without Labels
            primary_iter = iterate_once(self.primary_indices)
            secondary_iter = iterate_eternally(self.secondary_indices)
            return (
                primary_batch + secondary_batch
                for (primary_batch, secondary_batch)
                in zip(grouper(primary_iter, self.primary_batch_size),
                        grouper(secondary_iter, self.secondary_batch_size))
            )
        else: # Sampling Method for Fully Labeled Data
            primary_iter = iterate_once(self.primary_indices)            
            return (
                primary_batch
                for primary_batch
                in grouper(primary_iter, self.primary_batch_size)
            )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
