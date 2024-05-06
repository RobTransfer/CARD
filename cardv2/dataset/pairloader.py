from __future__ import print_function

import os
import socket
import random
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
import pandas
import torch


class contrastivetrainloader():
    """
    CIFAR100Instance+Sample Dataset
    """
    def __init__(self, args,data ,k=4096, mode='exact', is_sample=False, percent=1.0):
    
        self.k = k
        self.mode = mode
        self.is_sample = is_sample
        self.num_classes = None
        self.n_data = None
        self.tensors = None
        keys = args.dataset.split("/")
        torch.manual_seed(args.seed)
        if "UNSW-NB15" in keys:
        # mapping = {6:0,0:1,1:2,2:3,4:4,5:5,7:6,8:7,9:8}
        ##['Analysis'1 'Backdoor'2 'DoS'3 'Fuzzers'4 'Generic'5 'Normal0''Reconnaissance'6 'Shellcode'7 'Worms'8]
            data1 = pandas.read_csv(data)
            
            data1 = data1.drop_duplicates()
            print("Train Data.shape: ",data1.shape)
            data3= data1.drop(data1[data1['attack_cat'] == 0].index)
            data2 = data1[data1['attack_cat'] == 0].sample(n=35000,replace=False)
            
            if args.how == 'm':
                # data1=pandas.concat([data3,data2],ignore_index=True,axis=0)
                smallest = data1['attack_cat'].value_counts().iloc[1]
                if self.is_sample:
                    try:
                        def draw(group):
                            return group.sample(int(smallest*percent),replace=True)
                        data1 = data1.groupby(data1.columns[-1],group_keys=False,).apply(draw)
                    except:
                        def draw(group):
                            return group.sample(smallest,replace=True)
                        data1 = data1.groupby(data1.columns[-1],group_keys=False,).apply(draw)
                x_train = data1.iloc[:,:-2].values
                # data['attack_cat'] = data['attack_cat'].map(mapping)
                y_train = data1.iloc[:,-1].values
                
            elif args.how == 'b':
                # data1=pandas.concat([data3,data2],ignore_index=True,axis=0)
                smallest = data1['label'].value_counts().iloc[1]
                if self.is_sample:
                    try:
                        def draw(group):
                            return group.sample(int(smallest*percent),replace=True)
                        data1 = data1.groupby(data1.columns[-2],group_keys=False,).apply(draw)
                    except:
                        def draw(group):
                            return group.sample(smallest,replace=True)
                        data1 = data1.groupby(data1.columns[-2],group_keys=False,).apply(draw)
                x_train = data1.iloc[:,:-2].values
                y_train = data1.iloc[:,-2].values
            print("Values for each class: ",data1['attack_cat'].value_counts())
            print("x_train.shape: ",x_train.shape)   
            self.num_classes = len(np.unique(y_train))
            self.n_data = x_train.shape[0]
            self.tensors = [torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long)]
            
        else:
            data1 = pandas.read_csv(data).drop_duplicates()
            smallest = data1['attack_type'].value_counts().iloc[1]
            if self.is_sample:
                try:
                    def draw(group):
                        return group.sample(int(smallest*percent),replace=True)
                    data1 = data1.groupby(data1.columns[-1],group_keys=False,).apply(draw)
                except:
                    def draw(group):
                        return group.sample(smallest,replace=True)
                    data1 = data1.groupby(data1.columns[-1],group_keys=False,).apply(draw)
            print(data1['attack_type'].value_counts())
            x_train = data1.iloc[:,:-1].values
            y_train = data1.iloc[:,-1].values
            print("x_train.shape: ",x_train.shape)
            
            self.num_classes = len(np.unique(y_train))
            self.n_data = x_train.shape[0]
            self.tensors = [torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long)]
            
        
        num_samples = self.tensors[0].shape[0]
        self.label = self.tensors[1]
        self.data = self.tensors[0]
        if self.is_sample: 
            
            self.class_sort = [[] for i in range(self.num_classes)]
            for i in range(num_samples):
                self.class_sort[self.label[i].item()].append(i)
            self.cls_negative = [[] for i in range(self.num_classes)]
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.class_sort[j])
            
            self.class_sort = [np.asarray(self.class_sort[i]) for i in range(self.num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(self.num_classes)]

            # if 0 < percent < 1:
            #     n = int(len(self.cls_negative[0]) * percent)
            #     self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
            #                         for i in range(self.num_classes)]
            self.class_sort = np.asarray(self.class_sort)
            self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        
        img, target = self.data[index], self.label[index]
        
        # imgt, targett = self.testdata[index], self.labelt[index]

        
        if not self.is_sample:
            # directly return
            return img, target
        else:
            pos_idx = index
            
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
    def __len__(self):
        return self.tensors[0].size(0)
    def getnandclass(self):
        
        return self.num_classes,self.n_data
        
        
        

def get_dataloaders_sample(args,data,testdata,batch_size=128, num_workers=8, k=4096, mode='exact',
                                    is_sample=False, percent=1.0):
    trainset = contrastivetrainloader(args=args,data=data,mode = mode,is_sample=is_sample,percent=percent,k=k)
    
    numofclass,n_data = trainset.getnandclass()
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,num_workers=num_workers)

    keys = args.dataset.split("/")
    if "UNSW-NB15" in keys:
        datatest = pandas.read_csv(testdata)
        
        datatest = datatest.drop_duplicates()
        
        
        if args.how == 'm':
            x_test =datatest.iloc[:,:-2].values
            # datatest['attack_cat'] = datatest['attack_cat'].map(mapping)
            y_test = datatest.iloc[:,-1].values
        elif args.how == 'b':
            x_test =datatest.iloc[:,:-2].values
            y_test = datatest.iloc[:,-2].values
        
       
        test = [torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long)]
        
        test_dataset = TensorDataset(test[0],test[1])

        test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    else:
    
        datatest = pandas.read_csv(testdata).drop_duplicates()
    
        x_test =datatest.iloc[:,:-1].values
        y_test = datatest.iloc[:,-1].values
        
    
        
        test = [torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long)]
        
        test_dataset = TensorDataset(test[0],test[1])
       
        test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    return train_loader, test_loader,trainset,numofclass, n_data
