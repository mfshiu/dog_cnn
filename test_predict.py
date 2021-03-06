# -*- coding: utf-8 -*-
"""1214 Siamese Net dog CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LxnF-kw1mRuoblAKw0eFAAyzAR3qAUPF

# Spectral Net

### Importing the necessary Libraries
"""

#! gdown --id "1BrbfVxBBsCgDtlh-FW7N06fRdMwgWAkA" --output data.zip
#!unzip data.zip
# ! gdown --id "1vrJFkpEZu5CHp9i-EGM3JYYmqQORxdJT" --output data.zip
# !unzip data.zip

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch as to
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

"""## Training the Siamese Netwrok"""

path = "./狗鼻紋影像資料庫_segmented"

"""This function is used for sorting the img of the MNIST datset"""

def secondval( value ):
    
    return value[0]

"""##### The Data Set for Siamese Networks

Here we define the dataset class that will be used by dataloader for train-20 the siamese network
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
import warnings
warnings.simplefilter("ignore", UserWarning)

EPOCH = 10                #全部data訓練10次
BATCH_SIZE = 50           #每次訓練隨機丟50張圖像進去
LR =0.001                 #learning rate
DOWNLOAD_MNIST = True    #第一次用要先下載data,所以是True
if_use_gpu = 1            #使用gpu

img_size = 400
class SiamDataset(Dataset):
    
    def __init__(self, mode = "train-20"):
        
        # We import the MNIST dataset that is pre formatted and kept as a csv file 
        # in which each row contains a single image flattened out to 784 pixels
        # so each row contains 784 entries
        # after the import we reshape the image in the format required by pytorch i.e. (C,H,W)
        filenames = []
        labels = []
        img = []
        test_img = []
        teest_labels = []
        self.mode = mode
        
        

        for id in range(1,20):
          temp = []
          test_temp = []
          if id < 10:
            files= glob.glob(os.path.join(path,"dog0"+str(id)+"*/dog*/*.bmp"))
          else:
            files = glob.glob(os.path.join(path,"dog"+str(id)+"*/dog*/*.bmp"))
          for filename in files:
            filenames.append(filename)
            temp.append(filename)
            labels.append(id)
            #print(id, filename)
          if mode == "train-20":
            temp = temp[:18]
            labels = labels[:18]
            test_temp = temp[18:]
            test_labels = labels[18:]
          
          else:
            temp = temp[18:]
            labels = labels[18:]
          

          print(id, " length", len(temp))
          img.append(temp)
          test_img.append(test_temp)
        
        self.filenames = filenames
        self.labels = labels
        self.img = img
        self.test_img = test_img
        # train-20 時做 data augmentation
        self.transform = transforms.Compose([
            
            transforms.ToPILImage(),
            transforms.Compose([transforms.Scale((img_size,img_size))]),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(), # 隨機將圖片水平翻轉
            transforms.RandomRotation(30), # 隨機旋轉圖片
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0), ratio=(0.95, 1.3333333333333333), interpolation=2),

            transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
        ])
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Compose([transforms.Scale((img_size,img_size))]),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
        ])
    
    
    def __getitem__(self, idx):
        
        # this class is needed to be defined so that dataloader can be used
        # here instead of giving the real index values I have returned randomly generated img
        # so idx does not have any need but the function signature needs to be same so that the dataloader
        # can call this function
        
        # I create a positive pair with label of similarity 1
        
        clas = np.random.randint(1,19)
            
        length = len(self.img[clas])
        im1, im2 = np.random.randint(0,length,2)
        if self.mode =="testing":
          while im1 == im2:
             im2 = np.random.randint(0,length)

        
        
        if self.mode =="testing":
          img1 = self.test_transform(np.array(Image.open(self.img[clas][im1]).convert("RGB")))
          img2 = self.test_transform(np.array(Image.open(self.img[clas][im2]).convert("RGB")))
        else:
          img1 = self.transform(np.array(Image.open(self.img[clas][im1]).convert("RGB")))
          img2 = self.transform(np.array(Image.open(self.img[clas][im2]).convert("RGB")))
        
        img1 = to.tensor(np.reshape(img1,(3,img_size,img_size)), dtype=to.float32)
        img2 = to.tensor(np.reshape(img2,(3,img_size,img_size)), dtype=to.float32)
        y1 = to.tensor(np.ones(1,dtype=np.float32),dtype=to.float32)
                   
        
        # I create a negative pair with label of similarity 0
        
        len1 = len(self.img[clas])
        clas2 = np.random.randint(1,19)
        while clas2 == clas:
          clas2 = np.random.randint(1,19)
        
        len2 = len(self.img[clas2])
        
        im3 = np.random.randint(0,len1)
        im4 = np.random.randint(0,len2)
        
        #img3 = self.transform(np.array(Image.open(self.img[clas][im1]).convert("RGB")))
        #img4 = self.transform(np.array(Image.open(self.img[clas2][im4]).convert("RGB")))
        
        if self.mode =="testing":
          img3 = self.test_transform(np.array(Image.open(self.img[clas][im1]).convert("RGB")))
          img4 = self.test_transform(np.array(Image.open(self.img[clas2][im4]).convert("RGB")))
        else:
          img3 = self.transform(np.array(Image.open(self.img[clas][im1]).convert("RGB")))
          img4 = self.transform(np.array(Image.open(self.img[clas2][im4]).convert("RGB")))
          
        img3 = to.tensor(np.reshape(img3,(3,img_size,img_size)), dtype=to.float32)
        img4 = to.tensor(np.reshape(img4,(3,img_size,img_size)), dtype=to.float32)
        y2 = to.tensor(np.zeros(1,dtype=np.float32),dtype=to.float32)
        #print(y1, y2)
        return  img1, img2, y1, img3, img4, y2, clas,clas2
            
    def __len__(self):
        
        # here I gave a smaller length than the real dataset's length so that the train-20 can be faster
            
        return 200

"""##### The Model Definition of Siamese Network

Here unlike as stated in the paper I have used a single network and trained the dataset. This can be done as both the layers are completely tied even in the train-20.
"""

class Siamese(nn.Module):
    
    def __init__(self):
        super(Siamese,self).__init__()
        
        # A simple two layer convolution followed by three fully connected layers should do
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        
    def forward_once(self,x):
        out = self.cnn(x)
        #print(out.shape)
        out = out.view(-1, 256)
        #print(out.shape)
        return self.fc(out)
    
    def forward(self, x, y):    
        
        # doing the forwarding twice so as to obtain the same functions as that of twin networks
        
        out1 = self.forward_once(x)
        out2 = self.forward_once(y)
        
        return out1, out2
    
    
    def evaluate(self, x, y):
        
        # this can be used later for evalutation
        
        m = to.tensor(1.0, dtype=to.float32).cuda()
        
        if type(m) != type(x):
            x = to.tensor(x, dtype = to.float32, requires_grad = False).cuda()
            
        if type(m) != type(y):
            y = to.tensor(y, dtype = to.float32, requires_grad = False).cuda()
        
        x = x.view(-1,3,img_size,img_size)
        y = y.view(-1,3,img_size,img_size)
        
        with to.no_grad():
            
            out1, out2 = self.forward(x, y)
            
            return nn.functional.pairwise_distance(out1, out2)


siamdset = SiamDataset()

siam_test = Siamese().cuda()
siam_test.load_state_dict(torch.load("./Siamese model"))
siam_test.eval()

"""## Testing the model's prediction"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

trial = list()
siamdset = SiamDataset(mode= "testing")
for i in range(0,20):
    trial.append(siamdset[i])

threshold = 1
fig = plt.figure(1, figsize=(30,100))

i = 1 

for data in trial :
    

    im1, im2, lb1, im3, im4, lb2,C1,C2 = data

    diss1 = siam_test.evaluate(im1.cuda(),im2.cuda())
    diss2 = siam_test.evaluate(im3.cuda(),im4.cuda())
    
    im1 = np.concatenate((im1.numpy()[0],im2.numpy()[0]),axis=1)
    lb1 = lb1.numpy()
    
    im2 = np.concatenate((im3.numpy()[0],im4.numpy()[0]),axis=1)
    lb2 = lb2.numpy()
    
    diss1 = diss1.cpu().numpy().mean()
    diss2 = diss2.cpu().numpy().mean()

    acceptance = False ##接受與否

    ax1 = fig.add_subplot(40,4,i)
    
    
    ax1.title.set_text("label = "+str(lb1[0])+"\n"+"distance = "+str(diss1)+ " \n result = "+ str(diss1 < threshold))
    ax1.imshow(im1,cmap="gist_stern")
    
    ax2 = fig.add_subplot(40,4,i+1)
    ax2.title.set_text("label = "+str(lb2[0])+"\n"+"distance = "+str(diss2)+ " \n result = "+ str(diss2 < threshold))
    ax2.imshow(im2,cmap="gist_stern")

    i+=8

