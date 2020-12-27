# -*- coding: utf-8 -*-
"""1214 Siamese Net dog CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LxnF-kw1mRuoblAKw0eFAAyzAR3qAUPF

# Spectral Net

### Importing the necessary Libraries
"""

# ! gdown --id "1BrbfVxBBsCgDtlh-FW7N06fRdMwgWAkA" --output data.zip
# !unzip data.zip
# ! gdown --id "1vrJFkpEZu5CHp9i-EGM3JYYmqQORxdJT" --output data.zip
# !unzip data.zip

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch as to
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import Siamese
import os

"""## Training the Siamese Netwrok"""

path = "dataset/train-20"
output_path = "./output"
if not os.path.exists(output_path):
    os.makedirs(output_path)
trained_dir = './trained'
if not os.path.exists(trained_dir):
    os.makedirs(trained_dir)

"""This function is used for sorting the images of the MNIST datset"""


def secondval(value):
    return value[0]


"""##### The Data Set for Siamese Networks

Here we define the dataset class that will be used by dataloader for train the siamese network
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import glob
from PIL import Image
import warnings

warnings.simplefilter("ignore", UserWarning)

EPOCH = 10  # 全部data訓練10次
BATCH_SIZE = 50  # 每次訓練隨機丟50張圖像進去
LR = 0.001  # learning rate
DOWNLOAD_MNIST = True  # 第一次用要先下載data,所以是True
if_use_gpu = 1  # 使用gpu

img_size = 128


class SiamDataset(Dataset):

    def __init__(self, mode="train"):

        # We import the MNIST dataset that is pre formatted and kept as a csv file
        # in which each row contains a single image flattened out to 784 pixels
        # so each row contains 784 entries
        # after the import we reshape the image in the format required by pytorch i.e. (C,H,W)
        filenames = []
        labels = []
        img = []
        teest_labels = []
        self.mode = mode

        for id in range(0, 20):
            files = glob.glob(os.path.join(path, str(id).zfill(4), "*.*"))
            print(id, " length", len(files))
            img.append(files)

        self.filenames = filenames
        self.labels = labels
        self.img = img
        print("Total images: %d" % (len(self.img),))

        # train 時做 data augmentation
        self.transform = transforms.Compose([

            transforms.ToPILImage(),

            transforms.CenterCrop(400),
            # transforms.RandomHorizontalFlip(), # 隨機將圖片水平翻轉
            transforms.Compose([transforms.Scale((img_size, img_size))]),
            transforms.RandomRotation(50),  # 隨機旋轉圖片
            # transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0), ratio=(0.95, 1.3333333333333333), interpolation=2),

            transforms.ToTensor(),  # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
        ])
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(400),
            transforms.Compose([transforms.Scale((img_size, img_size))]),

            transforms.ToTensor(),  # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
        ])

    def __getitem__(self, idx):

        # this class is needed to be defined so that dataloader can be used
        # here instead of giving the real index values I have returned randomly generated images
        # so idx does not have any need but the function signature needs to be same so that the dataloader
        # can call this function

        # I create a positive pair with label of similarity 1

        clas = np.random.randint(0, 19)

        length = len(self.img[clas])
        im1, im2 = np.random.randint(0, length, 2)
        if self.mode == "testing":
            while im1 == im2:
                im2 = np.random.randint(0, length)
        # print(im1, im2)

        if self.mode == "testing":
            img1 = self.test_transform(np.array(Image.open(self.img[clas][im1]).convert("RGB")))
            img2 = self.test_transform(np.array(Image.open(self.img[clas][im2]).convert("RGB")))
        else:
            img1 = self.transform(np.array(Image.open(self.img[clas][im1]).convert("RGB")))
            img2 = self.transform(np.array(Image.open(self.img[clas][im2]).convert("RGB")))

        img1 = to.tensor(np.reshape(img1, (3, img_size, img_size)), dtype=to.float32)
        img2 = to.tensor(np.reshape(img2, (3, img_size, img_size)), dtype=to.float32)
        y1 = to.tensor(np.ones(1, dtype=np.float32), dtype=to.float32)

        # I create a negative pair with label of similarity 0

        len1 = len(self.img[clas])
        clas2 = np.random.randint(0, 19)
        while clas2 == clas:
            clas2 = np.random.randint(0, 19)

        len2 = len(self.img[clas2])

        im3 = np.random.randint(0, len1)
        im4 = np.random.randint(0, len2)

        # img3 = self.transform(np.array(Image.open(self.img[clas][im1]).convert("RGB")))
        # img4 = self.transform(np.array(Image.open(self.img[clas2][im4]).convert("RGB")))

        if self.mode == "testing":
            img3 = self.test_transform(np.array(Image.open(self.img[clas][im1]).convert("RGB")))
            img4 = self.test_transform(np.array(Image.open(self.img[clas2][im4]).convert("RGB")))
        else:
            img3 = self.transform(np.array(Image.open(self.img[clas][im1]).convert("RGB")))
            img4 = self.transform(np.array(Image.open(self.img[clas2][im4]).convert("RGB")))

        img3 = to.tensor(np.reshape(img3, (3, img_size, img_size)), dtype=to.float32)
        img4 = to.tensor(np.reshape(img4, (3, img_size, img_size)), dtype=to.float32)
        y2 = to.tensor(np.zeros(1, dtype=np.float32), dtype=to.float32)
        # print(y1, y2)
        return img1, img2, y1, img3, img4, y2, clas, clas2

    def __len__(self):

        # here I gave a smaller length than the real dataset's length so that the train can be faster
        if self.mode == "testing":
            return 20
        return 500


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)

        loss_contrastive = to.mean((1 - label) * to.pow(euclidean_distance, 2) +
                                   (label) * to.pow(to.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


"""## Training

Defining necessary parameters
"""
if __name__ == '__main__':

    siamdset = SiamDataset()

    train_dataloader = DataLoader(siamdset, shuffle=True, batch_size=20,
                                  num_workers=15)

    siam = Siamese().cuda()
    pretrained_model = "./trained/Siamese-old.pkl"
    print("Load from %s" % (pretrained_model, ))
    siam.load_state_dict(to.load(pretrained_model))

    testset = SiamDataset(mode="testing")

    test_dataloader = DataLoader(testset, shuffle=True, batch_size=1,
                                 num_workers=15)

    number_epochs = 200
    Criterion = ContrastiveLoss()
    Optimizer = to.optim.Adam(siam.parameters(), lr=0.01)

    counter = []
    loss_history = []
    iteration_number = 0
    start = 0

    start = 0
    siam.train()
    for epoch in range(start, start + number_epochs):
        for data in train_dataloader:
            # print(data)
            siam.train()
            img1, img2, label1, img3, img4, label2, c1, c2 = data

            Optimizer.zero_grad()

            # here we obtain the positive pairs' loss as well as the negative pairs' loss

            output1, output2 = siam(img1.cuda(), img2.cuda())
            output3, output4 = siam(img3.cuda(), img4.cuda())

            loss_pos = Criterion(output1, output2, label1.cuda())
            loss_neg = Criterion(output3, output4, label2.cuda())

            # the total loss is then computed and back propagated

            loss_contrastive = loss_pos + loss_neg

            loss_contrastive.backward()

            Optimizer.step()
        siam.eval()
        for data in test_dataloader:
            img1, img2, label1, img3, img4, label2, c1, c2 = data
            output1, output2 = siam(img1.cuda(), img2.cuda())
            output3, output4 = siam(img3.cuda(), img4.cuda())

            loss_pos = Criterion(output1, output2, label1.cuda())
            loss_neg = Criterion(output3, output4, label2.cuda())
            val_loss_contrastive = loss_pos + loss_neg

        # printing the train errors

        print("Epoch number {}\n  Current loss {} Val loss {}\n".format(epoch, loss_contrastive.item(),
                                                                        val_loss_contrastive.item()))
        counter.append(epoch + 100)
        loss_history.append(loss_contrastive.item())

    """##### Saving the model and plotting the error"""

    plt.plot(counter, loss_history)
    plt.savefig(os.path.join(output_path, "loss.png"))

    model_path = os.path.join(trained_dir, "Siamese.pkl")
    to.save(siam.state_dict(), model_path)
    siam_test = Siamese().cuda()
    siam_test.load_state_dict(to.load(model_path))
    siam_test.eval()

    """## Testing the model's prediction"""

    # Commented out IPython magic to ensure Python compatibility.
    # %matplotlib inline

    trial = list()
    siamdset = SiamDataset(mode="testing")
    siam.eval()
    siam_test = siam
    for i in range(0, 20):
        trial.append(siamdset[i])

    threshold = 1.1
    fig = plt.figure(1, figsize=(30, 100))
    plt.savefig("./output/trained.png")

    i = 1

    for data in trial:
        im1, im2, lb1, im3, im4, lb2, C1, C2 = data

        diss1 = siam_test.evaluate(im1.cuda(), im2.cuda())
        diss2 = siam_test.evaluate(im3.cuda(), im4.cuda())

        im1 = np.concatenate((im1.numpy()[0], im2.numpy()[0]), axis=1)
        lb1 = lb1.numpy()

        im2 = np.concatenate((im3.numpy()[0], im4.numpy()[0]), axis=1)
        lb2 = lb2.numpy()

        diss1 = diss1.cpu().numpy().mean()
        diss2 = diss2.cpu().numpy().mean()

        acceptance = False  ##接受與否

        ax1 = fig.add_subplot(40, 4, i)

        ax1.title.set_text(
            "label = " + str(lb1[0]) + "\n" + "distance = " + str(diss1) + " \n result = " + str(diss1 < threshold))
        ax1.imshow(im1, cmap="gist_stern")

        ax2 = fig.add_subplot(40, 4, i + 1)
        ax2.title.set_text(
            "label = " + str(lb2[0]) + "\n" + "distance = " + str(diss2) + " \n result = " + str(diss2 < threshold))
        ax2.imshow(im2, cmap="gist_stern")

        i += 8

    plt.savefig("./output/trial.png")
    plt.show()

