import numpy as np
import torchvision.transforms as transforms
import torch as to
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import random
import sys
from model import Siamese
import glob


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)

        loss_contrastive = to.mean((1 - label) * to.pow(euclidean_distance, 2) +
                                   (label) * to.pow(to.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


img_size = 128
max_epochs = 50


class SiamDataset(Dataset):

    def __init__(self, mode="train"):

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
        path = "dataset/train"

        for id in range(1, 20):
            temp = []
            test_temp = []
            files = glob.glob(os.path.join(path, str(id + 100).zfill(4), "*.bmp"))
            for filename in files:
                filenames.append(filename)
                temp.append(filename)
                labels.append(id)
                # print(id, filename)
            if mode == "train":
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
            return 10
        return 500


output_path = "./output"
if not os.path.exists(output_path):
    os.makedirs(output_path)
trained_dir = './trained'
if not os.path.exists(trained_dir):
    os.makedirs(trained_dir)


if __name__ == '__main__':
    use_gpu = len(sys.argv) > 1 and "gpu" == sys.argv[1] and to.cuda.is_available()
    if len(sys.argv) > 2:
        max_epochs = int(sys.argv[2])
    print("Use GPU: %s, Epochs: %d" % (use_gpu, max_epochs))
        
    dataset_dir = "./dataset/train"

    if use_gpu:
        siam = Siamese().cuda()
    else:
        siam = Siamese().cpu()
    Criterion = ContrastiveLoss()
    Optimizer = to.optim.Adam(siam.parameters(), lr=0.01)

    loss_history = []
    siam.train()
    for epoch in range(max_epochs):
        loops = 0
        train_dataloader = DataLoader(SiamDataset(mode="train"), shuffle=True, batch_size=20, num_workers=15)
        test_dataloader = DataLoader(SiamDataset(mode="test"), shuffle=True, batch_size=1, num_workers=15)
        for data in train_dataloader:
            loops += 1
            print("\rEpoch %d/%d, training loops: %d" % (epoch+1, max_epochs, loops), end="")
            siam.train()
            img1, img2, label1, img3, img4, label2 = data

            Optimizer.zero_grad()

            if use_gpu:
                output1, output2 = siam(img1.cuda(), img2.cuda())
                output3, output4 = siam(img3.cuda(), img4.cuda())
                loss_pos = Criterion(output1, output2, label1.cuda())
                loss_neg = Criterion(output3, output4, label2.cuda())
            else:
                output1, output2 = siam(img1.cpu(), img2.cpu())
                output3, output4 = siam(img3.cpu(), img4.cpu())
                loss_pos = Criterion(output1, output2, label1.cpu())
                loss_neg = Criterion(output3, output4, label2.cpu())
            
            loss_contrastive = loss_pos + loss_neg
            loss_contrastive.backward()

            Optimizer.step()

        siam.eval()

        for data in test_dataloader:
            print("\rEpoch %d/%d, testing loops: %d " % (epoch, max_epochs, loops), end="")
            img1, img2, label1, img3, img4, label2 = data

            if use_gpu:
                output1, output2 = siam(img1.cuda(), img2.cuda())
                output3, output4 = siam(img3.cuda(), img4.cuda())
                loss_pos = Criterion(output1, output2, label1.cuda())
                loss_neg = Criterion(output3, output4, label2.cuda())
            else:
                output1, output2 = siam(img1.cpu(), img2.cpu())
                output3, output4 = siam(img3.cpu(), img4.cpu())
                loss_pos = Criterion(output1, output2, label1.cpu())
                loss_neg = Criterion(output3, output4, label2.cpu())
            
            val_loss_contrastive = loss_pos + loss_neg

        print("\rEpoch {}/{} Current loss {} Val loss {}\n".format(
            epoch, max_epochs, loss_contrastive.item(), val_loss_contrastive.item()))

        loss_history.append(loss_contrastive.item())

    plt.plot([x for x in range(max_epochs)], loss_history)
    plt.savefig(os.path.join(output_path, "loss.png"))

    model_path = os.path.join(trained_dir, "Siamese.pkl")
    to.save(siam.state_dict(), model_path)
    if use_gpu:
        siam_test = Siamese().cuda()
    else:
        siam_test = Siamese().cpu()
    siam_test.load_state_dict(to.load(model_path))
    siam_test.eval()
