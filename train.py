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


class Siamese(nn.Module):

    def __init__(self, use_gpu):
        super(Siamese, self).__init__()
        self.use_gpu = use_gpu

        # A simple two layer convolution followed by three fully connected layers should do

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),

        )

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward_once(self, x):
        out = self.cnn(x)
        # print(out.shape)
        out = out.view(-1, 256)
        # print(out.shape)
        return self.fc(out)

    def forward(self, x, y):

        # doing the forwarding twice so as to obtain the same functions as that of twin networks

        out1 = self.forward_once(x)
        out2 = self.forward_once(y)

        return out1, out2

    def evaluate(self, x, y):
        if self.use_gpu:
            m = to.tensor(1.0, dtype=to.float32).cuda()
            if type(m) != type(x):
                x = to.tensor(x, dtype=to.float32, requires_grad=False).cuda()
            if type(m) != type(y):
                y = to.tensor(y, dtype=to.float32, requires_grad=False).cuda()
        else:
            m = to.tensor(1.0, dtype=to.float32).cpu()
            if type(m) != type(x):
                x = to.tensor(x, dtype=to.float32, requires_grad=False).cpu()
            if type(m) != type(y):
                y = to.tensor(y, dtype=to.float32, requires_grad=False).cpu()

        x = x.view(-1, 3, img_size, img_size)
        y = y.view(-1, 3, img_size, img_size)

        with to.no_grad():

            out1, out2 = self.forward(x, y)

            return nn.functional.pairwise_distance(out1, out2)


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
max_epochs = 5


class SiamDataset():
    class BaseDataset(Dataset):
        def __init__(self, img_pathss, size=0):
            self.img_pathss = img_pathss
            self.size = size
            if not self.size:
                self.size = len(self.img_pathss) * 2

        def __getitem__(self, idx):
            dog1, dog2 = random.sample(self.img_pathss, 2)

            dog1_1, dog1_2 = random.sample(dog1, 2)
            dog1_1_img = self.transform(np.array(Image.open(dog1_1).convert("RGB")))
            dog1_2_img = self.transform(np.array(Image.open(dog1_2).convert("RGB")))
            dog1_1_tensor = to.as_tensor(np.reshape(dog1_1_img, (3, img_size, img_size)), dtype=to.float32).clone().detach()
            dog1_2_tensor = to.as_tensor(np.reshape(dog1_2_img, (3, img_size, img_size)), dtype=to.float32).clone().detach()
            y1 = to.tensor(np.ones(1, dtype=np.float32), dtype=to.float32)

            dog2_1 = random.choice(dog2)
            dog2_1_img = self.transform(np.array(Image.open(dog2_1).convert("RGB")))
            dog2_1_tensor = to.as_tensor(np.reshape(dog2_1_img, (3, img_size, img_size)), dtype=to.float32).clone().detach()
            y2 = to.tensor(np.zeros(1, dtype=np.float32), dtype=to.float32)

            return dog1_1_tensor, dog1_2_tensor, y1, dog1_1_tensor, dog2_1_tensor, y2

        def __len__(self):
            return self.size

    class TrainDataset(BaseDataset):
        def __init__(self, img_pathss, size=0):
            super().__init__(img_pathss, size)

        def transform(self, imgs):
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(400),
                transforms.Compose([transforms.Resize((img_size, img_size))]),
                transforms.RandomRotation(50),
                transforms.ToTensor(),
            ])(imgs)

    class TestDataset(BaseDataset):
        def __init__(self, img_pathss, size=0):
            super().__init__(img_pathss, size)

        def transform(self, imgs):
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(400),
                transforms.Compose([transforms.Resize((img_size, img_size))]),
                transforms.ToTensor(),
            ])(imgs)

    def __init__(self, dataset_dir, test_ratio=0.1):
        self.test_ratio = test_ratio
        paths = [x for x in os.walk(dataset_dir)]
        self.img_pathss = []
        self.train_paths = None
        self.test_paths = None
        for path in paths[1:]:
            self.img_pathss.append([os.path.join(path[0], x) for x in path[2]])
        self.shuffle()

    def get_train_dataset(self):
        return SiamDataset.TrainDataset(self.train_paths)

    def get_test_dataset(self):
        return SiamDataset.TestDataset(self.test_paths)

    def shuffle(self):
        random.shuffle(self.img_pathss)
        test_size = int(len(self.img_pathss) * self.test_ratio)
        self.train_paths = self.img_pathss[test_size:]
        self.test_paths = self.img_pathss[:test_size]
        print("Train dogs: %d, Test dogs: %d" % (len(self.train_paths), len(self.test_paths)))


output_path = "./output"
if not os.path.exists(output_path):
    os.makedirs(output_path)
trained_dir = './trained'
if not os.path.exists(trained_dir):
    os.makedirs(trained_dir)


if __name__ == '__main__':
    use_gpu = len(sys.argv) > 1 and "gpu" == sys.argv[1] and to.cuda.is_available()
    print("Use GPU: %s" % (use_gpu,))
        
    dataset_dir = "./dataset/train"

    siam_dataset = SiamDataset(dataset_dir)
    train_dataloader = DataLoader(siam_dataset.get_train_dataset(), shuffle=True, batch_size=20, num_workers=15)
    test_dataloader = DataLoader(siam_dataset.get_test_dataset(), shuffle=True, batch_size=20, num_workers=15)

    if use_gpu:
        siam = Siamese(use_gpu).cuda()
    else:
        siam = Siamese(use_gpu).cpu()
    Criterion = ContrastiveLoss()
    Optimizer = to.optim.Adam(siam.parameters(),lr = 0.01 )

    loss_history = []
    siam.train()
    for epoch in range(max_epochs):
        loops = 0
        for data in train_dataloader:
            loops += 1
            print("\rEpoch %d/%d, training loops: %d" % (epoch, max_epochs, loops), end="")
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

        print()
        for data in test_dataloader:
            print("\rEpoch %d/%d, testing loops: %d" % (epoch, max_epochs, loops), end="")
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

        print("\nEpoch number {}/{}\n\tCurrent loss {} Val loss {}\n".format(
            epoch, max_epochs, loss_contrastive.item(), val_loss_contrastive.item()))

        loss_history.append(loss_contrastive.item())

    plt.plot([x + 100 for x in range(max_epochs)], loss_history)

    model_path = os.path.join(trained_dir, "Siamese.pkl")
    to.save(siam.state_dict(), model_path)
    if use_gpu:
        siam_test = Siamese().cuda()
    else:
        siam_test = Siamese().cpu()
    siam_test.load_state_dict(to.load(model_path))
    siam_test.eval()
