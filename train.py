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
            dog1_1_tensor = to.as_tensor(np.reshape(dog1_1_img, (3, img_size, img_size)), dtype=to.float32)
            dog1_2_tensor = to.as_tensor(np.reshape(dog1_2_img, (3, img_size, img_size)), dtype=to.float32)
            y1 = to.tensor(np.ones(1, dtype=np.float32), dtype=to.float32)

            dog2_1 = random.choice(dog2)
            dog2_1_img = self.transform(np.array(Image.open(dog2_1).convert("RGB")))
            dog2_1_tensor = to.as_tensor(np.reshape(dog2_1_img, (3, img_size, img_size)), dtype=to.float32)
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
        return SiamDataset.TrainDataset(self.train_paths, len(self.train_paths)*4)

    def get_test_dataset(self):
        return SiamDataset.TestDataset(self.test_paths, len(self.test_paths)*4)

    def shuffle(self):
        random.shuffle(self.img_pathss)
        test_size = int(len(self.img_pathss) * self.test_ratio)
        self.train_paths = self.img_pathss[test_size:]
        self.test_paths = self.img_pathss[:test_size]


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
        
    dataset_dir = "dataset/train-20"

    if use_gpu:
        siam = Siamese().cuda()
    else:
        siam = Siamese().cpu()
    Criterion = ContrastiveLoss()
    Optimizer = to.optim.Adam(siam.parameters(), lr=0.01)

    siam_dataset = SiamDataset(dataset_dir)
    print("Train dogs: %d, Test dogs: %d" % (len(siam_dataset.train_paths), len(siam_dataset.test_paths)))
    loss_history = []
    siam.train()
    for epoch in range(max_epochs):
        loops = 0
        siam_dataset.shuffle()
        train_dataloader = DataLoader(siam_dataset.get_train_dataset(), shuffle=True, batch_size=20, num_workers=15)
        test_dataloader = DataLoader(siam_dataset.get_test_dataset(), shuffle=True, batch_size=1, num_workers=15)
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
