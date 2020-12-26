from model import Siamese
from model import is_use_gpu
import numpy as np
import sys
import os
from torch.utils.data import DataLoader, Dataset
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import glob
import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image
import warnings


class TestDataset(Dataset):
    def __init__(self, image_path_1, image_path_2):
        self.image_path_1 = image_path_1
        self.image_path_2 = image_path_2
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Compose([transforms.Scale((img_size, img_size))]),
            # transforms.CenterCrop(img_size),
            transforms.Resize(img_size),

            transforms.ToTensor(),  # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
        ])

    def __getitem__(self, idx):
        img1 = self.transform(np.array(Image.open(self.image_path_1).convert("RGB")))
        img2 = self.transform(np.array(Image.open(self.image_path_2).convert("RGB")))

        img1 = torch.Tensor(np.reshape(img1, (3, img_size, img_size)))
        img2 = torch.Tensor(np.reshape(img2, (3, img_size, img_size)))
        y1 = torch.Tensor(np.ones(1, dtype=np.float32))
        return img1, img2

    def __len__(self):
        return 1


img_size = 128
threshold = 0.76
model_path = "./model.pkl"


def sigmoid(x):
    return 1/(1+math.exp(-x))


def verify_dogs(left_dog, right_dog):
    image_path_1 = "./static/dogs/%s/0.PNG" % format(left_dog)
    image_path_2 = "./static/" + right_dog  #dogs/0001/1.PNG"
    test_set = TestDataset(image_path_1, image_path_2)
    test_dataloader = DataLoader(test_set, shuffle=True, batch_size= 1, num_workers=0)

    if is_use_gpu():
        siam_test = Siamese().cuda()
        siam_test.load_state_dict(torch.load(model_path))
        siam_test.eval()
        for data in test_dataloader:
            im1, im2 = data
            diss = siam_test.evaluate(im1.cuda(),im2.cuda())
    else:
        siam_test = Siamese().cpu()
        siam_test.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        siam_test.eval()
        for data in test_dataloader:
            im1, im2 = data
            diss = siam_test.evaluate(im1.cpu(), im2.cpu())

    # 0 ~ 1
    similarity = 2 * (1 - math.fabs(sigmoid(diss)))

    return bool(similarity > threshold), similarity
