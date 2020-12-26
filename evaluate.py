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


img_size = 128
# threshold = 0.76
model_path = "./trained/Siamese-500.pkl"
use_gpu = True


class TestDataset(Dataset):
    def __init__(self, left_image, right_images):
        self.left_image = left_image
        self.right_images = right_images
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Compose([transforms.Scale((img_size, img_size))]),
            transforms.Resize(img_size),
            transforms.ToTensor(),  # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
        ])

    def __getitem__(self, idx):
        left_img = self.left_image
        right_img = self.right_images[idx]

        img1 = self.transform(np.array(Image.open(left_img).convert("RGB")))
        img2 = self.transform(np.array(Image.open(right_img).convert("RGB")))

        img1 = torch.Tensor(np.reshape(img1, (3, img_size, img_size)))
        img2 = torch.Tensor(np.reshape(img2, (3, img_size, img_size)))
        y1 = torch.Tensor(np.ones(1, dtype=np.float32))
        return img1, img2

    def __len__(self):
        return len(self.right_images)


def sigmoid(x):
    return 1/(1+math.exp(-x))


siam_test = Siamese().cpu()
siam_test.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
siam_test.eval()
def evaludate_cpu(data):
    im1, im2 = data
    diss = siam_test.evaluate(im1.cpu(), im2.cpu())
    return diss


def verify_dogs(left_dogs, right_dogs):
    inferences = []

    for left_dog in left_dogs:
        inf = []
        test_set = TestDataset(left_dog, right_dogs)
        test_dataloader = DataLoader(test_set, shuffle=False, batch_size= 1, num_workers=0)
        for i, data in enumerate(test_dataloader):
            diss = evaludate_cpu(data)
            similarity = 2 * (1 - math.fabs(sigmoid(diss))) * 100
            inf.append(similarity)
            print("\r%s | %s = %f" % (left_dog, right_dogs[i], similarity), end="")
        print()
        inferences.append(inf)

    return inferences


def calculate_far_frr(inferences, group_size, threshold):
    fa, fr = 0, 0
    dogs = len(inferences)
    group_count = int(dogs / group_size)
    for dog in range(dogs):
        dog_group = int(dog / group_size)
        for g in range(group_count):
            if g == dog_group:
                for i in range(group_size):
                    if not (inferences[dog][g * group_size + i] > threshold):
                        fr += 1
            else:
                for i in range(group_size):
                    if inferences[dog][g * group_size + i] > threshold:
                        fa += 1

    fa_base = dogs * dogs - dogs * group_size
    fr_base = dogs * group_size
    return fa / fa_base, fr / fr_base


if __name__ == '__main__':
    inference_output_path = "output/inferences-chinatrust.tsv"
    eer_output_path = "output/far_frr-chinatrust.tsv"
    dog_input_root = "dataset/verification"
    dog_count = 85
    group_size = 2

    dog_paths = [x for x in os.walk(dog_input_root)]
    dog_paths.sort()
    img_paths = []
    for dog_path in dog_paths[1: dog_count + 1]:
        img_paths.extend(os.path.join(dog_path[0], x) for x in dog_path[2][:group_size])

    inferences = verify_dogs(img_paths, img_paths)
    dogs = len(img_paths)
    group_count = int(dogs / group_size)
    header = [""]
    for g in range(group_count):
        for i in range(group_size):
            header.append("{}-{}".format(g+1, i+1))
    lines = ["\t".join(header) + "\n"]
    for dd in range(dogs):
        line = header[dd + 1] + "\t"
        d = 0
        for g in range(group_count):
            for i in range(group_size):
                line += "{}\t".format(inferences[dd][d])
                d += 1
        lines.append(line[:-1] + "\n")
    with open(inference_output_path, "w") as fp:
        fp.writelines(lines)

    lines = ["Threshold\tFAR\tFRR\n"]
    for i in range(101):
        far, frr = calculate_far_frr(inferences, group_size, threshold=i)
        lines.append("%d\t%f\t%f\n" % (i, far, frr))

    with open(eer_output_path, "w") as fp:
        fp.writelines(lines)
