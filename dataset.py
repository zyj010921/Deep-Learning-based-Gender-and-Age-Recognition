# -*- coding = utf-8 -*-
# @Time : 2023/5/1 17:26
# @Author : zyj
# @File : dataset.py
# @Software : PyCharm
import os

import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms


class AgeGenderDataset(Dataset):
    def __init__(self, root_dir):
        # Normalize: image => [-1, 1]  （利于更好的训练）
        # ToTensor() => Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225]),
                                             transforms.Resize((64, 64))
                                             ])

        img_files = os.listdir(root_dir)  # 存放的是所有图片的文件名

        # age: 0 ~100, 0 :male, 1 :female
        self.ages = []
        self.genders = []
        # 注意：self.image存放的是图片的路径
        self.images = []

        for file_name in img_files:
            age_gender_group = file_name.split("_")#原代码
            # age_gender_group = file_name.split("-")
            age_ = age_gender_group[0]
            gender_ = age_gender_group[1]
            self.genders.append(np.float32(gender_))
            # 将age缩小到[0, 1]的范围内
            self.ages.append(np.float32(age_) / 116)
            # os.path.join() 将路径和文件名合成为一个路径
            self.images.append(os.path.join(root_dir, file_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image_path = self.images[idx]
        else:
            image_path = self.images[idx]

        # img = cv.imread(image_path)  # BGR order
        img = Image.open(image_path).convert("RGB")
        sample = {'image': self.transform(img), 'age': self.ages[idx], 'gender': self.genders[idx]}

        # 返回一个字典形式
        return sample


# 数据集验证
if __name__ == "__main__":
    # transform1 = transforms.Compose([transforms.ToTensor()])
    ds = AgeGenderDataset("../data/UTKFace")
    for i in range(len(ds)):
        sample = ds[i]
        print(i, sample['image'].size(), sample['age'])
        # 提取一个batch的数据
        if i == 3:
            break
    # 定义数据加载器
    dataloader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
    for i in range(8):
        img = transforms.ToPILImage()(ds[i]["image"])
        # img2 = transforms.ToPILImage()(ds[1]["image"])
        # print(label, label2)
        img.show()  # 展示图片
        # img2.show()