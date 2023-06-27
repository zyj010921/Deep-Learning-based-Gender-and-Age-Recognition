# -*- coding = utf-8 -*-
# @Time : 2023/5/7 11:27
# @Author : zyj
# @File : estimation.py
# @Software : PyCharm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageFont, Image, ImageDraw
from torchvision import transforms

from My_project.Code7.Net import MyMulitpleTaskNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 设置中文字体
model_age = MyMulitpleTaskNet()
model_gen = MyMulitpleTaskNet()
model_age.load_state_dict(torch.load("./models/best_age51.635_gen54.038.pth"))
model_gen.load_state_dict(torch.load("./models/best_gen85.829.pth"))
model_age = model_age.to(device)
model_gen = model_gen.to(device)

#图像预处理
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(),
                                transforms.Resize((64, 64)),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                                ])
image_path = "../data/UTKTest/32_1_0_20170103163352672.jpg.chip.jpg"
image_path = "../data/UTKTest/34_0_1_20170113145505629.jpg.chip.jpg"
image_path = "../data/UTKTest/44_0_0_20170104201051081.jpg.chip.jpg"
image_path = "../data/UTKTest/55_1_1_20170113185105936.jpg.chip.jpg"
img_pil = Image.open(image_path)
# img.show()

# 预处理
input_img = transform(img_pil)  # torch.Size([3, 64, 64])
input_img = input_img.unsqueeze(0).to(device) # torch.Size([1, 3, 64, 64])

# 执行前向预测
pred1 = model_age(input_img)
pred2 = model_gen(input_img)
print(pred1)
print(pred2)

pred_age = int(pred1[0] * 100)
# pred_gen = int(pred2[1].argmax(1)[0])
pred_gen = int(torch.argmax(pred2[1], dim=1))
print(pred_age)
print(pred_gen)

draw = ImageDraw.Draw(img_pil)
gen = '男' if pred_gen == 0 else '女'
age = "{}~{}".format(pred_age-2,pred_age+2)
# print(gen)
# print(age)

text = u"年龄：{}\n性别：{}".format(age,gen)
text_utf8 = text.encode("utf-8")
font = ImageFont.truetype('simhei.ttf', size=24)
draw.text((15,20),text_utf8.decode('utf-8'),font=font,fill=(255,0,0))
img_pil.show()
