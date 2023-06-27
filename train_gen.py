# -*- coding = utf-8 -*-
# @Time : 2023/5/1 17:00
# @Author : zyj
# @File : train_age.py
# @Software : PyCharm
import os

import numpy as np
import torch
from PIL.Image import Image
from torch.utils.data import Dataset, DataLoader

# [age] is an integer from 0 to 116,
# [gender] is either 0 (male) or 1 (female)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from My_project.Code7.Net import MyMulitpleTaskNet
from My_project.Code7.dataset import AgeGenderDataset

max_age = 100
num_epochs = 300
BATCH_SIZE = 8
learning_rate= 1e-2
age_acc = 0.0
gen_acc = 0.0
max_age_acc = 22.896
max_gen_acc = 0.0

# 数据加载
# ds = AgeGenderDataset("E:\PythonCode\\torch2\\fromChatGPT\data\\trainset")
ds = AgeGenderDataset("../data/UTKFace")
dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)

# testds = AgeGenderDataset("E:\PythonCode\\torch2\\fromChatGPT\data\\testset")
testds = AgeGenderDataset("../data/UTKTest")
testloader = DataLoader(testds, batch_size=BATCH_SIZE,shuffle=False,drop_last=True)

train_data_size = ds.__len__()
test_data_size = testds.__len__()
print("训练数据集长度为:{}".format(train_data_size))
print("测试数据集长度为:{}".format(test_data_size))

# enumerate将可迭代对象组合为索引序列  例如：[(0, 'Tom'), (1, 'Jerry')]
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(), sample_batched['gender'])
    break

# 检查设备
# 检查是否可以利用GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.')
else:
    print('CUDA is available!')

class MyMulitpleTaskNet(torch.nn.Module):
    def __init__(self):
        super(MyMulitpleTaskNet, self).__init__()
        self.cnn_layers = torch.nn.Sequential(
            # x => [3, 64, 64]
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2, 2),
            # x => [32, 32, 32]
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2, 2),
            # x => [64, 16, 16]
            torch.nn.Conv2d(64, 96, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(96),
            torch.nn.MaxPool2d(2, 2),
            # x => [96, 8, 8]
            torch.nn.Conv2d(96, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2, 2),

            # x => [128, 4, 4]
            torch.nn.Conv2d(128, 196, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(196),
            torch.nn.MaxPool2d(2, 2)
            # x => [196, 2, 2]
        )

        # 全局最大池化
        self.global_max_pooling = torch.nn.AdaptiveMaxPool2d((1, 1))
        # x => [196, 1, 1]

        # 预测age（回归）
        self.age_fc_layers = torch.nn.Sequential(
            torch.nn.Linear(196, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 1),
            torch.nn.Dropout(p=0.5),
            torch.nn.Sigmoid()
        )

        # 预测gender（分类）
        self.gender_fc_layers = torch.nn.Sequential(
            torch.nn.Linear(196, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 2),
            torch.nn.ReLU()#测试
        )

    def forward(self, x):
        # x => [3, 64, 64]
        x = self.cnn_layers(x)

        # x => [196, 2, 2]
        B, C, H, W = x.size()
        out = self.global_max_pooling(x).view(B, -1)  # -1的值由其他层推断出来

        # 全连接层
        out_age = self.age_fc_layers(out)
        out_gender = self.gender_fc_layers(out)
        return out_age, out_gender

# 训练
model = MyMulitpleTaskNet()
# model = torch.load("./models/best_35.006.pth")
model.load_state_dict(torch.load("./models/best_gen85.829.pth"))

print(model)

# 使用GPU
if train_on_gpu:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# sets the module in training mode.
model.train()

# 损失函数
mse_loss = torch.nn.MSELoss()
cross_loss = torch.nn.CrossEntropyLoss()
# cross_loss = torch.nn.BCELoss() #二分类交叉熵函数
index = 0

writer = SummaryWriter("./logs_gen_train")

for epoch in range(num_epochs):
    train_loss = 0.0
    age_correct = 0
    gen_correct = 0
    # 依次取出每一个图片与label
    for sample_batched in tqdm(dataloader):
        images_batch, age_batch, gender_batch = \
            sample_batched['image'], sample_batched['age'], sample_batched['gender']
        if train_on_gpu:
            images_batch, age_batch, gender_batch = images_batch.cuda(), age_batch.cuda(), gender_batch.cuda()

        optimizer.zero_grad()

        # forward pass
        m_age_out, m_gender_out = model(images_batch)
        age_batch = age_batch.view(-1, 1).float()
        gender_batch = gender_batch.long()

        # print("m_age_out is {} \nm_gender_out is {} \n".format(m_age_out_, m_gender_out_))
        # print("age_batch is {}".format(age_batch))
        # print("gender_batch is {} \n".format(gender_batch))
        # print("view age_batch is {}".format(age_batch))
        # print("view gender_batch is {}".format(gender_batch))
        # print("age_batch is {} \ngender_batch is {} \n".format(age_batch,gender_batch))

        # calculate the batch loss
        # loss = cross_loss(m_gender_out, gender_batch)
        loss =cross_loss(m_gender_out, gender_batch)
        # backward pass
        loss.backward()

        # perform a single optimization step (parameter update)
        optimizer.step()

        # update training loss
        train_loss += loss.item()
        # if index % 100 == 0:
        #     print('step: {} \tTraining Loss: {:.6f} '.format(index, loss.item()))
        # index += 1

        for i in range(BATCH_SIZE):
            if int(m_age_out[i] * 100) <= int(age_batch[i] * 100) + 2 and int(m_age_out[i] * 100) >= int(
                    age_batch[i] * 100) - 2:
                # print("训练年龄输出 is {} , 实际年龄 is {}".format(int(m_age_out[i] * 100), int(age_batch[i] * 100)))
                age_correct += 1
        gen_correct += int(torch.sum(torch.eq(torch.argmax(m_gender_out, dim=1), gender_batch)))
    print("训练成功率为：{:.3f}%".format(gen_correct/train_data_size * 100))

    writer.add_scalar("train_age_acc",age_correct / train_data_size * 100,epoch)
    writer.add_scalar("train_gen_acc", gen_correct / train_data_size * 100, epoch)

    # print("train age_acc is {:.3f}%, age_correct is {}".format((age_correct / train_data_size * 100),age_correct))
    # print("train gen_acc is {:.3f}%, gen_correct is {}".format((gen_correct / train_data_size * 100),gen_correct))

    #   初步测试
    model.eval()
    with torch.no_grad():
        age_correct = 0
        gen_correct = 0
        total = 0
        temp_cnt = 0
        for sample_batched in tqdm(testloader):
            images_batch, age_batch, gender_batch = sample_batched['image'], sample_batched['age'], sample_batched['gender']
            if train_on_gpu:
                images_batch, age_batch, gender_batch = images_batch.cuda(), age_batch.cuda(), gender_batch.cuda()

            m_age_out, m_gender_out = model(images_batch)

            age_batch = age_batch.view(-1, 1).float()
            gender_batch = gender_batch.long()
            # for i in range(BATCH_SIZE):
            #     total += 1

            for i in range(BATCH_SIZE):
                if int(m_age_out[i]*100) >= 60 and int(age_batch[i]*100) >= 60:
                    # print("m_age_out is {} , age_batch is {}".format(int(m_age_out[i]*100),int(age_batch[i]*100)))
                    temp_cnt += 1
                    age_correct += 1
                if int(m_age_out[i]*100) <=  int(age_batch[i]*100)+2 and int(m_age_out[i]*100) >=  int(age_batch[i]*100)-2:
                    # print("m_age_out is {} , age_batch is {}".format(int(m_age_out[i]*100),int(age_batch[i]*100)))
                    age_correct += 1
                total += 1
            # age_correct += int(torch.sum(torch.eq(m_age_out, age_batch)))
            gen_correct += int(torch.sum(torch.eq(torch.argmax(m_gender_out, dim=1), gender_batch)))



            # print("test age_batch is {}\ntest m_age_out is {}".format(age_batch, m_age_out))
            # print("test gen_batch is {}\ntest m_gender_out is {}".format(gender_batch, m_gender_out))
            # print("view age_batch is {}".format(age_batch))
            # print("view gen_batch is {}".format(gender_batch))

            # print("test age_correct is {}".format(age_correct))
            # print("test gen_corrrect is {}".format(gen_correct))
        print("total is {}".format(total))
        print(">65 is {}".format(temp_cnt))
        age_acc = age_correct / total * 100
        gen_acc = gen_correct / total * 100

        # if age_acc > max_age_acc:
        #     if max_age_acc == 0:
        #         print()
        #     else:
        #         # os.remove("./models/best_age{:.3f}_gen{:.3f}.pth".format(max_age_acc,max_gen_acc))
        #         print("已删除成功率为{:.3f}的模型".format(max_age_acc))
        #
        #     max_age_acc = age_acc
        #
        #     torch.save(model.state_dict(), "./models/best_age{:.3f}_gen{:.3f}.pth".format(max_age_acc,gen_acc))
        #     print("已保存成功率为{:.3f}的模型".format(max_age_acc))
        #
        #     model.load_state_dict(torch.load("./models/best_age{:.3f}_gen{:.3f}.pth".format(max_age_acc,gen_acc)))
        #     print("已更换best_age{:.3f}_gen{:.3f}".format(max_age_acc,gen_acc))
        if gen_acc > max_gen_acc:
            print()
            # if max_age_acc == 0:
            #     print()
            # else:
            #     os.remove("./models/best_age{:.3f}_gen{:.3f}.pth".format(age_acc,max_gen_acc))
            #     print("已删除成功率为{:.3f}的模型".format(max_age_acc))

            max_gen_acc = gen_acc

            torch.save(model.state_dict(), "./models/best_gen{:.3f}.pth".format(max_gen_acc))
            print("已保存gender成功率为{:.3f}的模型".format(max_gen_acc))

            # model.load_state_dict(torch.load("./models/best_gen{:.3f}.pth".format(max_gen_acc)))
            print("使用成功率为：{:.3f}%".format(max_gen_acc))



        # print("\n第 {} 轮，年龄准确率为：{:.3f}%".format(epoch, age_acc))
        print("第 {} 轮，性别准确率为：{:.3f}%".format(epoch, gen_acc))
        # writer.add_scalar("test_age_acc", age_acc, epoch)
        # writer.add_scalar("test_gen_acc", gen_acc, epoch)

    # 计算平均损失
    train_loss = train_loss / train_data_size

    # 显示训练集与验证集的损失函数
    print('\nEpoch: {} \tTraining Loss: {:.6f} '.format(epoch, train_loss))

    print("\nmax_age_acc is {:.3f}%, max_gen_acc is {:.3f}%".format(max_age_acc,max_gen_acc))

    # 模型保存

# save model
# sets the module in evaluation mode.
# model.eval()
# torch.save(model, './models/age_gender_model.pt')
