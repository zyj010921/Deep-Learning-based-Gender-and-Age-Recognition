# -*- coding = utf-8 -*-
# @Time : 2023/4/27 17:25
# @Author : zyj
# @File : renamePic_Old.py
# @Software : PyCharm
import os

path = "../data/crop_part1"
# aim_path = "../data/AgeClassified20/trainset/5~10"

# 获取该目录下所有文件，存入列表中
fileList = os.listdir(path)
print(fileList)
n = 0
for filename in fileList:
    # 设置旧文件名（就是路径+文件名）
    # oldname = path+ os.sep + fileList[n]  # os.sep添加系统分隔符
    # 设置旧文件名（就是路径+文件名）
    oldname = path + os.sep + filename  # os.sep添加系统分隔符

        # print(oldname)
    age = int(filename.split('_')[0])
    gen = int(filename.split('_')[1].split('.')[0])
    # print("gen ",gen)
    if  gen == 1:
        gen = "0"
    elif gen == 0:
        gen = "1"
    try:
        name = str(age) + '_' + gen + '_' + "{}".format(n) + '_'+ ".jpg.chip.jpg"
        # name = filename.split('-')[0] + "-" + filename.split('-')[1] + "-" +filename.split('-')[2].split('p')[0] + ".png"
        # print("newname is {}".format(name))
    except:
        continue
    n += 1
    # print(oldname)
    newname = "../data/temp/" + os.sep +name       # 改名，存到temp文件夹中，需要手动划分测试集与数据集
    os.rename(oldname,newname)
    print(oldname + "====>" + newname)