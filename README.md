# Deep-Learning-based-Gender-and-Age-Recognition
使用一个CNN网络模型对人脸图像进行性别分类和年龄估计
年龄预测成功率约为50%，性别预测成功率约为85%;
程序使用UTK数据集进行训练与测试，下载地址https://susanqq.github.io/UTKFace/
注：请注意程序中涉及到文件位置的代码，请自行更改;

运行环境：
  python3.9
  torch 2.0.0+cu118
  numpy 1.24.1
  注：支持CPU运行，若要使用GPU，请自行安装CUDA、CUDNN;

使用说明：
  1. 代码文件存放在Code文件夹，同一目录下创建data文件夹，数据集下载解压后使用renamePic_New.py进行重命名，文件路径自行指定;
  2. data文件夹下创建UTKFace与UTKTest文件夹用来存放训练集与测试集;
  3. Net.py中存放网络模型,dataset存放数据处理代码;
  4. train_age.py用以训练年龄预测模型，train_gen.py用以训练性别预测模型，estimation.py进行效果展示;
  5. 模型存放位置与运行日志存放位置请自行指定与创建;
