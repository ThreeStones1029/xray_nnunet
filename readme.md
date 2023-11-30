---
layout:     post   				    # 使用的布局（不需要改）
title:      实例分割 				# 标题 
subtitle:   nnUnet分割X线片椎骨 #副标题
date:       2023-11-27 				# 时间
author:     BY ThreeStones1029 						# 作者
header-img: img/about_bg.jpg 	    #这篇文章标题背景图片
catalog: true 						# 是否归档
tags:	图像分割							#标签
---

[TOC]

# 一、任务

## 1.1.主要要求

分割椎体侧位X线片，主要分割**椎体、椎弓根、肋骨**，植入物，分割示例如下图：

<img src="https://cdn.jsdelivr.net/gh/ThreeStones1029/blogimages/img/image-20231127113828966.png" alt="image-20231127113828966" style="zoom: 33%;" />

相比nnUnet，MedNeXT升点不多，可以先用nnUnet对x线片做个分割，方法如下:
**step1: 先把椎骨区域检测出来**

**Step2:对每节检测到的椎骨 (可以把原有椎骨检测框区域放大一倍，cover上下椎骨)，进一步实施语义分割，分割出椎体、椎弓、肋骨(胸骨有，腰椎没有)三块区域，棘突显影应该很浅所以可以暂时不考虑分割棘突区域。**

可以DRR预训练，然后再用术前X线片调优，然后用术中X线片进一步调优，最后用一些边界修正算法(上次你论文里的) 修正下就行

# 二、计划

## 2.1.了解这个网络和背景

nnUnet也称no-new U-Net，自适应的U-Net网络，首次提出使用在医学图像分割数据集上，在不同数据集（或不同的部位）的医学图像上进行分割时，往往需要具有不同结构的网络和不同的训练方案，自适应是指模型在对不同的数据集进行训练时，可以自动的调整 batch-size、patch-size 等，以达到很好的效果。

作者在一个医学图像分割十项全能比赛当中的 6 个数据集集中都取到了当时最好的结果，这个比赛一共 10 个数据集，它会给出 7 个数据集让你训练，然后在其他 3 个数据集上进行验证。

nnU-Net 由三种基础的 U-Net 网络组成，分别是 2D U-Net，3D U-Net 和 U-Net Cascade。其中，2D 和 3D U-Net 产生一个全像素的分割结果，U-Net Cascade 先产生一个低像素的分割结果，再对其进行微调。

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9zMi5heDF4LmNvbS8yMDIwLzAyLzIxLzN1b2J2dC5qcGc?x-oss-process=image/format,png)

[更多可以看论文](https://arxiv.org/abs/1904.08128)

![nnU-Net overview](https://github.com/MIC-DKFZ/nnUNet/raw/master/documentation/assets/nnU-Net_overview.png)

## 2.2.跑通代码

目前X线片数据集还没有制作，可以先下载别的数据集跑通，计划先跑通眼底数据集。

## 2.3.X线片数据集制作

需要制作数据集，计划使用label-studio这个工具，可以多人标注

![image-20231127205357537](https://cdn.jsdelivr.net/gh/ThreeStones1029/blogimages/img/image-20231127205357537.png)

## 2.4.模型训练

先不加载预训练模型训练，看看分割效果

训练过程中可以去做DRR的数据集，再做预训练

## 2.5.用水平集或者别的方法优化分割结果

![image-20231127210222544](https://cdn.jsdelivr.net/gh/ThreeStones1029/blogimages/img/image-20231127210222544.png)

# 三、实施

## 3.1.下载nn-unet代码

![image-20231127114227825](https://cdn.jsdelivr.net/gh/ThreeStones1029/blogimages/img/image-20231127114227825.png)

[代码链接](https://github.com/MIC-DKFZ/nnUNet)

## 3.2.配置环境

### 3.2.1.环境要求

环境要求，需要根据自己的电脑显卡配置安装

根据官方文档，最好使用python3.9

torch与自己的硬件匹配

### 3.2.2.我的环境安装使用命令

（1）创建虚拟环境

~~~bash
conda create -n nnunet python==3.9
~~~

（2）安装torch

~~~bash
# 需要先进入虚拟环境 conda activate nnunet
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
~~~

（3）安装其余包

如果你安装的torch版本小于2.0.0，需要修改pyproject.toml的第33行

torch>=2.0.0修改为你安装的版本号（若你安装的版本号大于2.0.0不需要修改），由于我安装的1.12.1所以我修改为了torch>=1.12.1

~~~bash
# 在根目录下
pip install -e .
~~~

（4）新建文件夹

在nnunetv2文件夹下新建三个文件夹

~~~bash
nnUNet_raw # 原数据集放置的文件夹
nnUNet_preprocessed # 预处理后数据保存的地方
nnUnet_results # 训练模型权重将会保存的地方
~~~

（5）设置环境变量

ubuntu系统可以设置

~~~bash
export nnUNet_raw="/home/jjf/Desktop/nnUNet/nnuetv2/nnUNet_raw"
export nnUNet_preprocessed="/home/jjf/Desktop/nnUNet/nnuetv2/nnUNet_preprocessed"
export nnUNet_results="/home/jjf/Desktop/nnUNet/nnuetv2/nnUnet_results"
~~~

windows可以不设置

修改nnunetv2/paths.py

~~~bash
# 注释掉这三行
# nnUNet_raw = os.environ.get('nnUNet_raw')
# nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
# nnUNet_results = os.environ.get('nnUNet_results')

# 加上以下三行
nnUNet_raw = "你新建的这个对应文件夹路径"
nnUNet_preprocessed = "新建的这个对应文件夹路径"
nnUNet_results = "新建的这个对应文件夹路径"
~~~

## 3.3.数据集制作

### 3.3.1.数据集标注

计划采用label-studio

访问方式：10.201.178.193:8080

标注案例：待确定



### 3.3.2.数据集格式转换

待写

## 3.4.测试眼球底分割数据集

目前我们还没有数据集，所以先测试一下能不能跑通CHASEDB1数据集

### 3.4.1.数据集下载

[下载链接](https://aistudio.baidu.com/datasetdetail/81247)

把下好的数据集解压后放在前面新建的nnUNet_raw文件夹

### 3.4.2.数据集格式转换

![image-20231127214443835](https://cdn.jsdelivr.net/gh/ThreeStones1029/blogimages/img/image-20231127214443835.png)

（原图像和mask需要同一个格式）

我们将原图jpg修改为png，不能单纯修改后缀,可以使用PIL.Image修改

可以新建一个tools文件夹，里面新建data_format_preprocess.py文件，代码如下：

~~~python
from PIL import Image
import os

class DataFormatConversion:
    def __init__(self, images_folder, save_images_folder):
        """
        param, images_folder:raw images folder path
        param, save_images_folder: images save path
        """
        self.images_folder = images_folder
        self.save_images_folder = save_images_folder


    def jpgs2pngs(self):

        for root, dirs, files in os.walk(self.images_folder):
            for file in files:
                if file.endswith(".jpg"):
                    image = Image.open(os.path.join(self.images_folder, file))
                    file_name = file.split('.')[0]
                    image.save(os.path.join(self.save_images_folder, file_name + ".png"))
                    print('The ', file, ' conversion from JPG to PNG is successful')


if __name__ == "__main__":
    data_format_conversion = DataFormatConversion(images_folder="nnunetv2/nnUNet_raw/CHASEDB1",save_images_folder="nnunetv2/nnUNet_raw/CHASEDB1")
    data_format_conversion.jpgs2pngs()
~~~

在nnUnet文件夹下运行

~~~bash
python nnunetv2/tools/data_format_preprocess.py
~~~

运行后，再把文件夹里面的jpg删除了，只留下png

### 3.4.3.数据集划分

因为只做测试，可以手动划分一下，原数据集总共84张图片，包括28张原图，每一张原图对应两张mask（两名医生分别标注了一张），总共56张mask，我们选任意一个人的标注就行，我统一选的1st那个人标注的

~~~bash
images_Tr为训练集原图
images_Ts为测试集原图
labels_Tr为训练集标签
labels_Ts为测试集标签
~~~

![image-20231127200720406](https://cdn.jsdelivr.net/gh/ThreeStones1029/blogimages/img/image-20231127200720406.png)

训练集如下图

![image-20231127221927351](https://cdn.jsdelivr.net/gh/ThreeStones1029/blogimages/img/image-20231127221927351.png)

训练集mask

![image-20231127213212254](https://cdn.jsdelivr.net/gh/ThreeStones1029/blogimages/img/image-20231127213212254.png)

### 3.4.4.json文件生成





## 3.5.X线片数据集训练

### 3.5.1.数据集制作

待写

# 四、参考资料

[1.代码地址](https://github.com/MIC-DKFZ/nnUNet)

[2.论文地址](https://arxiv.org/abs/1904.08128)

[3.CHASEDB1眼底数据集地址](https://aistudio.baidu.com/datasetdetail/81247)

[4.我的代码github地址](https://github.com/ThreeStones1029/xray_nnunet)

