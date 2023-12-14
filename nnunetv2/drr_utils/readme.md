<!--
 * @Descripttion: 
 * @version: 
 * @Author: ShuaiLei
 * @Date: 2023-12-06 11:21:07
 * @LastEditors: ShuaiLei
 * @LastEditTime: 2023-12-10 16:56:47
-->
---
layout:     post   				    # 使用的布局（不需要改）
title:      实例分割 				# 标题 
subtitle:   nnUnet分割X线片椎骨 #副标题
date:       2023-12-10 				# 时间
author:     BY ThreeStones1029 						# 作者
header-img: img/about_bg.jpg 	    #这篇文章标题背景图片
catalog: true 						# 是否归档
tags:	图像分割							#标签
---

[TOC]

# X线片分割
# 更新:本文档DRR最新代码已重构成单独项目,具体看drr_utils仓库,这里作为测试版不再更新
# 一.DRR
## 1.1.数据生成
### 1.1.1.运行
在drr_utils目录下运行main.py
~~~bash
python main.py
~~~
~~~
PS:
(1)如果是windows系统应该可以直接运行成功,如果运行失败是ITK问题,可以将Windows_ITK_Gen_Drr文件下的itk_dll文件里的全部删除,重新运行Setup_exe.msi安装到itk_dll文件下.

(2)linuxs系统这里只有动态库,可能需要编译Linux_ITK_Gen_Drr下的gendrr.cpp文件
编译命令,在build目录下
cmake ..
make
记得把CmakeLists里面的ITK路径换成你自己ITK的路径
~~~
### 1.1.2.参数解释
~~~python
def main():
    dataset = GEN_CUT_DRRMASK_DATASET(data_root_path="data", has_pedicle=True, APorLA_orientation="AP", is_vis=True)
    dataset.create_dataset()
~~~

~~~bash
(1)data_root_path="data"
为drr以及对应生成的mask的根目录,里面的AP,为生成的正位数据根目录,LA为侧位数据根目录

(2)has_pedicle=True
表示是否有椎弓根ct数据,默认手动分割已经有了,所有ct数据都放在ct_mask下

(3)APorLA_orientation="AP"
表示生成正位数据或者侧位数据

(4)is_vis=True
表示是否可视化,如果需要可视化mask,可以到对应AP,LA文件下的vis文件下查看
~~~

投影参数
~~~bash
投影参数可以到gen_init_dataset.py修改,默认参数如下
self.sdr = 500 表示射线源到投影平面距离一半
self.height = 1536 表示生成的初始图像大小 可在正位或侧位images,以及masks文件下查看生成的初始数据
self.delx = 0.25 表示像素距离
self.num_samples = 2 表示每一例nii文件生成的drr数量
self.rot_range_list 为随机的角度范围
self.trans_range_list = 随机的偏移距离范围
注意:
[90, 180, 180] 为正位
[0, 90, 0] 为侧位
建议随机的范围不要过大,随机角度过大,投影生成的会造成mask,重叠较大
~~~

裁剪参数
~~~bash
数据需要裁剪出每一节椎体
裁剪逻辑为根据mask中心生成最小bbox
扩大bbox一定倍数裁剪,侧位默认扩大1.2倍,正位默认扩大1.5倍
扩大倍数系数可以在gen_cut_dataset.py 56 58行处修改
~~~


### 1.1.3.数据制作
当然由于数据隐私,nii文件我没有上传到github,需要百度网盘下载,制作数据主要有两种方式
(1)使用mimics制作
这个可以看群里的视频

(2)使用slicer制作
具体过程可看附件
