'''
Descripttion: 
version: 
Author: ShuaiLei
Date: 2023-11-28 20:24:27
LastEditors: ShuaiLei
LastEditTime: 2023-11-29 20:13:59
'''
import os
from PIL import Image
import numpy as np


class MasksPreprocess:
    def __init__(self, images_folder) -> None:
        self.images_folder = images_folder

    
    def conver_to0_255images(self):
        """把label转换为二值图像,也即只有0, 255两个值"""
        for root, dirs, files in os.walk(self.images_folder):
            for file in files:
                if file.endswith(".png"):
                    img = Image.open(os.path.join(root, file))
                    img_array = np.array(img)
                    binary_array = np.where(img_array == 0, 0, 255)
                    binary_img = Image.fromarray(binary_array.astype(np.uint8))
                    binary_img.save(os.path.join(root, file))

    
    def check_channels_num(self):
        for root, dirs, files in os.walk(self.images_folder):
            for file in files:
                img = Image.open(os.path.join(root, file))
                print(file, " channels num = ", len(img.getbands()))


if __name__ == "__main__":
    Data_preprocess = MasksPreprocess("nnunetv2/nnUNet_predict")
    Data_preprocess.conver_to0_255images()