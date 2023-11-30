'''
Descripttion: 
version: 
Author: ShuaiLei
Date: 2023-11-29 21:11:18
LastEditors: ShuaiLei
LastEditTime: 2023-11-29 22:21:24
'''
import cv2
import numpy as np
import os
import glob
 

class DiceEvaluation:
    def __init__(self, GT_folder_path, Pre_floder_path):
        self.GT_folder_path = GT_folder_path
        self.Pre_floder_path = Pre_floder_path


    def gen_imgs_path_list(self, image_folder):
        imgs_path = glob.glob(os.path.join(image_folder, '*.png')) 
        return imgs_path


    def calculate_image_dice(self, gt_img_path, pre_img_path):
        """
        计算两张二值化图片的Dice系数
        :param gt_img_path: 真实二值化图片路径
        :param pre_img_path: 预测二值化图片路径
        :return: Dice系数
        """
        # 读取二值化图片
        gt = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
        pre = cv2.imread(pre_img_path, cv2.IMREAD_GRAYSCALE)

        # 计算交集和并集
        intersection = np.logical_and(gt, pre)
        union = np.logical_or(gt, pre)

        # 计算Dice系数
        dice_coefficient = (2.0 * intersection.sum()) / (gt.sum() + pre.sum())

        return dice_coefficient


    def calculate_images_mean_dice(self):
        sum_dice = 0
        
        gt_files = self.gen_imgs_path_list(self.GT_folder_path) 
        pre_files = self.gen_imgs_path_list(self.Pre_floder_path)
        
        try:
            if len(gt_files) != len(pre_files):
                raise ValueError("Check that the number of true and predicted images are the same")
        except ValueError as e:
            print(f"error:{e}")

        num_imgs = len(gt_files)
        
        for _ in range(num_imgs):
            file_name = os.path.basename(gt_files[_])
            gt_img_path = os.path.join(self.GT_folder_path, file_name)
            pre_img_path = os.path.join(self.Pre_floder_path, file_name)
            dice = self.calculate_image_dice(gt_img_path, pre_img_path)
            print(file_name, dice)
            sum_dice += dice
        return sum_dice / num_imgs


if __name__ == "__main__":

    gt_image_path = 'nnunetv2/nnUNet_raw/Dataset100_CHASEDB1/labelsTs/Image_11L.png'
    pre_image_path = 'nnunetv2/nnUNet_predict/Image_11L.png'

    # 计算Dice系数
    pre_gt_eval = DiceEvaluation("nnunetv2/nnUNet_raw/Dataset100_CHASEDB1/labelsTs",
                                 "nnunetv2/nnUNet_predict")
    mean_dice = pre_gt_eval.calculate_images_mean_dice()
    print(f"Mean Dice is: {mean_dice}")