'''
Descripttion: 本文件主要用来通过ITK将3D_mask生成2D_mask来制作数据
version: 
Author: ShuaiLei
Date: 2023-12-05 15:46:18
LastEditors: ShuaiLei
LastEditTime: 2023-12-10 14:16:12
'''
from genDRR import genDRR
from drr_image_postprocess import gen_2D_mask, flipdrr, rot_image
import os
import numpy as np
from glob import glob
import time
import cv2
from gen_json import InitDatasetJson
import platform


class GenInitDRRMask:
    def __init__(self, ct_root_path, APorLA_orientation, save_image_file, init_dataset_json_path) -> None:
        """
        params
        self.sdr
        self.height
        self.delx
        self.ct_root_path
        self.APorLA_orientation 
        self.save_image_file 
        self.num_samples
        self.rot_range_list
        self.trans_range_list
        """
        self.sdr = 500
        self.height = 1536
        self.delx = 0.25
        self.ct_root_path = ct_root_path
        self.APorLA_orientation = APorLA_orientation
        self.save_image_file = save_image_file
        self.num_samples = 2

        if self.APorLA_orientation == "AP":
            self.rot_range_list = [(80, 100), (170, 190), (170, 190)]
            self.trans_range_list = [(-10, 10), (-10, 10), (-10, 10)]
        else:
            self.rot_range_list = [(-10, 10), (80, 100), (-10, 10)]
            self.trans_range_list = [(-10, 10), (-10, 10), (-10, 10)]

        self.rotations, self.translations = self.gen_random_pose_parameters()

        self.init_dataset_json = InitDatasetJson(init_dataset_json_path)
        self.init_json() # 初始化数据集json文件
    

    def init_json(self):
        # 添加投影参数信息
        self.init_dataset_json.add_info(self.sdr, self.height, self.delx)
        # 添加mask类别信息
        self.init_dataset_json.add_masks_categories()

        
    # 随机生成参数
    def gen_random_pose_parameters(self):
        """
        rot_range_list:指定三个方向的旋转参数范围
        trans_range_list:指定三个方向的偏移大小范围
        rotation = [90, 180, 180], translation = [0, 0, 0]为正位
        num_samples:为生成的数量
        """
        rotations = []
        translations = []
        for _ in range(self.num_samples):
            rotation = []
            translation = []
            rotation.append(np.random.uniform(*self.rot_range_list[0]))
            rotation.append(np.random.uniform(*self.rot_range_list[1]))
            rotation.append(np.random.uniform(*self.rot_range_list[2]))
            rotations.append(rotation)

            translation.append(np.random.uniform(*self.trans_range_list[0]))
            translation.append(np.random.uniform(*self.trans_range_list[0]))
            translation.append(np.random.uniform(*self.trans_range_list[0]))
            translations.append(translation)

        return rotations, translations


    def gen_multple_cts_drrs_and_masks(self):
        '''
        description: 多个CT生成正位drr以及mask
        param {*} self
        param {*} ct_root_path
        return {*}
        '''
        ct_path_list = []
        ct_name_list = os.listdir(self.ct_root_path)
        for ct_name in ct_name_list:
            ct_path_list.append(os.path.join(self.ct_root_path, ct_name))
        
        if self.APorLA_orientation == "AP":
            for single_ct_path in ct_path_list:
                self.gen_single_ct_drrs_and_masks(single_ct_path)
        
        if self.APorLA_orientation == "LA":
            for single_ct_path in ct_path_list:
                self.gen_single_ct_drrs_and_masks(single_ct_path)

        # 生成初始数据集json文件
        self.init_dataset_json.gen_init_dataset_json()
        # 保存json文件
        self.init_dataset_json.save_dataset()


    def gen_single_ct_drrs_and_masks(self, single_ct_path):
        '''
        description: 单个CT生成drr以及mask
        param {*} sdr
        param {*} heighty
        param {*} delx
        param {*} single_ct_path
        param {*} save_image_file
        param {*} rotations
        param {*} translations
        return {*}
        '''
        # create save folder
        os.makedirs(os.path.join(self.save_image_file, "masks"), exist_ok=True)
        os.makedirs(os.path.join(self.save_image_file, "images"), exist_ok=True)

        i = 0
        # get ct name
        if platform.system().lower() == "linux":
            ct_name = single_ct_path.split("/")[-1]
        else:
            ct_name = single_ct_path.split("\\")[-1]
        self.init_dataset_json.add_ct(ct_name)
        for rotation, translation in zip(self.rotations, self.translations):
            filepaths = glob(os.path.join(single_ct_path, '*seg.nii.gz'))

            # 需呀将CT放到路径开头,这样才能生成正确的json文件
            ct_filepath = os.path.join(single_ct_path, ct_name + '.nii.gz')
            filepaths.insert(0, ct_filepath)
            
            i += 1
            for filepath in filepaths:
                basename = os.path.basename(filepath)
                basename_wo_ext = basename[:basename.find('.nii.gz')]
                
                if "seg" not in basename_wo_ext:
                    self.gen_drr(ct_name, i, rotation, translation, filepath)

                # AP only need to gen vertebrae body mask
                if basename_wo_ext.endswith("body_seg") and self.APorLA_orientation == "AP":
                    self.gen_AP_masks(basename_wo_ext, ct_name, i, rotation, translation, filepath)

                # LA only need to gen vertebrae body pecidle other mask
                if self.APorLA_orientation == "LA" and (basename_wo_ext.endswith("body_seg") or basename_wo_ext.endswith("pedicle_seg") or basename_wo_ext.endswith("other_seg")):
                    self.gen_LA_masks(basename_wo_ext, ct_name, i, rotation, translation, filepath)

                

    def gen_AP_masks(self, basename_wo_ext, ct_name, i, rotation, translation, filepath):
        # get cur vertebrae name
        vertebrae_name = basename_wo_ext[:basename_wo_ext.find('seg')]
        mask_name = ct_name + "_" + vertebrae_name + str(i) + ".png"
        saveIMG = os.path.join(self.save_image_file, "masks", mask_name)

        # generate drr
        genDRR(self.sdr, self.height, self.delx, rotation, translation, filepath, saveIMG)
        # generate 2d mask
        gen_2D_mask(saveIMG)

        # add mask info to json
        width, height = cv2.imread(saveIMG).shape[:2]
        self.init_dataset_json.add_mask(mask_name, self.APorLA_orientation, width, height, rotation, translation)
        self.init_dataset_json.add_ct_vertebrae_categoties(mask_name)


    def gen_LA_masks(self, basename_wo_ext, ct_name, i, rotation, translation, filepath):
        # get cur vertebrae name
        vertebrae_name = basename_wo_ext[:basename_wo_ext.find('seg')]
        mask_name = ct_name + "_" + vertebrae_name + str(i) + ".png"
        saveIMG = os.path.join(self.save_image_file, "masks", mask_name)

        # generate drr
        genDRR(self.sdr, self.height, self.delx, rotation, translation, filepath, saveIMG)
        # generate 2d mask
        gen_2D_mask(saveIMG)  

        # 侧位需要顺时针旋转90度
        rot_image(saveIMG)
        
        # add mask info to json
        width, height = cv2.imread(saveIMG).shape[:2]
        self.init_dataset_json.add_mask(mask_name, self.APorLA_orientation, width, height, rotation, translation)
        self.init_dataset_json.add_ct_vertebrae_categoties(mask_name)


    def gen_drr(self, ct_name, i, rotation, translation, filepath):
        drr_image_name = ct_name + "_" + str(i) + ".png"
        saveIMG = os.path.join(self.save_image_file, "images", drr_image_name)
        genDRR(self.sdr, self.height, self.delx, rotation, translation, filepath, saveIMG)
        
        # 侧位需要顺时针旋转90度
        if self.APorLA_orientation == "LA":
            rot_image(saveIMG)

        # add mask info to json
        width, height = cv2.imread(saveIMG).shape[:2]

        self.init_dataset_json.add_image(drr_image_name, ct_name, self.APorLA_orientation, width, height, rotation, translation)


if __name__ == "__main__":

    start_time = time.time()

    init_dataset = GenInitDRRMask(ct_root_path="data/ct_mask", APorLA_orientation="LA", init_dataset_json_path="data/LA/LA_init_dataset.json")
    init_dataset.gen_multple_cts_drrs_and_masks()

    print("consume_time:", time.time() - start_time)