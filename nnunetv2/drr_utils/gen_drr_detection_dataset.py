'''
Description: 使用投影方式重新生成DRR检测数据,侧位需要投影全部,正位只需要body部分
version: 
Author: ThreeStones1029 221620010039@qq.com
Date: 2023-12-11 11:32:05
LastEditors: ShuaiLei
LastEditTime: 2023-12-11 22:24:53
'''
import platform
from genDRR import genDRR
from coco_detection_data import COCODetectionData
from dataset_sample import Dataset_sample
from drr_image_postprocess import rot_image, gen_2D_mask, compute_min_bbox_coverage_mask
from file_management import create_folder, linux_windows_split_name
import json
import numpy as np
import os
import cv2
import glob


class GenDetectionDataset:
    def __init__(self, ct_root_path, save_image_file, init_dataset_json_path) -> None:
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
        self.height = 2000
        self.delx = 0.25
        self.threshold = 0
        self.AP_num_samples = 50
        self.LA_num_samples = 50
        self.ct_root_path = ct_root_path
        self.save_image_file = save_image_file
        self.AP_rot_range_list = [(75, 105), (165, 195), (165, 195)]
        self.AP_trans_range_list = [(-15, 15), (-15, 15), (-15, 15)]
        self.LA_rot_range_list = [(-15, 15), (75, 105), (-15, 15)]
        self.LA_trans_range_list = [(-15, 15), (-15, 15), (-15, 15)]
        self.AP_rotations, self.AP_translations = self.gen_random_pose_parameters(self.AP_rot_range_list, self.AP_trans_range_list, self.AP_num_samples)
        self.LA_rotations, self.LA_translations = self.gen_random_pose_parameters(self.LA_rot_range_list, self.LA_trans_range_list, self.LA_num_samples)
        self.detection_dataset = COCODetectionData()
        # create save folder
        create_folder(os.path.join(self.save_image_file, "masks"))
        create_folder(os.path.join(self.save_image_file, "images"))


    # 随机生成参数
    def gen_random_pose_parameters(self, rot_range_list, trans_range_list, num_samples):
        dataset_sample = Dataset_sample()
        rotations, translations = dataset_sample.Monte_Carlo_sample_dataset(rot_range_list,trans_range_list, num_samples)
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
        # 得到所有ct子文件夹
        for ct_name in ct_name_list:
            if os.path.isdir(os.path.join(self.ct_root_path, ct_name)):
                ct_path_list.append(os.path.join(self.ct_root_path, ct_name))
        # 对于每例CT同时侧位与正位
        for single_ct_path in ct_path_list:
            self.gen_AP_drrs_and_masks(single_ct_path)
            self.gen_LA_drrs_and_masks(single_ct_path)
        # 保存json文件
        self.detection_dataset.to_json()


    def gen_AP_drrs_and_masks(self, single_ct_path):
        # get ct name
        ct_name = linux_windows_split_name(single_ct_path)
        filepaths = glob(os.path.join(single_ct_path, '*body_seg.nii.gz'))
        # 需呀将原CT放到路径开头,为了先生成整个CT的drr,再单独生成每节椎体的drr
        ct_filepath = os.path.join(single_ct_path, ct_name + '.nii.gz')
        filepaths.insert(0, ct_filepath)
        i = 0
        for rotation, translation in zip(self.AP_rotations, self.AP_translations):
            i += 1
            for filepath in filepaths:
                basename = os.path.basename(filepath)
                basename_wo_ext = basename[:basename.find('.nii.gz')]
                if "seg" not in basename_wo_ext:
                    self.gen_drr(ct_name, i, rotation, translation, filepath, "AP")
                else:
                    self.gen_mask(basename_wo_ext, ct_name, i, rotation, translation, filepath, "AP")


    def gen_LA_drrs_and_masks(self, single_ct_path):
        # get ct name
        ct_name = linux_windows_split_name(single_ct_path)
        filepaths = glob(os.path.join(single_ct_path, '*seg.nii.gz'))
        # 需要将原CT放到路径开头,为了先生成整个CT的drr,再单独生成每节椎体的drr
        ct_filepath = os.path.join(single_ct_path, ct_name + '.nii.gz')
        filepaths.insert(0, ct_filepath)
        i = 0
        for rotation, translation in zip(self.LA_rotations, self.LA_translations):
            i += 1
            for filepath in filepaths:
                basename = os.path.basename(filepath)
                basename_wo_ext = basename[:basename.find('.nii.gz')]
                if "seg" not in basename_wo_ext:
                    self.gen_drr(ct_name, i, rotation, translation, filepath, "LA")
                if len(basename_wo_ext.split("_")) == 2:
                    self.gen_mask(basename_wo_ext, ct_name, i, rotation, translation, filepath, "LA")


    def gen_drr(self, ct_name, i, rotation, translation, filepath, APorLA):
        drr_image_name = ct_name + "_" + str(i) + ".png"
        saveIMG = os.path.join(self.save_image_file, "images", drr_image_name)
        genDRR(self.sdr, self.height, self.delx, self.threshold, rotation, translation, filepath, saveIMG)
        # 侧位需要顺时针旋转90度
        if APorLA == "LA":
            rot_image(saveIMG)
        # add mask info to json
        width, height = cv2.imread(saveIMG).shape[:2]
        self.detection_dataset.add_image(drr_image_name, ct_name, APorLA, width, height, rotation, translation)


    def gen_mask(self, basename_wo_ext, ct_name, i, rotation, translation, filepath, APorLA):
        # get cur vertebrae name
        vertebrae_name = basename_wo_ext[:basename_wo_ext.find('seg')]
        mask_name = ct_name + "_" + vertebrae_name + str(i) + ".png"
        saveIMG = os.path.join(self.save_image_file, "masks", mask_name)
        # generate drr
        genDRR(self.sdr, self.height, self.delx, self.threshold, rotation, translation, filepath, saveIMG)
        # generate 2d mask
        gen_2D_mask(saveIMG)
        # 侧位需要顺时针旋转90度
        if APorLA == "LA":
            rot_image(saveIMG)
        # add annotaion info to json
        category_name = vertebrae_name.split("_")[0]
        category_id = self.detection_dataset.catname2catid[category_name]
        bbox = compute_min_bbox_coverage_mask(saveIMG)
        self.detection_dataset.add_annotation(category_id, bbox, iscrowd=0)
        

if __name__ == "__main__":
    pass
