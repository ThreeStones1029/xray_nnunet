'''
Description: 
version: 
Author: ThreeStones1029 221620010039@qq.com
Date: 2023-12-07 21:28:15
LastEditors: ShuaiLei
LastEditTime: 2023-12-09 21:06:09
'''
from datetime import datetime
import json
import os
from collections import defaultdict
import cv2
import numpy as np
from matplotlib import pyplot as plt
import re



class InitDatasetJson:
    def __init__(self, init_dataset_json_path):
        '''
        description: 
        self.dataset 初始数据集
        self.dataset_info 初始数据集基本信息
        self.dataset_images 初始数据集drr
        self.dataset_masks 初始数据集mask
        self.dataset_masks_categories 初始数据集类别
        self.dataset_cts 投影使用的ct信息
        self.images_num 记录drr数量
        self.masks_num 记录masks数量
        self.catname2catid mask类别名字到类别id映射
        self.catid2catname mask类别id到类别名字映射
        self.init_dataset_json_path 生成的json文件保存路径
        return {*}
        '''
        self.dataset = dict()
        self.dataset_info = dict()
        self.dataset_images = []
        self.dataset_masks = []
        self.dataset_masks_categories = []
        self.dataset_cts = []
        self.cts_num = 0
        self.images_num = 0
        self.masks_num = 0
        self.catname2catid = dict()
        self.catid2catname = dict()
        self.init_dataset_json_path = init_dataset_json_path


    def gen_init_dataset_json(self):
        self.dataset["info"] = self.dataset_info
        self.dataset["images"] = self.dataset_images
        self.dataset["masks"] = self.dataset_masks
        self.dataset["cts"] = self.dataset_cts
        self.dataset["masks_categories"] = self.dataset_masks_categories


    def add_info(self, sdr, height, delx):

        self.dataset_info["dataset_info"] = {"description": "drrs and masks dataset",
                                             "version": "1.0",
                                             "Author": "ThreeStones1029",
                                             "Date": datetime.today().strftime('%Y-%m-%d')}
        
        self.dataset_info["projection_parameter"] = {"sdr": sdr,
                                                     "height": height,
                                                     "delx": delx}
        
        
    def add_image(self, image_name,ct_name, AP_or_LA, width, height, rotation, translation):
        image_info = {}
        self.images_num  = self.images_num + 1
        image_info["id"] = self.images_num
        image_info["image_name"] = image_name
        image_info["ct_id"] = self.cts_num
        image_info["ct_name"] = ct_name + ".nii.gz"
        image_info["AP_or_LA"] = AP_or_LA
        image_info["width"] = width
        image_info["height"] = height
        image_info["rotation"] = rotation
        image_info["translation"] = translation

        self.dataset_images.append(image_info)


    def add_mask(self, mask_name, AP_or_LA, width, height, rotation, translation):
        mask_info = {}
        self.masks_num = self.masks_num + 1
        mask_info["id"] = self.masks_num
        mask_info["mask_name"] = mask_name
        mask_info["image_id"] = self.images_num 
        basename = os.path.basename(mask_name)
        mask_category_name = basename.split("_")[-2]
        mask_info["mask_category_name"] = mask_category_name
        mask_info["mask_category_id"] = self.catname2catid[mask_category_name]
        mask_info["vertebrae_category_name"] = basename.split("_")[-3]
        mask_info["AP_or_LA"] = AP_or_LA
        mask_info["width"] = width
        mask_info["height"] = height
        mask_info["rotation"] = rotation
        mask_info["translation"] = translation
        
        self.dataset_masks.append(mask_info)


    def add_ct(self, ct_name):
        ct_info = {}
        self.cts_num = self.cts_num + 1
        ct_info['id'] = self.cts_num
        ct_info['ct_name'] = ct_name + ".nii.gz"
        ct_info["vertebrae_categoties"] = []
        self.dataset_cts.append(ct_info)


    def add_ct_vertebrae_categoties(self, mask_name):
        vertebrae_category_name = os.path.basename(mask_name).split("_")[-3]
        if vertebrae_category_name not in self.dataset_cts[self.cts_num - 1]["vertebrae_categoties"]:
            self.dataset_cts[self.cts_num - 1]["vertebrae_categoties"].append(vertebrae_category_name)

        

    def add_masks_categories(self):
        self.dataset_masks_categories = [
                                        {"id": 1,
                                        "name": "body",
                                        "supercategory": "vertebrae"},
                                        {"id": 2,
                                        "name": "pedicle",
                                        "supercategory": "vertebrae"},
                                        {"id": 3,
                                        "name": "other",
                                        "supercategory": "vertebrae"}
                                        ]
        
        for category in self.dataset_masks_categories:
            self.catid2catname[category["id"]] = category["name"]
            self.catname2catid[category["name"]] = category["id"]
            
        self.dataset["masks_categories"] = self.dataset_masks_categories

    
    def save_dataset(self):
        with open(self.init_dataset_json_path, "w", encoding='utf-8') as f:
            json.dump(self.dataset, f)


class DatasetJsonTools:
    def __init__(self, dataset_file):
        self.Imageid2Masks = defaultdict(list)
        self.Imageid2Image = defaultdict(list)
        
        if type(dataset_file) is str:
            with open(dataset_file, "r") as f:
                self.dataset = json.load(f)
        else:
            print("please check if " , dataset_file, " is path")

        self.gen_Imageid2Image()
        self.gen_Imageid2Masks()

    
    def gen_Imageid2Image(self):
        for image_info in self.dataset["images"]:
            self.Imageid2Image[image_info["id"]].append(image_info)
        

    def gen_Imageid2Masks(self):
        """
        this function will be used to gen mapping which image id to masks
        """
        for mask_info in self.dataset["masks"]:
            
            self.Imageid2Masks[mask_info["image_id"]].append(mask_info)


    def get_vertebrae_name2masks(self, image_id):
        Vertebraename2Masks = defaultdict(list)
        for mask_info in self.Imageid2Masks[image_id]:
            Vertebraename2Masks[mask_info["vertebrae_category_name"]].append(mask_info)
        return Vertebraename2Masks


    def vis_mask(self, image_id, cat_name, drr_images_path, masks_path, vis_save_path):
        Vertebraename2Masks = self.get_vertebrae_name2masks(image_id)

        single_cat_masks = Vertebraename2Masks[cat_name]
        
        if image_id % (len(self.dataset["images"])/ len(self.dataset["cts"])) == 0:
            id = int(len(self.dataset["images"])/ len(self.dataset["cts"]))
        else:
            id = int(image_id % (len(self.dataset["images"])/ len(self.dataset["cts"])))

        vis_img_name = self.Imageid2Image[image_id][0]["ct_name"].split(".")[0] + "_" +  cat_name + "_" + str(id) + ".png"
        
        
        drr = cv2.imread(os.path.join(drr_images_path, vis_img_name))
        merge_img = np.copy(drr)
        color_list = [[0, 255, 0], [0, 0, 255], [255, 0, 0]]

        for i in range(len(single_cat_masks)):
            mask = cv2.imread(os.path.join(masks_path, single_cat_masks[i]["mask_name"]), cv2.IMREAD_GRAYSCALE)
            merge_img[mask > 0] = color_list[i]
        
        print("vis", os.path.join(drr_images_path, vis_img_name), "successfully")

        cv2.imwrite(os.path.join(vis_save_path, vis_img_name), merge_img)

        # show
        # merge_img = cv2.cvtColor(merge_img,cv2.COLOR_BGR2RGB)
        # #matplotlib显示图像
        # plt.imshow(merge_img)
        # plt.show()


    # def extract_prefix(self, file_name):
    #     # 使用正则表达式提取文件名中的字符部分
    #     match = re.match(r'^([a-zA-Z_]+)\d*\.png$', file_name)
        
    #     if match:
    #         # 返回匹配到的字符部分
    #         return match.group(1)
    #     else:
    #         # 如果未匹配到，可以根据需要返回一个默认值或者抛出异常等
    #         return None


    def vis_masks(self, drr_images_path, masks_path, vis_save_path):
        for image_id in self.Imageid2Masks.keys():
        
            Vertebraename2Masks = self.get_vertebrae_name2masks(image_id)
            
            for cat_name in Vertebraename2Masks.keys():
                self.vis_mask(image_id, cat_name, drr_images_path, masks_path, vis_save_path)


if __name__ == "__main__":
    # init_dataset = InitDatasetJson("data/LA/LA_init_dataset.json")
    # init_dataset.add_info(500, 10, 2)
    # init_dataset.add_masks_categories()
    # init_dataset.gen_init_dataset_json()
    #     # 保存json文件
    # init_dataset.save_dataset()
    init_dataset = DatasetJsonTools("data/LA/LA_init_dataset.json")
    

    init_dataset.vis_masks("data/LA/cut_images", "data/LA/cut_masks", "data/LA/vis")