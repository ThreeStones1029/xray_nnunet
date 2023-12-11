'''
Descripttion: 
version: 
Author: ThreeStones1029 221620010039@qq.com
Date: 2023-12-08 20:24:16
LastEditors: ShuaiLei
LastEditTime: 2023-12-11 22:09:42
'''
from gen_json import DatasetJsonTools
import numpy as np
import cv2
import os


def cut_drrs_and_masks(init_dataset_json_path, init_drrs_path, init_masks_path, cut_drrs_save_path, cut_masks_save_path):

    '''
    description: 
    init_dataset_json_path
    init_drrs_path
    init_masks_path
    cut_drrs_save_path
    cut_masks_save_path
    return {*}
    '''
    dataset_json_tools = DatasetJsonTools(init_dataset_json_path)
    for image_id in dataset_json_tools.Imageid2Masks.keys():
        # debug use
        # if image_id != 1:
        #     continue
        Vertebraename2Masks = dataset_json_tools.get_vertebrae_name2masks(image_id)
        drr_image_info = dataset_json_tools.Imageid2Image[image_id][0]
        for cat_name in Vertebraename2Masks.keys():
            # debug use
            # if cat_name != "L2":
            #     continue
            drr_image = cv2.imread(os.path.join(init_drrs_path, drr_image_info["image_name"]))
            all_mask = cv2.imread(os.path.join(init_masks_path, Vertebraename2Masks[cat_name][0]["mask_name"]), cv2.IMREAD_GRAYSCALE)
            for i in range(1, len(Vertebraename2Masks[cat_name])):
                mask = cv2.imread(os.path.join(init_masks_path, Vertebraename2Masks[cat_name][i]["mask_name"]), cv2.IMREAD_GRAYSCALE)
                all_mask = all_mask + mask
            # 确保叠加后还是0, 255
            all_mask = np.clip(all_mask, 0, 255)
            bbox = compute_min_bbox_coverage_mask(all_mask)
            # 扩大了适当倍 侧位设置为1.2,正位为1.5倍
            if Vertebraename2Masks[cat_name][0]["AP_or_LA"] == "LA":
                cut_bbox_coordinate = get_cut_bbox(bbox, all_mask.shape[:2][0], all_mask.shape[:2][1], 1.2)
            else:
                cut_bbox_coordinate = get_cut_bbox(bbox, all_mask.shape[:2][0], all_mask.shape[:2][1], 1.5)
            # 裁剪
            drr_image_name = drr_image_info["image_name"]
            separate_name_list = drr_image_name.split("_")
            cut_drr_image_name = ""
            for i in range(len(separate_name_list) - 1):
                if i == 0:
                    cut_drr_image_name = cut_drr_image_name + separate_name_list[i]
                else:
                    cut_drr_image_name = cut_drr_image_name + "_" + separate_name_list[i]
            cut_drr_image_name = cut_drr_image_name + "_" + cat_name + "_" + separate_name_list[-1]
            cut_drr_image = drr_image[cut_bbox_coordinate[1]: cut_bbox_coordinate[3], cut_bbox_coordinate[0]: cut_bbox_coordinate[2]]
            cv2.imwrite(os.path.join(cut_drrs_save_path, cut_drr_image_name), cut_drr_image)
            print("cut", os.path.join(cut_drrs_save_path, cut_drr_image_name), "save successfully")
            for i in range(len(Vertebraename2Masks[cat_name])):
                mask = cv2.imread(os.path.join(init_masks_path, Vertebraename2Masks[cat_name][i]["mask_name"]), cv2.IMREAD_GRAYSCALE)
                cut_mask = mask[cut_bbox_coordinate[1]: cut_bbox_coordinate[3], cut_bbox_coordinate[0]: cut_bbox_coordinate[2]]
                cv2.imwrite(os.path.join(cut_masks_save_path, Vertebraename2Masks[cat_name][i]["mask_name"]), cut_mask)
                print("cut", os.path.join(cut_masks_save_path, Vertebraename2Masks[cat_name][i]["mask_name"]), "save successfully")

    
def compute_min_bbox_coverage_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_bbox = cv2.boundingRect(contours[0])
    return min_bbox


def according_detection():
    pass


def get_cut_bbox(bbox, width, heigth, expand_coefficient):
    x, y, w, h = bbox
    center_x, center_y = x +w/2, y + h/2
    expand_w = expand_coefficient * w
    expand_h = expand_coefficient * h
    new_min_x = center_x - expand_w / 2 if center_x - expand_w / 2 > 0 else 0
    new_min_y = center_y - expand_h / 2 if center_y - expand_h / 2 > 0 else 0
    new_max_x = center_x + expand_w / 2 if center_x + expand_w / 2 < width else width
    new_max_y = center_y + expand_h / 2 if center_y + expand_h / 2 < heigth else heigth
    return [int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)]


if __name__ == "__main__":
    cut_drrs_and_masks("data/LA/LA_init_dataset.json",
                       "data/LA/images",
                       "data/LA/masks",
                       "data/LA/cut_images",
                       "data/LA/cut_masks")