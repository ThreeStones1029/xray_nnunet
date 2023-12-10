'''
Descripttion: this file will be used to generate cut drrs and masks.
version: 1.0
Author: ThreeStones1029 221620010039@qq.com
Date: 2023-12-09 11:29:36
LastEditors: ShuaiLei
LastEditTime: 2023-12-10 14:52:42
'''
import os
import time
from gen_pedicle import GenPedicles
from gen_init_dataset import GenInitDRRMask
from gen_cut_dataset import cut_drrs_and_masks
from gen_json import DatasetJsonTools
from file_management import create_folder


class GEN_CUT_DRRMASK_DATASET:
    def __init__(self, data_root_path, has_pedicle, APorLA_orientation, is_vis):
        self.data_root_path = data_root_path
        self.ct_mask_path = os.path.join(self.data_root_path , "ct_mask")
        self.APorLA_orientation = APorLA_orientation
        self.data_path = os.path.join(self.data_root_path , self.APorLA_orientation)
        self.init_dataset_json_path = os.path.join(self.data_path, self.APorLA_orientation + "_init_dataset.json")
        self.cut_dataset_json_path = os.path.join(self.data_path, self.APorLA_orientation + "_cut_dataset.json")
        
        
        self.init_dataset_images_path = create_folder(os.path.join(self.data_path , "images"))
        self.init_dataset_masks_path = create_folder(os.path.join(self.data_path , "masks"))
        self.cut_dataset_images_path = create_folder(os.path.join(self.data_path , "cut_images"))
        self.cut_dataset_masks_path = create_folder(os.path.join(self.data_path , "cut_masks"))
        self.vis_path = create_folder(os.path.join(self.data_path , "vis"))
        
        self.has_pedicle = has_pedicle
        self.is_vis = is_vis
        

    
    def create_dataset(self):
        start_time = time.time()
        # 第一步:生成椎弓根seg文件,如果有就不需要了
        if not self.has_pedicle:
            sub_folder_names_list = os.listdir(self.ct_mask_path)
            for sub_folder_name in sub_folder_names_list:
                pedicle_nii =  GenPedicles()
                pedicle_nii.gen_pedicles(os.path.join(self.ct_mask_path, sub_folder_name))

        # 第二步:生成初始数据集
        init_dataset = GenInitDRRMask(ct_root_path=self.ct_mask_path, 
                                      APorLA_orientation=self.APorLA_orientation, 
                                      save_image_file=self.data_path,
                                      init_dataset_json_path=self.init_dataset_json_path)
        init_dataset.gen_multple_cts_drrs_and_masks()


        # 第三步:裁剪成最终数据集
        cut_drrs_and_masks(self.init_dataset_json_path,
                           self.init_dataset_images_path,
                           self.init_dataset_masks_path,
                           self.cut_dataset_images_path,
                           self.cut_dataset_masks_path)


        # 第四步: 是否需要生成可视化文件
        if self.is_vis:
            dataset_json_tools = DatasetJsonTools(self.init_dataset_json_path)
    
            dataset_json_tools.vis_masks(self.cut_dataset_images_path, self.cut_dataset_masks_path, self.vis_path)

        print("consume_time:", time.time() - start_time)

            
def main():
    dataset = GEN_CUT_DRRMASK_DATASET(data_root_path="data", has_pedicle=True, APorLA_orientation="AP", is_vis=True)
    dataset.create_dataset()


if __name__ == "__main__":
    main()