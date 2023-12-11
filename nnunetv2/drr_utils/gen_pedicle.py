'''
Descripttion: this file will be used to gen pecidle.nii.gz
version: 
Author: ShuaiLei
Date: 2023-12-05 16:24:26
LastEditors: ShuaiLei
LastEditTime: 2023-12-11 22:10:01
'''
import nibabel as nib
import os
import glob


class GenPedicles:
    def __init__(self):
        self.catnames = []


    def load_nii(self, nii_path):
        if os.path.exists(nii_path):
            data = nib.load(nii_path)
            return data
        else:
            print("please check ", os.path.basename(nii_path), " in ", os.path.abspath(os.path.join(nii_path, "..")))
    

    def save_nii(self, data, save_path):
        nib.save(data, save_path)
        print("data successfully save in ", save_path)


    def get_catname_list(self, vertebraes_path):
        files_path = glob.glob(os.path.join(vertebraes_path, "*seg.nii.gz"))
        for file_path in files_path:
            filename = os.path.basename(file_path)
            catname = filename.split("_")[0]
            if catname not in self.catnames:
                self.catnames.append(catname)


    def gen_pedicle(self, vertebrae_path, vertebrae_catname):
        """
        vertebrae_path: 单个椎体路径
        vertebrae_catname:椎体类别名
        """
        vertebrae = self.load_nii(os.path.join(vertebrae_path, vertebrae_catname + "_all_seg.nii.gz"))
        body = self.load_nii(os.path.join(vertebrae_path, vertebrae_catname + "_body_seg.nii.gz"))
        other = self.load_nii(os.path.join(vertebrae_path, vertebrae_catname + "_other_seg.nii.gz"))
        vertebrae_data = vertebrae.get_fdata()
        body_data = body.get_fdata()
        other_data = other.get_fdata()
        # 椎弓根 = 整体 - body - other
        pedicle_data = vertebrae_data - body_data - other_data
        # 用nii保存,同时需要拷贝原来的坐标系位置
        pedicle = nib.Nifti1Image(pedicle_data, affine=vertebrae.header.get_best_affine())
        self.save_nii(pedicle, os.path.join(vertebrae_path, vertebrae_catname + "_pedicle_seg.nii.gz"))


    def gen_pedicles(self, vertebraes_path):
        """
        vertebraes_path: 多个椎体路径
        """
        self.get_catname_list(vertebraes_path)
        print(self.catnames)
        for catname in self.catnames:
            self.gen_pedicle(vertebraes_path, catname)


if __name__ == "__main__":
    gen_dataset = GenPedicles()
    gen_dataset.gen_pedicles("data/ct_mask/cha_zhi_lin")