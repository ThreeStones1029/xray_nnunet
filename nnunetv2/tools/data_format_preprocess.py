'''
Descripttion: 
version: 
Author: ShuaiLei
Date: 2023-11-27 21:50:16
LastEditors: ShuaiLei
LastEditTime: 2023-11-28 18:27:59
'''
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


def rename_images(images_folder):
    for root, dirs, files in os.walk(images_folder):
        for file in files:
            new_filename = file.split('_')[0] + '_' + file.split('_')[1] + ".png"
            os.rename(os.path.join(root, file),os.path.join(root, new_filename))



if __name__ == "__main__":
    # data_format_conversion = DataFormatConversion(images_folder="/home/jjf/Downloads/CHASEDB1",
    #                                               save_images_folder="/home/jjf/Downloads/CHASEDB1")
    # data_format_conversion.jpgs2pngs()

    rename_images("nnunetv2/nnUNet_raw/CHASEDB1/train/masks")
    rename_images("nnunetv2/nnUNet_raw/CHASEDB1/val/masks")

    
