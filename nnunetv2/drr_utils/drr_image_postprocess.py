'''
Descripttion: 
version: 
Author: ShuaiLei
Date: 2023-12-05 15:49:06
LastEditors: ShuaiLei
LastEditTime: 2023-12-08 21:12:11
'''
import cv2 as cv
from json_postprocess import coco_annotations


def flipdrr(image_path):
    image = cv.imread(image_path)
    h, w, channels = image.shape[0:3]
    for row in range(h):
        for col in range(w):
            for c in range(channels):
                pixel = image[row, col, c]
                image[row, col, c] = 255-pixel
    cv.imwrite(image_path, image)


def gen_2D_mask(img_path):
    image = cv.imread(img_path)
    threshold_value = 0
    _, threshold_image = cv.threshold(image, threshold_value, 255, cv.THRESH_BINARY)
    cv.imwrite(img_path, threshold_image)


def rot_image(image_path):
    image = cv.imread(image_path)
    imageR270 = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    cv.imwrite(image_path, imageR270)
 

if __name__ == "__main__":
    # rot_image("data/LA/images/cha_zhi_lin_1.png")
    gen_2D_mask("Image_01L.png")