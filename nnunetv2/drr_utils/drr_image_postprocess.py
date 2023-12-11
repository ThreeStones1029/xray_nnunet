'''
Descripttion: 
version: 
Author: ShuaiLei
Date: 2023-12-05 15:49:06
LastEditors: ShuaiLei
LastEditTime: 2023-12-11 21:12:02
'''
import cv2
from json_postprocess import coco_annotations


def flipdrr(image_path):
    image = cv2.imread(image_path)
    h, w, channels = image.shape[0:3]
    for row in range(h):
        for col in range(w):
            for c in range(channels):
                pixel = image[row, col, c]
                image[row, col, c] = 255-pixel
    cv2.imwrite(image_path, image)


def gen_2D_mask(img_path):
    image = cv2.imread(img_path)
    threshold_value = 0
    _, threshold_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imwrite(img_path, threshold_image)


def rot_image(image_path):
    image = cv2.imread(image_path)
    imageR270 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(image_path, imageR270)


def compute_min_bbox_coverage_mask(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_bbox = cv2.boundingRect(contours[0])

    return min_bbox
 

if __name__ == "__main__":
    # rot_image("data/LA/images/cha_zhi_lin_1.png")
    gen_2D_mask("Image_01L.png")