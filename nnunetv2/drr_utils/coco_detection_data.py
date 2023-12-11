'''
Description: 
version: 
Author: ThreeStones1029 221620010039@qq.com
Date: 2023-12-11 11:21:50
LastEditors: ShuaiLei
LastEditTime: 2023-12-11 21:16:23
'''
import json 
from datetime import datetime


class COCODetectionData:
    def __init__(self):
        self.info = {
            "description": "Spine Detection DataSet",
            "url": "https://github.com/ThreeStones1029",
            "version": "1.0",
            "year": 2023,
            "contributor": "ShuaiLei",
            "Date": datetime.today().strftime('%Y-%m-%d')
        }
        self.annotation_num = 0
        self.image_num = 0
        self.images = []
        self.categories = []
        self.annotations = []
        self.catid2catname = dict()
        self.catname2catid = dict()

    def add_image(self, file_name, ct_name, APorLA, width, height, rotation, translation):
        self.image_num += 1
        image = {
            "id": self.image_num,
            "width": width,
            "height": height,
            "file_name": file_name,
            "ct_name": ct_name,
            "APorLA": APorLA,
            "rotation": rotation,
            "translation": translation
        }
        self.images.append(image)

    def add_categories(self):
        self.categories = [
                            {"id": 1,
                            "name": "L5",
                            "supercategory": "vertebrae"},
                            {"id": 2,
                            "name": "L4",
                            "supercategory": "vertebrae"},
                            {"id": 3,
                            "name": "L3",
                            "supercategory": "vertebrae"},
                            {"id": 4,
                            "name": "L2",
                            "supercategory": "vertebrae"},
                            {"id": 5,
                            "name": "L1",
                            "supercategory": "vertebrae"},
                            {"id": 6,
                            "name": "T12",
                            "supercategory": "vertebrae"},
                            {"id": 7,
                            "name": "T11",
                            "supercategory": "vertebrae"},
                            {"id": 8,
                            "name": "T10",
                            "supercategory": "vertebrae"},
                            {"id": 9,
                            "name": "T9",
                            "supercategory": "vertebrae"}
                            ]
        
        for category in self.categories:
            self.catid2catname[category["id"]] = category["name"]
            self.catname2catid[category["name"]] = category["id"]
        

    def add_annotation(self, category_id, bbox, iscrowd=0):
        self.annotation_num += 1
        annotation = {
            "id": self.annotation_num,
            "image_id": self.image_num,
            "category_id": category_id,
            "area": bbox[2] * bbox[3],
            "bbox": bbox,
            "iscrowd": iscrowd
        }
        self.annotations.append(annotation)

    def to_json(self, save_path):
        coco_data = {
            "info": self.info,
            "images": self.images,
            "categories": self.categories,
            "annotations": self.annotations
        }

        with open(save_path, 'w') as json_file:
            json.dump(coco_data, json_file)