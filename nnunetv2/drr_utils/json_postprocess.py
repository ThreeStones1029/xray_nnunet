'''
Descripttion: this file will be used to fliter box
version: 
Author: ShuaiLei
Date: 2023-12-07 16:10:44
LastEditors: ShuaiLei
LastEditTime: 2023-12-07 21:28:26
'''
import json
from collections import defaultdict
import time


class coco_annotations:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.cat_id2cat_name = dict()
        print('loading annotations into memory...')
        tic = time.time()
        # 通过路径下载
        if type(annotation_file) == str:
            with open(annotation_file, 'r') as f:
                dataset = json.load(f)
            assert type(dataset)==list, 'annotation file format {} not supported'.format(type(dataset))
            self.dataset = {"annotations": dataset}
        # 直接加载list
        elif type(annotation_file) == list:
            self.dataset = {"annotations": annotation_file}
        else:
            print("annotation_file must be path or list")

        self.createIndex()
        print('Done (t={:0.2f}s)'.format(time.time()- tic))


    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs, img_idToFilename = {}, {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                img_idToFilename[ann['image_id']] = ann['file_name']

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.img_idToFilename = img_idToFilename
        self.cats = cats

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        if 'info' in self.dataset:
            for key, value in self.dataset['info'].items():
                print('{}: {}'.format(key, value))
        else:
            print("dataset don't have info, please check your json file")


    def gen_cat_id2cat_name(self):
        for ann in self.dataset["annotations"]:
            if ann["category_id"] not in self.cat_id2cat_name.keys():
                self.cat_id2cat_name[ann["category_id"]] = ann["category_name"]

        return self.cat_id2cat_name


def fliter_bbox(detection_result_json, flitered_result_json):
    '''
    description: 
    param {*} detection_result_json
    param {*} flitered_result_json
    return {*}
    '''
    flitered_result = []
    with open(detection_result_json, "r") as f:
        detection_result = json.load(f)

    for ann in detection_result:
        if ann["score"] > 0.6:
            flitered_result.append(ann)

    with open(flitered_result_json, "w") as f:
        json.dump(flitered_result, f)


if __name__ == "__main__":
    fliter_bbox("./data/AP/detection_results/bbox.json", "./data/AP/detection_results/flitered_bbox.json")