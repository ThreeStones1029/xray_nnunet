'''
Descripttion: 
version: 
Author: ThreeStones1029 221620010039@qq.com
Date: 2023-12-10 14:04:34
LastEditors: ShuaiLei
LastEditTime: 2023-12-10 14:06:46
'''
import os


def create_folder(path):
    os.makedirs(path, exist_ok=True)
    return path