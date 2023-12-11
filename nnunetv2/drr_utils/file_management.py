'''
Descripttion: 
version: 
Author: ThreeStones1029 221620010039@qq.com
Date: 2023-12-10 14:04:34
LastEditors: ShuaiLei
LastEditTime: 2023-12-11 22:02:56
'''
import os
import platform

def create_folder(path):
    os.makedirs(path, exist_ok=True)
    return path

def linux_windows_split_name(path):
    if platform.system().lower() == "linux":
        name = path.split("/")[-1]
    else:
        name = path.split("\\")[-1]