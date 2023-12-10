'''
Descripttion: 
version: 
Author: ThreeStones1029 221620010039@qq.com
Date: 2023-12-05 15:51:28
LastEditors: ShuaiLei
LastEditTime: 2023-12-10 14:10:26
'''
from ctypes import cdll, c_int, c_char_p, c_float


def linuxgenDRR(sdr, height, delx, rotation, translation, ctDir, saveIMG):
    # 绕X、Y和Z轴的旋转角度（以度为单位）
    rx = rotation[0]
    ry = rotation[1]
    rz = rotation[2]
    # 三维平移向量的分量
    tx = translation[0]
    ty = translation[1]
    tz = translation[2]

    # 旋转中心的坐标
    cx = 0
    cy = 0
    cz = 0

    # 源到成像平面的距离（即源与检测器之间的距离）
    sid = sdr * 2

    # DRR图像在X和Y轴上的像素间隔
    sx = delx
    sy = delx

    # DRR图像的宽度和高度（以像素为单位）
    dx = height
    dy = height

    # DRR图像的原点偏移量
    o2Dx = 0
    o2Dy = 0

    # 图像阈值
    # 较低的阈值：
    # 将保留更多原始 CT 数据中的像素，包括低密度区域。
    # 这可能导致生成的 DRR 图像中显示更多的细节，但也可能包含一些噪音或无关信息。
    # 较高的阈值：
    # 将过滤掉原始 CT 数据中的低密度区域，只保留高密度区域。
    # 这可能导致生成的 DRR 图像中显示更强的骨骼结构，但可能丢失一些低密度的软组织信息。
    threshold = -400

    # CT文件位置，要求.nii.gz
    ct_file_path = ctDir

    # DRR保存位置，后缀.png
    drr_save_path = saveIMG

    #加载cpp的itk生成drr共享库
    lib_itk = cdll.LoadLibrary("./Linux_ITK_Gen_Drr/build/libitk_drr.so")

    # 输入参数
    lib_itk.Generate_drr.argtypes = [c_float, c_float, c_float,
                        c_float, c_float, c_float,
                        c_float, c_float, c_float,
                        c_float, c_float, c_float,
                        c_int, c_int, c_float,
                        c_float,c_float, c_char_p, c_char_p]
    # 输入参数
    lib_itk.Generate_drr.restype = c_int

    lib_itk.Generate_drr(rx,ry, rz, tx, ty, tz, cx, cy, cz, sid, sx, sy, dx, dy, o2Dx, o2Dy, threshold, ct_file_path.encode('utf-8'), drr_save_path.encode('utf-8'))

if __name__ == "__main__":
    rotation = [
                2.095659358283953,
                86.68437641085228,
                -7.27514076304468
            ]
    translation = [
                5.640817832353026,
                -3.0283780712299047,
                5.985537664464523]

    sdr = 500
    delx = 0.25
    height = 1536
    ctDir = "data_test/ct_mask/cha_zhi_lin/cha_zhi_lin.nii.gz"
    saveIMG = "data_test/AP/images/cha_zhi_lin_1.png.png"
    linuxgenDRR(sdr, height, delx, rotation, translation, ctDir, saveIMG)