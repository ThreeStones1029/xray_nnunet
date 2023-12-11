'''
Descripttion: 
version: 
Author: ThreeStones1029 221620010039@hhu.edu.cn
Date: 2023-12-10 14:12:45
LastEditors: ShuaiLei
LastEditTime: 2023-12-10 21:49:42
'''
import subprocess


# 调用ITK
def windowsgenDRR(sdr, height, delx, threshold, rotation, translation, ctDir, saveIMG):
    # 默认参数
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

    threshold = threshold
    ctDir = ctDir
    drrSave = saveIMG

    # 参数
    args = [
        "./Windows_ITK_Gen_Drr/itk_dll/ITK_TOOLS.exe",
        str(rx),
        str(ry),
        str(rz),
        str(tx),
        str(ty),
        str(tz),
        str(cx),
        str(cy),
        str(cz),
        str(sid),
        str(sx),
        str(sy),
        str(dx),
        str(dy),
        str(o2Dx),
        str(o2Dy),
        str(threshold),
        ctDir,
        drrSave
    ]

    subprocess.run(args)
