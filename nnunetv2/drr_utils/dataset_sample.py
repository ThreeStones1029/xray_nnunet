import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


class Dataset_sample:
    """用于生成数据集样本"""
    def __init__(self):
        self.rot_range_list = None
        self.trans_range_list = None
        self.num_samples = None

    # 网格法均匀采样
    def generate_uniform_samples_grid(self, rot_range_list, trans_range_list, num_samples):
        # 确定三维空间边界范围
        rot_x_min, rot_x_max = rot_range_list[0]
        rot_y_min, rot_y_max = rot_range_list[1]
        rot_z_min, rot_z_max = rot_range_list[2]
        tran_x_min, tran_x_max = trans_range_list[0]
        tran_y_min, tran_y_max = trans_range_list[1]
        tran_z_min, tran_z_max = trans_range_list[2]
        num_samples_per_axis = math.ceil(num_samples**(1/3))
        rotations = []
        translations = []
        for i in range(num_samples_per_axis):
            for j in range(num_samples_per_axis):
                for k in range(num_samples_per_axis):
                    # 在每个体素内生成样本点
                    rot_x = np.linspace(rot_x_min, rot_x_max, num_samples_per_axis, endpoint=False)[i]
                    rot_y = np.linspace(rot_y_min, rot_y_max, num_samples_per_axis, endpoint=False)[j]
                    rot_z = np.linspace(rot_z_min, rot_z_max, num_samples_per_axis, endpoint=False)[k]
                    tran_x = np.linspace(tran_x_min, tran_x_max, num_samples_per_axis, endpoint=False)[i]
                    tran_y = np.linspace(tran_y_min, tran_y_max, num_samples_per_axis, endpoint=False)[j]
                    tran_z = np.linspace(tran_z_min, tran_z_max, num_samples_per_axis, endpoint=False)[k]
                    rotations.append((rot_x, rot_y, rot_z))
                    translations.append((tran_x, tran_y, tran_z))

        return rotations, translations

    # 蒙特卡洛方法
    def Monte_Carlo_sample_dataset(self, rot_range_list, trans_range_list, num_samples):
        """
        rot_range_list:指定三个方向的旋转参数范围
        trans_range_list:指定三个方向的偏移大小范围
        rotation = [90, 180, 180], translation = [0, 0, 0]为正位
        num_samples:为生成的数量
        """
        rotations = []
        translations = []
        for _ in range(num_samples):
            rotation = []
            translation = []
            rotation.append(np.random.uniform(*rot_range_list[0]))
            rotation.append(np.random.uniform(*rot_range_list[1]))
            rotation.append(np.random.uniform(*rot_range_list[2]))
            rotations.append(rotation)
            translation.append(np.random.uniform(*trans_range_list[0]))
            translation.append(np.random.uniform(*trans_range_list[1]))
            translation.append(np.random.uniform(*trans_range_list[2]))
            translations.append(translation)

        return rotations, translations

    def show_3D_resample(self, samples):
        """
        可视化三维样本
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 提取样本中的 x、y、z 坐标
        x_coords = [sample[0] for sample in samples]
        y_coords = [sample[1] for sample in samples]
        z_coords = [sample[2] for sample in samples]

        # 绘制散点图
        ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o')
        # 设置图形标题和坐标轴标签
        ax.set_title('Uniform Samples Visualization')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        # 显示图形
        plt.show()


if __name__ == "__main__":
    num_samples_list = [216, 216] # 正位数量， 侧位数量
    rot_range_list = [[(60, 120), (150, 210), (150, 210)], [(-30, 30), (60, 120), (-30, 30)]]
    trans_range_list = [[(-30, 30), (-30, 30), (-30, 30)], [(-30, 30), (-30, 30), (-30, 30)]]

    dataset_sample = Dataset_sample()
    
    # 网格法
    rotations_front1, translations_front1 = dataset_sample.generate_uniform_samples_grid(rot_range_list[0], trans_range_list[0], num_samples_list[0])
    rotations_side1, translations_side1 = dataset_sample.generate_uniform_samples_grid(rot_range_list[1], trans_range_list[1], num_samples_list[1])
    dataset_sample.show_3D_resample(rotations_front1 + rotations_side1)
    

    # 蒙特卡洛方法生成随机数，但不均匀
    rotations_front2, translations_front2 = dataset_sample.Monte_Carlo_sample_dataset(rot_range_list[0], trans_range_list[0], num_samples_list[0])
    rotations_side2, translations_side2 = dataset_sample.Monte_Carlo_sample_dataset(rot_range_list[1], trans_range_list[1], num_samples_list[1])
    dataset_sample.show_3D_resample(rotations_front2 + rotations_side2)