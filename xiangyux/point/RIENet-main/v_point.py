import open3d as o3d
import numpy as np
import pickle
import copy


# 定义一个函数，应用旋转矩阵和平移矩阵
def apply_transformation(point_cloud, rotation_matrix, translation_vector):
    # 获取点云数据
    points = np.asarray(point_cloud.points)
    # 应用旋转矩阵和平移向量
    points = np.dot(points, rotation_matrix.T) + translation_vector
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud


# 读取 .pkl 文件并提取数据
def load_pkl_data(pkl_file):
    with open(pkl_file, 'rb') as f:
        data_dict = pickle.load(f)

    source_coords = data_dict['source']  # 源点云
    transformed_coords = data_dict['target']  # 目标点云
    rotation_matrix = data_dict['rotation']  # 旋转矩阵
    translation_vector = data_dict['translation']  # 平移向量

    return source_coords, transformed_coords, rotation_matrix, translation_vector


# 加载点云并进行可视化
def visualize_point_clouds_with_transformation(pkl_file):
    # 读取 .pkl 文件的数据
    source_coords, transformed_coords, rotation_matrix, translation_vector = load_pkl_data(pkl_file)

    # 创建 Open3D 点云对象
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_coords)

    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_coords)

    # 可视化点云
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加源点云和变换后的点云到可视化窗口
    vis.add_geometry(source_pcd)
    vis.add_geometry(transformed_pcd)

    # 应用旋转矩阵和平移向量到源点云
    source_pcd_transformed = apply_transformation(copy.deepcopy(source_pcd), rotation_matrix, translation_vector)
    vis.add_geometry(source_pcd_transformed)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    # 示例文件，替换为你实际的 .pkl 文件路径
    pkl_file = '/xiangyux/point/RIENet-main/data/ca_test/6adq.pC_point.pkl'  # 替换为实际文件路径

    visualize_point_clouds_with_transformation(pkl_file)
