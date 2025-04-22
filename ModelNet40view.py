import h5py
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ModelNet40Viewer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_h5_files = []
        self.test_h5_files = []
        self.id2file_mappings = {}
        self.shape_names = []
        self.data_cache = {}  # 缓存已加载的数据
        
        # 加载类别名称
        self.load_shape_names()
        
        # 加载训练集和测试集文件
        self.find_h5_files()
        
        # 加载id2file映射
        self.load_id2file_mappings()
        
    def load_shape_names(self):
        """加载类别名称"""
        shape_names_file = os.path.join(self.data_dir, "shape_names.txt")
        if os.path.exists(shape_names_file):
            with open(shape_names_file, 'r') as f:
                self.shape_names = [line.strip() for line in f.readlines()]
            print(f"成功加载 {len(self.shape_names)} 个类别名称")
        else:
            print("未找到类别名称文件，将使用数字标签")
    
    def find_h5_files(self):
        """查找所有H5文件"""
        for file in os.listdir(self.data_dir):
            if file.endswith(".h5"):
                if "train" in file:
                    self.train_h5_files.append(os.path.join(self.data_dir, file))
                elif "test" in file:
                    self.test_h5_files.append(os.path.join(self.data_dir, file))
        
        self.train_h5_files.sort()
        self.test_h5_files.sort()
        
        print(f"找到 {len(self.train_h5_files)} 个训练集文件和 {len(self.test_h5_files)} 个测试集文件")
    
    def load_id2file_mappings(self):
        """加载ID到文件的映射"""
        for file in os.listdir(self.data_dir):
            if file.endswith("id2file.json"):
                path = os.path.join(self.data_dir, file)
                with open(path, 'r') as f:
                    mapping = json.load(f)
                    self.id2file_mappings[file] = mapping
        
        print(f"加载了 {len(self.id2file_mappings)} 个ID到文件映射")
    
    def get_h5_data(self, h5_file):
        """从H5文件获取数据"""
        if h5_file in self.data_cache:
            return self.data_cache[h5_file]
        
        with h5py.File(h5_file, 'r') as f:
            data = f['data'][:]
            label = f['label'][:]
            # 有些H5文件还包含法线信息
            normal = f['normal'][:] if 'normal' in f else None
            
            self.data_cache[h5_file] = (data, label, normal)
            return data, label, normal
    
    def get_class_distribution(self, split='all'):
        """获取类别分布统计"""
        class_counts = np.zeros(len(self.shape_names) if self.shape_names else 40, dtype=int)
        
        files_to_check = []
        if split == 'train' or split == 'all':
            files_to_check.extend(self.train_h5_files)
        if split == 'test' or split == 'all':
            files_to_check.extend(self.test_h5_files)
        
        for h5_file in files_to_check:
            _, labels, _ = self.get_h5_data(h5_file)
            for label in labels:
                class_counts[label[0]] += 1
        
        return class_counts
    
    def get_samples_by_class(self, class_idx, split='all', max_samples=10):
        """获取指定类别的样本"""
        samples = []
        files_to_check = []
        
        if split == 'train' or split == 'all':
            files_to_check.extend(self.train_h5_files)
        if split == 'test' or split == 'all':
            files_to_check.extend(self.test_h5_files)
        
        for h5_file in files_to_check:
            data, labels, normals = self.get_h5_data(h5_file)
            for i, label in enumerate(labels):
                if label[0] == class_idx and len(samples) < max_samples:
                    sample_data = {
                        'points': data[i],
                        'file': os.path.basename(h5_file),
                        'index': i
                    }
                    if normals is not None:
                        sample_data['normal'] = normals[i]
                    samples.append(sample_data)
            
            if len(samples) >= max_samples:
                break
        
        return samples
    
    def visualize_point_cloud(self, points, title="点云可视化"):
        """可视化点云"""
        # 计算点云的统计信息
        num_points = points.shape[0]
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        center = np.mean(points, axis=0)
        point_range = max_coords - min_coords
        volume = point_range[0] * point_range[1] * point_range[2]
        density = num_points / volume if volume > 0 else 0
        
        # 在终端输出详细统计信息
        print("\n===== 点云详细统计信息 =====")
        print(f"点数量: {num_points}")
        print(f"坐标范围:")
        print(f"  X: {min_coords[0]:.6f} 到 {max_coords[0]:.6f}, 范围: {point_range[0]:.6f}")
        print(f"  Y: {min_coords[1]:.6f} 到 {max_coords[1]:.6f}, 范围: {point_range[1]:.6f}")
        print(f"  Z: {min_coords[2]:.6f} 到 {max_coords[2]:.6f}, 范围: {point_range[2]:.6f}")
        print(f"点云中心: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")
        print(f"点云体积: {volume:.6f}")
        print(f"点云密度: {density:.6f} 点/立方单位")
        print(f"点云最大绝对坐标值: {np.max(np.abs(points)):.6f}")
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制点云
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='b', alpha=0.8)
        
        # 设置坐标轴比例相等
        max_range = np.max(np.abs(points))
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        plt.tight_layout()
        plt.show()
    
    def interactive_browser(self):
        """交互式浏览器"""
        while True:
            print("\n==== ModelNet40 数据集浏览器 ====")
            print("1. 查看所有类别")
            print("2. 查看类别分布")
            print("3. 按类别浏览点云")
            print("4. 退出")
            
            choice = input("请输入选项 (1-4): ")
            
            if choice == '1':
                self.show_all_classes()
            elif choice == '2':
                self.show_class_distribution()
            elif choice == '3':
                self.browse_by_class()
            elif choice == '4':
                print("感谢使用，再见！")
                break
            else:
                print("无效选项，请重新输入")
    
    def show_all_classes(self):
        """显示所有类别"""
        if not self.shape_names:
            print("警告：未找到类别名称，将显示类别索引")
            classes = [f"类别 #{i}" for i in range(40)]
        else:
            classes = self.shape_names
        
        print("\n==== 所有类别 ====")
        for i, class_name in enumerate(classes):
            print(f"{i:2d}. {class_name}")
    
    def show_class_distribution(self):
        """显示类别分布"""
        split_choice = input("选择数据集 (train/test/all): ").lower()
        if split_choice not in ['train', 'test', 'all']:
            print("无效选择，默认使用'all'")
            split_choice = 'all'
        
        class_counts = self.get_class_distribution(split_choice)
        
        if not self.shape_names:
            classes = [f"类别 #{i}" for i in range(len(class_counts))]
        else:
            classes = self.shape_names
        
        print(f"\n==== {split_choice} 数据集的类别分布 ====")
        for i, (class_name, count) in enumerate(zip(classes, class_counts)):
            print(f"{i:2d}. {class_name}: {count} 个样本")
    
    def browse_by_class(self):
        """按类别浏览点云"""
        # 显示所有类别
        self.show_all_classes()
        
        # 获取用户选择的类别
        class_idx = int(input("\n请选择类别索引: "))
        if class_idx < 0 or class_idx >= (len(self.shape_names) if self.shape_names else 40):
            print("无效的类别索引")
            return
        
        # 获取数据集选择
        split_choice = input("选择数据集 (train/test/all): ").lower()
        if split_choice not in ['train', 'test', 'all']:
            print("无效选择，默认使用'all'")
            split_choice = 'all'
        
        # 获取样本数量
        try:
            max_samples = int(input("要显示多少个样本? (默认10): "))
        except ValueError:
            max_samples = 10
        
        # 获取指定类别的样本
        class_name = self.shape_names[class_idx] if self.shape_names else f"类别 #{class_idx}"
        print(f"\n正在获取 {class_name} 的 {max_samples} 个样本...")
        samples = self.get_samples_by_class(class_idx, split_choice, max_samples)
        
        if not samples:
            print(f"未找到 {class_name} 的样本")
            return
        
        print(f"找到 {len(samples)} 个样本")
        
        # 显示样本列表
        for i, sample in enumerate(samples):
            print(f"{i:2d}. 来自 {sample['file']} 的样本 #{sample['index']}")
        
        # 获取用户选择的样本
        sample_idx = int(input("\n请选择样本索引以可视化 (-1 返回): "))
        if sample_idx < 0 or sample_idx >= len(samples):
            return
        
        # 可视化选定的样本
        selected_sample = samples[sample_idx]
        title = f"{class_name} - {selected_sample['file']} 样本 #{selected_sample['index']}"
        self.visualize_point_cloud(selected_sample['points'], title)


if __name__ == "__main__":
    data_dir = r"C:\Users\Z\Desktop\modelnet40_ply_hdf5_2048"
    viewer = ModelNet40Viewer(data_dir)
    viewer.interactive_browser()