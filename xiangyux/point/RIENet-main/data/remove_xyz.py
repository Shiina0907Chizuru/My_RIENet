import os
import shutil

def move_pkl_files(source_dir, target_dir):
    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录及其子目录
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.pkl'):  # 只选中.pkl文件
                # 获取文件的完整路径
                source_file = os.path.join(root, file)
                # 获取目标文件的路径
                target_file = os.path.join(target_dir, file)
                
                # 检查目标文件夹中是否已有同名文件，如果有可以选择重命名或覆盖
                if os.path.exists(target_file):
                    # 选择覆盖，或可以改为重命名
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    os.remove(target_file)

                # 移动文件到目标文件夹
                shutil.move(source_file, target_file)
                print(f"Moved: {source_file} -> {target_file}")

# 使用示例
source_directory = '/xiangyux/point/point_data'  # 替换为源文件夹路径
target_directory = '/xiangyux/point/RegTR-main/src/datasets/ca_point/train'  # 替换为目标文件夹路径

move_pkl_files(source_directory, target_directory)
