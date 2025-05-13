import os
import numpy as np
import csv
from typing import Union, Tuple
import open3d as o3d
from cvhelpers.open3d_helpers import to_o3d_pcd


# 加载keypoints但是点可能会太少
def load_xyz(file_path):
    pcd = o3d.io.read_point_cloud(file_path, format="xyz")
    return pcd


# 加载采样点,包含了密度信息和密度向量
def load_sample_points(file_path, density=False):
    point_list = []
    vector_list = []
    density_list = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        sample = float(lines[0].strip())
        origin_x, origin_y, origin_z = [float(i) for i in lines[3].strip().split()]
        for i in range(5, len(lines)):
            line = lines[i]
            if i % 2:
                _, x, y, z = line.strip().split()
                point_list.append(
                    [float(x) * sample + origin_x, float(y) * sample + origin_y, float(z) * sample + origin_z])
            else:
                v_x, v_y, v_z, d = line.strip().split()
                vector_list.append([float(v_x), float(v_y), float(v_z)])
                density_list.append([float(d)])
    if density:
        return np.array(point_list), np.array(vector_list), np.array(density_list)
    return np.array(point_list), np.array(vector_list)


def compute_overlap(src: Union[np.ndarray, o3d.geometry.PointCloud],
                    tgt: Union[np.ndarray, o3d.geometry.PointCloud],
                    search_voxel_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算两个点云之间的重叠区域，并返回对应信息。
    """
    try:
        if isinstance(src, np.ndarray):
            src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
            src_xyz = src
        else:
            src_pcd = src
            src_xyz = np.asarray(src.points)

        if isinstance(tgt, np.ndarray):
            tgt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt))
            tgt_xyz = tgt
        else:
            tgt_pcd = tgt
            tgt_xyz = np.asarray(tgt.points)

        tgt_corr = np.full(tgt_xyz.shape[0], -1, dtype=int)
        src_tree = o3d.geometry.KDTreeFlann(src_pcd)
        for i, t in enumerate(tgt_xyz):
            [k, idx, _] = src_tree.search_radius_vector_3d(t, search_voxel_size)
            if k > 0:
                tgt_corr[i] = idx[0]

        src_corr = np.full(src_xyz.shape[0], -1, dtype=int)
        tgt_tree = o3d.geometry.KDTreeFlann(tgt_pcd)
        for i, s in enumerate(src_xyz):
            [k, idx, _] = tgt_tree.search_radius_vector_3d(s, search_voxel_size)
            if k > 0:
                src_corr[i] = idx[0]

        src_indices = np.arange(len(src_corr))
        valid_src = src_corr >= 0
        mutual = np.zeros(len(src_corr), dtype=bool)
        mutual[valid_src] = (tgt_corr[src_corr[valid_src]] == src_indices[valid_src])
        src_tgt_corr = np.stack([np.nonzero(mutual)[0], src_corr[mutual]])
        has_corr_src = src_corr >= 0
        has_corr_tgt = tgt_corr >= 0

        return has_corr_src, has_corr_tgt, src_tgt_corr
    except Exception as e:
        print(f"计算重叠率时出错: {e}")
        return None, None, None
def find_source_and_target_files(folder_path, point_files):
    """ 根据文件名自动匹配源文件和目标文件 """
    source_file, target_file = None, None
    for file in point_files:
        file_path = os.path.join(folder_path, file)
        if "mol_new31.txt" in file.lower():
            source_file = file_path
        elif file.lower().endswith("_3.00.txt") and "_mol_3.00.txt" not in file.lower():
            target_file = file_path
    return source_file, target_file

def find_source_and_target_files(folder_path, point_files):
    """ 根据文件名自动匹配源文件和目标文件 """
    source_file, target_file = None, None
    for file in point_files:
        file_path = os.path.join(folder_path, file)
        if "mol_new3.txt" in file.lower():
            source_file = file_path
            print("source_file",source_file)
        elif file.lower().endswith("_3.00.txt") and "_mol_3.00.txt" not in file.lower():
            target_file = file_path
            print("target_file",target_file)
    return source_file, target_file

def find_and_compare_map_folders(parent_directory, output_csv):
    result_data = []
    i = 0
    for folder in os.listdir(parent_directory):
        folder_path = os.path.join(parent_directory, folder)
        i = i+1
        if os.path.isdir(folder_path):
            point_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
            if len(point_files) < 2:
                continue  # 跳过不完整的文件夹

            source_file, target_file = find_source_and_target_files(folder_path, point_files)
            if not source_file or not target_file:
                print(f"⚠️ {folder_path} 缺少 source 或 target 文件，跳过...")
                continue

            try:
                print(f"📂 处理: {folder_path}")
                print("num",i)
                sample_src_points, _ = load_sample_points(source_file)
                sample_tgt_points, _ = load_sample_points(target_file)

                src, tgt, src_tgtcorr = compute_overlap(sample_src_points, sample_tgt_points, 3)
                if src is None:
                    raise ValueError("重叠率计算失败")

                src_count, tgt_count = len(src), len(tgt)
                num_mutual = src_tgtcorr.shape[1] if src_tgtcorr is not None else 0
                src_ratio = num_mutual / src_count if src_count > 0 else 0
                tgt_ratio = np.sum(tgt) / tgt_count if tgt_count > 0 else 0

                result_data.append({
                    "Folder Path": folder_path,
                    "Source File": os.path.basename(source_file),
                    "Target File": os.path.basename(target_file),
                    "Source Points": src_count,
                    "Target Points": tgt_count,
                    "Mutual Correspondences": num_mutual,
                    "Source Ratio": src_ratio,
                    "Target Ratio": tgt_ratio,
                    "Low Overlap": "1" if (src_ratio < 0.2 and tgt_ratio < 0.2) else "0",
                    "Error": "无"
                })
            except Exception as e:
                print(f"❌ 计算失败: {folder_path}, 错误: {e}")
                result_data.append({
                    "Folder Path": folder_path,
                    "Source File": "N/A",
                    "Target File": "N/A",
                    "Source Points": "N/A",
                    "Target Points": "N/A",
                    "Mutual Correspondences": "N/A",
                    "Source Ratio": "N/A",
                    "Target Ratio": "N/A",
                    "Low Overlap": "N/A",
                    "Error": str(e)
                })

    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = ["Folder Path", "Source File", "Target File", "Source Points", "Target Points",
                      "Mutual Correspondences", "Source Ratio", "Target Ratio", "Low Overlap", "Error"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for data in result_data:
            writer.writerow(data)

    print(f"✅ 数据已保存至 {output_csv}")

if __name__ == "__main__":
    parent_dir = "/xiangyux/AF3temp/test"  # 请修改为实际的父目录路径
    output_csv_file = "/xiangyux/AF3temp/test/overlap.csv"  # 输出 CSV 文件路径

    find_and_compare_map_folders(parent_dir, output_csv_file)
    print("finshed😎")