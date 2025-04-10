import os
import subprocess
import multiprocessing

current_dir = '/defaultShare/FinialPDB'
def process_folder(folder, current_dir):
    # 构建文件夹的完整路径
    folder_path = os.path.join(current_dir, folder)
    print("folder_path",folder_path)
    out_path = os.path.join('/xiangyux/point/RIENet-main/data/ca_vo_data', folder)
  #  print("out_path",out_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        #print("out_path", out_path)


    # 确保路径是文件夹而不是文件
    if os.path.isdir(folder_path):
        # 遍历文件夹中的所有文件
        for file in os.listdir(folder_path):
            # 构建文件的完整路径
            file_path = os.path.join(folder_path, file)
            if file.endswith("pdb"):
                pdb_path = file_path

        print("out",out_path)
        print(folder_path,out_path)
        command = ["python", "/xiangyux/point/RIENet-main/data/ca_point.py", folder_path, out_path]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        # 打印输出
        print("stdout:", stdout.decode())
        print("stderr:", stderr.decode())

def main():
    current_dir = '/defaultShare/FinialPDB'  # 替换为实际路径
    print("current_dir",current_dir)
    folders = [folder for folder in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, folder))]
    with multiprocessing.Pool(16) as pool:
        pool.starmap(process_folder, [(folder, current_dir) for folder in folders])

    print("finished all data")
if __name__ == "__main__":
    main()