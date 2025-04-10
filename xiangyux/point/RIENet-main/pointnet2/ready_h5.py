import h5py

# 设置文件路径
file_path = '/xiangyux/point/RIENet-main/pointnet2/test/modelnet_clean.h5'

# 打开文件
with h5py.File(file_path, 'r') as f:
    # 获取文件中的所有键
    print("文件中的内容:", list(f.keys()))
    
    # 读取数据集
    label = f['label'][:]
    source = f['source'][:]
    target = f['target'][:]
    transform = f['transform'][:]
    
    # 输出数据的形状或部分内容
    print("label 数据:", label.shape)  # 或者 print(label[:5]) 查看前5个
    print("source 数据:", source.shape)  # 或者 print(source[:5]) 查看前5个
    print("target 数据:", target.shape)  # 或者 print(target[:5]) 查看前5个
    print("transform 数据:", transform.shape)  # 或者 print(transform[:5]) 查看前5个
