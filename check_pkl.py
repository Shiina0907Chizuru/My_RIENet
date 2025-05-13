import pickle

# # 读取PKL文件
pkl_path = "C:\\Users\\Z\\Desktop\\3j9p_point_train.pkl"

# pkl_path = "E:\\ZJUT\\Research\\MrZhouDeepLearning\\点云配准pointnet\\My_RIENet\\6coz_point_target_c_train_c.pkl"
print(f"尝试读取文件: {pkl_path}")

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)
    
print("\n文件中的键:")
print(list(data.keys()))

print("\n每个键的详细信息:")
for key in data.keys():
    value = data[key]
    if hasattr(value, 'shape'):
        shape = value.shape
    elif hasattr(value, '__len__'):
        shape = f"长度: {len(value)}"
    else:
        shape = "无形状/长度信息"
        
    print(f"\n键名: {key}")
    print(f"类型: {type(value)}")
    print(f"形状/长度: {shape}")
    
    # 打印值的样本（限制长度，避免输出过多）
    if isinstance(value, (list, dict, str)):
        sample = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
        print(f"值样本: {sample}")
    elif hasattr(value, 'tolist') and callable(getattr(value, 'tolist')):
        try:
            sample = str(value.tolist()[:3]) + "..." if len(value.tolist()) > 3 else str(value.tolist())
            print(f"值样本: {sample}")
        except:
            print("无法获取值样本")
