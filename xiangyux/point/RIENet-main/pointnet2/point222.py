import torch
from torch.autograd import Function
import pointnet2_cuda as pointnet2

# Furthest Point Sampling
class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance.
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        # 调用 pointnet2 的 CUDA 扩展函数进行最远点采样
        pointnet2.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None

# 测试 FurthestPointSampling 和 PointNet2 是否正确安装和运行
def test_pointnet2_fps():
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    print("CUDA is available!")

    # 创建一个假设的点云数据 (B, N, 3)
    batch_size = 2
    num_points = 1024
    xyz = torch.randn(batch_size, num_points, 3).cuda()  # 随机生成点云数据并将其移动到 GPU

    # 设定采样点数
    npoint = 128

    # 使用自定义的最远点采样函数进行测试
    try:
        sampled_points = FurthestPointSampling.apply(xyz, npoint)
        print(f"Furthest Point Sampling successful! Sampled points shape: {sampled_points.shape}")
    except Exception as e:
        print(f"Error in Furthest Point Sampling: {e}")

# 运行测试
if __name__ == "__main__":
    test_pointnet2_fps()
