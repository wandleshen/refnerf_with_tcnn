"""针对 Ref-NeRF 的工具函数

代码来自 https://github.com/google-research/multinerf 并修改为 PyTorch 版本
"""

import numpy as np
import torch

def reflect(viewdirs, normals):
    """计算反射向量，论文公式 (4)"""
    return 2.0 * np.sum(
        normals * viewdirs, axis=-1, keepdims=True) * normals - viewdirs

def l2_normalize(x, eps=np.finfo(float).eps):
    """向量归一化"""
    return x / np.sqrt(np.maximum(np.sum(x**2, axis=-1, keepdims=True), eps))

def generalized_binomial_coeff(a, k):
    """计算通用二项式系数"""
    return np.prod(a - np.arange(k)) / np.math.factorial(k)

def assoc_legendre_coeff(l, m, k):
    """计算关联 Legendre 系数

    返回第 (l, m) 个关联 Legendre 系数的 cos^k(theta)*sin^m(theta) 系数
    """
    return ((-1)**m * 2**l * np.math.factorial(l) / np.math.factorial(k) /
          np.math.factorial(l - k - m) *
          generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l))

def sph_harm_coeff(l, m, k):
    """计算球谐系数"""
    return (np.sqrt(
        (2.0 * l + 1.0) * np.math.factorial(l - m) /
        (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))

def get_ml_array(deg_view):
    """创建所有 (l, m) 的组合对"""
    ml_list = []
    for i in range(deg_view):
        l = 2**i
        for m in range(l+1):
            ml_list.append((m, l))

    return np.array(ml_list).T

def generate_ide_fn(deg_view):
    """生成 IDE 函数，论文公式 (6)"""
    if deg_view > 5:
        raise ValueError("deg_view must be no more than 5")

    ml_array = get_ml_array(deg_view)
    l_max = 2**(deg_view-1)

    # 创建一个包含所有系数的矩阵，
    # 当其被一个 z 范德蒙矩阵左乘时，
    # 在 z 的数据就是最终的结果。
    mat = torch.zeros((l_max+1, ml_array.shape[1])).cuda()
    for i, (m, l) in enumerate(ml_array.T):
        for k in range(l - m + 1):
            mat[k, i] = sph_harm_coeff(l, m, k)

    def integrated_dir_enc_fn(xyz, kappa_inv):
        """返回 IDE ，论文公式 (7)(8)
        
        Args:
            xyz: [..., 3] 需要评估的坐标系
            kappa_inv: [..., 1] vMS 分布系数的倒数
        """
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]

        ml_array_cu = torch.from_numpy(ml_array).cuda()
        # 计算 z 范德蒙矩阵
        vmz = torch.cat([z**i for i in range(mat.shape[0])], axis=-1)

        # 计算 x+yi 范德蒙矩阵
        vmxy = torch.cat([(x + 1j * y)**m for m in ml_array_cu[0, :]], axis=-1)

        # 得到球谐函数
        sph_harms = vmxy * (vmz @ mat)

        # 使用 vMS 分布衰减
        sigma = 0.5 * ml_array_cu[1, :] * (ml_array_cu[1, :] + 1)
        ide = sph_harms * torch.exp(-sigma * kappa_inv)

        # 分成实部和虚部
        return torch.cat([torch.real(ide), torch.imag(ide)], axis=-1)

    return integrated_dir_enc_fn
