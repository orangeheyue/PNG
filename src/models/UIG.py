# User-Item Popular-Niche Interest Graph (UIG) model
# This file is part of the PNG project.
# Author: orangeai-research
# Date: 2025-07-18
# Version: 1.0
# Description: This module implements the User-Item Popular Interest Graph (UIG) model for analyzing user interests.

import torch
import torch.nn as nn


import torch
import torch.nn as nn
import pywt
import numpy as np
from torch.nn.functional import softmax

class UIG(nn.Module):
    """
    User-Item Popular Interest Graph (UIG) model for analyzing user interests.
    Desc: 基于小波变换，对用户-物品交互图进行流行兴趣和小众兴趣的分析。将原始用户-物品兴趣图分解为流行兴趣图和小众兴趣图。

    Args:
        inputs: self.R: tensor(indices=tensor([[    0,     0,     0,  ..., 19444, 19444, 19444],
                        [    0,  1587,  1879,  ...,  6959,  7005,  7022]]),
        values=tensor([0.1925, 0.0358, 0.1826,  ..., 0.1543, 0.0870, 0.1054]),
        device='cuda:0', size=(19445, 7050), nnz=118551, layout=torch.sparse_coo), 原始用户-物品兴趣图, size=(19445, 7050)
    
    Methods:
        - forward: Computes the User-Item Popular Interest Graph (UIPG) and User-Item Niche Interest Graph (UING).

    Returns:
        UIPG: User-Item Popular Interest Graph,用户-物品流行兴趣图
        UING: User-Item Niche Interest Graph,用户-物品小众冷门兴趣图
    
    Note:
        This model is designed to analyze user interests based on a sparse user-item interaction matrix.
        It computes the Popular-Niche Interest of items for each user.
    
    """
    def __init__(self, R: torch.Tensor, wavelet='haar', level=1):
        super(UIG, self).__init__()
        self.R = R
        self.wavelet = wavelet
        self.level = level
        
        # 计算物品流行度
        self.item_popularity = self._compute_item_popularity()
        
        # 小波变换需要的参数
        self.wavelet_obj = pywt.Wavelet(wavelet)
        self.filter_len = self.wavelet_obj.dec_len
        
        # 缓存设备
        self.device = R.device if R.is_cuda else 'cpu'

    def _compute_item_popularity(self):
        """
            Desc:计算每个物品的流行度，定义为所有用户对该物品的兴趣之和
            Method: 使用稀疏矩阵的行求和来计算物品流行度
            TODO: 优化点:当前计算的方式是采用的R交互图,但是R交互图里经过拉普拉斯变换后的值,并不能精准代表用户兴趣，
            可以考虑使用更复杂的流行度计算方法，例如基于用户物品交互矩阵的加权平均 

        """
        item_popularity = torch.sparse.sum(self.R, dim=0).to_dense()
        return item_popularity

    def _dwt_1d(self, signal):
        """
        对输入信号执行一维离散小波变换
        """
        # 转换为numpy进行小波变换
        signal_np = signal.cpu().numpy()
        
        # 执行小波变换
        coeffs = pywt.wavedec(signal_np, self.wavelet, level=self.level)
        
        # 分解系数为近似系数和细节系数
        approx_coeffs = coeffs[0]
        detail_coeffs = coeffs[1:]
        
        # 转换回torch张量
        approx_coeffs = torch.tensor(approx_coeffs, device=self.device, dtype=torch.float32)
        
        # 细节系数列表转换为张量
        detail_coeffs_list = []
        for dc in detail_coeffs:
            detail_coeffs_list.append(torch.tensor(dc, device=self.device, dtype=torch.float32))
        
        return approx_coeffs, detail_coeffs_list

    def _idwt_1d(self, approx_coeffs, detail_coeffs_list):
        """
        对小波系数执行一维逆离散小波变换
        """
        # 转换为numpy进行逆变换
        approx_coeffs_np = approx_coeffs.cpu().numpy()
        detail_coeffs_np = [dc.cpu().numpy() for dc in detail_coeffs_list]
        
        # 合并系数
        coeffs = [approx_coeffs_np] + detail_coeffs_np
        
        # 执行逆小波变换
        reconstructed_signal = pywt.waverec(coeffs, self.wavelet)
        
        # 转换回torch张量
        reconstructed_signal = torch.tensor(reconstructed_signal, device=self.device, dtype=torch.float32)
        
        return reconstructed_signal

    def forward(self):
        """
        计算用户-物品流行兴趣图(UIPG)和用户-物品小众兴趣图(UING)
        """
        # 确保R是COO格式的稀疏张量，并进行coalesce处理
        if self.R.layout != torch.sparse_coo:
            self.R = self.R.to_sparse_coo()
        self.R = self.R.coalesce()  # 合并重复索引，解决Cannot get indices的错误
        
        # 获取稀疏矩阵的索引和值
        indices = self.R.indices()
        values = self.R.values()
        
        # 创建结果稀疏矩阵的索引和值
        uipg_indices = []
        uipg_values = []
        uing_indices = []
        uing_values = []
        
        # 为了提高效率，按用户分组处理
        unique_users = torch.unique(indices[0])
        
        for user in unique_users:
            # 获取当前用户的所有交互
            user_mask = indices[0] == user
            user_items = indices[1, user_mask]
            user_values = values[user_mask]
            
            # 对用户的兴趣向量执行小波变换
            # 首先构建用户完整的兴趣向量
            user_vector = torch.zeros(self.R.size(1), device=self.device)
            user_vector[user_items] = user_values
            
            # 执行小波变换
            approx_coeffs, detail_coeffs_list = self._dwt_1d(user_vector)
            
            # 从小波系数重构信号
            # 1. 仅使用近似系数重构流行兴趣
            popular_signal = self._idwt_1d(approx_coeffs, [torch.zeros_like(dc) for dc in detail_coeffs_list])
            
            # 2. 仅使用细节系数重构小众兴趣
            niche_signal = self._idwt_1d(torch.zeros_like(approx_coeffs), detail_coeffs_list)
            
            # 截取原始交互的物品位置
            user_popular_values = popular_signal[user_items]
            user_niche_values = niche_signal[user_items]
            
            # 归一化处理
            if torch.sum(user_popular_values) > 0:
                user_popular_values = softmax(user_popular_values, dim=0)
            
            if torch.sum(user_niche_values) > 0:
                user_niche_values = softmax(user_niche_values, dim=0)
            
            # 收集结果
            user_indices = torch.stack([torch.full_like(user_items, user), user_items])
            uipg_indices.append(user_indices)
            uipg_values.append(user_popular_values)
            uing_indices.append(user_indices)
            uing_values.append(user_niche_values)
        
        # 合并所有用户的结果
        if uipg_indices:
            uipg_indices = torch.cat(uipg_indices, dim=1)
            uipg_values = torch.cat(uipg_values)
            UIPG = torch.sparse_coo_tensor(uipg_indices, uipg_values, size=self.R.size())
        else:
            UIPG = torch.sparse_coo_tensor([[], []], [], size=self.R.size())
        
        if uing_indices:
            uing_indices = torch.cat(uing_indices, dim=1)
            uing_values = torch.cat(uing_values)
            UING = torch.sparse_coo_tensor(uing_indices, uing_values, size=self.R.size())
        else:
            UING = torch.sparse_coo_tensor([[], []], [], size=self.R.size())
        
        return UIPG, UING

# 使用示例
if __name__ == "__main__":
    # 创建一个小型的稀疏用户-物品交互矩阵用于测试
    indices = torch.tensor([[0, 0, 1, 1, 2, 2],
                           [0, 1, 1, 2, 2, 0]])
    values = torch.tensor([0.8, 0.2, 0.6, 0.4, 0.7, 0.3])
    R = torch.sparse_coo_tensor(indices, values, size=(3, 3))
    
    # 初始化模型
    model = UIG(R)
    
    # 前向传播
    UIPG, UING = model.forward()
    
    print("原始用户-物品兴趣矩阵 R:")
    print(R.to_dense())
    
    print("\n用户-物品流行兴趣图 UIPG:")
    print(UIPG.to_dense())
    
    print("\n用户-物品小众兴趣图 UING:")
    print(UING.to_dense())