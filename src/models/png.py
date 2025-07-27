# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from pytorch_wavelets import DWT1D
from scipy.stats import dirichlet

class PNG(nn.Module):
    """
    User-Item Popular Niche Interest Graph (UIG)模型
    基于小波变换提取用户-物品交互图中的流行兴趣和小众兴趣，优化内存版本
    所有张量强制为float32类型，避免类型冲突
    """
    def __init__(self, R: torch.Tensor, wavelet='db3', level=3, device='cuda:0'):
        super(PNG, self).__init__()
        # 强制输入矩阵为float32类型（核心修改）
        self.R = R.coalesce().to(device).to(dtype=torch.float32)
        self.device = device
        self.wavelet = wavelet
        self.level = level
        self.dtype = torch.float32  # 统一类型标识
        
        # 矩阵维度
        self.num_users = R.size(0)
        self.num_items = R.size(1)
        
        # 小波变换层（共享权重）
        self.dwt = DWT1D(wave=wavelet, J=level).to(device)
        
        # 预计算索引和值（已为float32）
        self.indices = self.R.indices()  # [2, nnz]
        self.values = self.R.values()    # [nnz]（float32）
        
        # 存储中间结果（延迟计算）
        self.item_popularity = None
        self.user_activity = None
        self.user_low = None  # 用户低频系数（大众兴趣）
        self.user_high = None # 用户高频系数（小众兴趣）
        self.item_low = None  # 物品低频系数（大众兴趣）
        self.item_high = None # 物品高频系数（小众兴趣）
        
        # 输出结果
        self.UIPG = None
        self.UING = None
        self.quadrants = None

    def _compute_item_popularity(self):
        """计算物品流行度（强制float32）"""
        if self.item_popularity is None:
            # 初始化流行度向量（显式指定float32）
            pop = torch.zeros(self.num_items, device=self.device, dtype=self.dtype)
            # 按物品索引累加交互值（带对数惩罚）
            counts = torch.bincount(self.indices[1], minlength=self.num_items)
            log_counts = torch.log1p(counts.to(self.dtype))  # 确保log_counts为float32
            pop.scatter_add_(0, self.indices[1], self.values * log_counts[self.indices[1]])
            # 归一化（L2范数）
            self.item_popularity = pop / (torch.norm(pop) + 1e-8)
        return self.item_popularity

    def _compute_user_activity(self):
        """计算用户活跃度（强制float32）"""
        if self.user_activity is None:
            # 初始化活跃度向量（显式指定float32）
            act = torch.zeros(self.num_users, device=self.device, dtype=self.dtype)
            # 依赖物品流行度
            pop = self._compute_item_popularity()
            # 按用户索引累加（除以流行度惩罚）
            pop_vals = pop[self.indices[1]] + 1e-8  # 避免除零
            act.scatter_add_(0, self.indices[0], self.values / torch.log1p(pop_vals))
            # 归一化（L2范数）
            self.user_activity = act / (torch.norm(act) + 1e-8)
        return self.user_activity

    def _get_user_signals(self, user_idx):
        """生成单个用户的交互信号（强制float32）"""
        mask = (self.indices[0] == user_idx)
        item_ids = self.indices[1][mask]
        vals = self.values[mask]
        
        # 构建信号向量（显式指定float32）
        signal = torch.zeros(self.num_items, device=self.device, dtype=self.dtype)
        signal[item_ids] = vals * (1 + self.user_activity[user_idx])
        return signal.unsqueeze(0).unsqueeze(0)  # [1,1,D]

    def _get_item_signals(self, item_idx):
        """生成单个物品的交互信号（强制float32）"""
        mask = (self.indices[1] == item_idx)
        user_ids = self.indices[0][mask]
        vals = self.values[mask]
        
        # 构建信号向量（显式指定float32）
        signal = torch.zeros(self.num_users, device=self.device, dtype=self.dtype)
        signal[user_ids] = vals * (1 + self.item_popularity[item_idx])
        return signal.unsqueeze(0).unsqueeze(0)  # [1,1,D]

    def _wavelet_transform_batch(self, is_user=True, batch_size=32):
        """批量小波变换（强制float32）"""
        num_entities = self.num_users if is_user else self.num_items
        # 初始化低频/高频张量（显式指定float32）
        low_freq = torch.zeros(num_entities, device=self.device, dtype=self.dtype)
        high_freq = torch.zeros(num_entities, device=self.device, dtype=self.dtype)
        
        # 分批次处理
        for start in range(0, num_entities, batch_size):
            end = min(start + batch_size, num_entities)
            batch_low = []
            batch_high = []
            
            for idx in range(start, end):
                # 获取信号（用户/物品）
                if is_user:
                    signal = self._get_user_signals(idx)
                else:
                    signal = self._get_item_signals(idx)
                
                # 小波变换
                lf, hf_list = self.dwt(signal)
                # 低频系数：全局趋势（取均值）
                batch_low.append(torch.mean(lf).item())  # 转为Python标量，避免类型冲突
                # 高频系数：细节变化（取能量和）
                hf_energy = torch.sum(torch.stack([torch.norm(hf) for hf in hf_list])).item()
                batch_high.append(hf_energy)
            
            # 批量写入结果（确保float32）
            low_freq[start:end] = torch.tensor(batch_low, device=self.device, dtype=self.dtype)
            high_freq[start:end] = torch.tensor(batch_high, device=self.device, dtype=self.dtype)
        
        return low_freq, high_freq

    def _dirichlet_weight(self, low, high):
        """基于Dirichlet分布计算权重（输出float32）"""
        # 确保alpha为正值且为float32
        alpha_low = torch.clamp(low, min=1e-6)
        alpha_high = torch.clamp(high, min=1e-6)
        # 归一化alpha
        alpha_sum = alpha_low + alpha_high
        alpha_low /= alpha_sum
        alpha_high /= alpha_sum
        
        # 批量采样Dirichlet分布
        weights = []
        for a1, a2 in zip(alpha_low.cpu().numpy(), alpha_high.cpu().numpy()):
            weights.append(dirichlet.rvs([a1*10, a2*10], size=1)[0, 0])
        
        # 转换为float32张量
        return torch.tensor(weights, device=self.device, dtype=self.dtype)

    def forward(self):
        """前向传播（所有输出强制为float32）"""
        # 1. 计算基础指标
        self._compute_item_popularity()
        self._compute_user_activity()
        
        # 2. 小波变换（分用户和物品）
        print("处理用户信号...")
        self.user_low, self.user_high = self._wavelet_transform_batch(is_user=True)
        print("处理物品信号...")
        self.item_low, self.item_high = self._wavelet_transform_batch(is_user=False)
        
        # 3. Dirichlet权重计算
        user_pop_weight = self._dirichlet_weight(self.user_low, self.user_high)  # [U]
        item_pop_weight = self._dirichlet_weight(self.item_low, self.item_high)  # [I]
        user_niche_weight = 1 - user_pop_weight
        item_niche_weight = 1 - item_pop_weight
        
        # 4. 构建兴趣图（稀疏表示，强制float32）
        # 流行兴趣图：用户大众权重 × 物品大众权重 × 交互值
        uipg_vals = self.values * user_pop_weight[self.indices[0]] * item_pop_weight[self.indices[1]]
        self.UIPG = torch.sparse_coo_tensor(
            self.indices, 
            uipg_vals,  # 已为float32
            (self.num_users, self.num_items), 
            device=self.device,
            dtype=self.dtype  # 显式指定为float32
        )
        
        # 小众兴趣图
        uing_vals = self.values * user_niche_weight[self.indices[0]] * item_niche_weight[self.indices[1]]
        self.UING = torch.sparse_coo_tensor(
            self.indices, 
            uing_vals, 
            (self.num_users, self.num_items), 
            device=self.device,
            dtype=self.dtype
        )
        
        # 5. 四象限表征（稀疏存储，float32）
        self.quadrants = {
            'pop_user_pop_item': torch.sparse_coo_tensor(
                self.indices, 
                self.values * user_pop_weight[self.indices[0]] * item_pop_weight[self.indices[1]],
                (self.num_users, self.num_items), 
                device=self.device,
                dtype=self.dtype
            ),
            'pop_user_niche_item': torch.sparse_coo_tensor(
                self.indices, 
                self.values * user_pop_weight[self.indices[0]] * item_niche_weight[self.indices[1]],
                (self.num_users, self.num_items), 
                device=self.device,
                dtype=self.dtype
            ),
            'niche_user_pop_item': torch.sparse_coo_tensor(
                self.indices, 
                self.values * user_niche_weight[self.indices[0]] * item_pop_weight[self.indices[1]],
                (self.num_users, self.num_items), 
                device=self.device,
                dtype=self.dtype
            ),
            'niche_user_niche_item': torch.sparse_coo_tensor(
                self.indices, 
                self.values * user_niche_weight[self.indices[0]] * item_niche_weight[self.indices[1]],
                (self.num_users, self.num_items), 
                device=self.device,
                dtype=self.dtype
            )
        }
        
        return self.UIPG, self.UING, self.quadrants
    
# 测试用例（模拟大规模稀疏矩阵）
def test_uig_memory_efficiency():
    # 生成19445×7050的稀疏矩阵（模拟输入）
    num_users, num_items = 19445, 7050
    nnz = 118551  # 非零元素数量
    
    # 随机生成交互数据
    user_idx = torch.randint(0, num_users, (nnz,))
    item_idx = torch.randint(0, num_items, (nnz,))
    values = torch.rand(nnz) * 0.5  # 交互强度
    
    # 构建稀疏矩阵
    indices = torch.stack([user_idx, item_idx])
    R = torch.sparse_coo_tensor(indices, values, (num_users, num_items), device='cuda:0')
    
    # 模型测试
    model = UIG(R, wavelet='db3', level=2)
    UIPG, UING, quads = model.forward()
    print("quads:", quads['pop_user_pop_item'].to_dense())
    # 验证结果
    print(f"UIPG 非零元素: {UIPG._nnz()}")
    print(f"UING 非零元素: {UING._nnz()}")
    print(f"四象限非零元素: {[quads[k]._nnz() for k in quads]}")
    
    # 内存使用检查
    print(f"用户低频系数内存: {model.user_low.element_size() * model.user_low.nelement() / 1024 / 1024:.2f} MB")
    print(f"物品高频系数内存: {model.item_high.element_size() * model.item_high.nelement() / 1024 / 1024:.2f} MB")

# if __name__ == "__main__":
#     test_uig_memory_efficiency()