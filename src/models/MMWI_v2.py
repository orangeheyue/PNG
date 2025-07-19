import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class MultiModalWaveletInterestAttention(nn.Module):
    def __init__(self, embed_dim: int, wavelet_name: str = 'db1', decomp_level: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.decomp_level = decomp_level
        
        # 多级小波变换组件
        self.dwt = DWT1DForward(wave=wavelet_name, J=decomp_level)
        self.idwt = DWT1DInverse(wave=wavelet_name)
        
        # 门控融合组件
        self.low_gate = nn.Sequential(
            nn.Linear(embed_dim*3, 1),
            nn.Sigmoid()
        )
        self.high_gate = nn.Sequential(
            nn.Linear(embed_dim*3, 1),
            nn.Sigmoid()
        )
        
        # 频域特征增强
        self.low_enhance = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        self.high_enhance = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        
        # 动态频率注意力
        self.freq_attention = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim//4),
            nn.LayerNorm(embed_dim//4),
            nn.GELU(),
            nn.Linear(embed_dim//4, 2),
            nn.Softmax(dim=-1)
        )
        
        # 对比学习投影头
        self.contrast_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, 16)  # 压缩到低维空间
        )
        
        # 残差连接
        self.res_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def multi_level_decomp(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """多级小波分解"""
        x = x.unsqueeze(1)  # [B,1,D]
        cA_list = []
        cD_list = []
        
        current = x
        for _ in range(self.decomp_level):
            cA, cD = self.dwt(current)
            cA_list.append(cA.squeeze(1))
            cD_list.append(cD[0].squeeze(1))
            current = cA  # 下一级分解
        
        return cA_list, cD_list

    def multi_level_recon(self, cA_list: List[torch.Tensor], cD_list: List[torch.Tensor]) -> torch.Tensor:
        """多级小波重构"""
        assert len(cA_list) == len(cD_list) == self.decomp_level
        
        current_cA = cA_list[-1].unsqueeze(1)
        current_cD = [cD_list[-1].unsqueeze(1)]
        
        for i in reversed(range(self.decomp_level-1)):
            current_cA = self.idwt((current_cA, current_cD))
            current_cA = cA_list[i].unsqueeze(1) + current_cA  # 跨级连接
            current_cD = [cD_list[i].unsqueeze(1)]
            
        return self.idwt((current_cA, current_cD)).squeeze(1)

    def gate_fusion(self, components: List[torch.Tensor], gate_net: nn.Module) -> torch.Tensor:
        """门控融合机制"""
        concated = torch.cat(components, dim=-1)  # [B,3*D]
        gate = gate_net(concated)  # [B,1]
        
        # 加权融合
        weighted = sum(gate[:,i] * comp for i, comp in enumerate(components))
        return weighted / (gate.sum(dim=1, keepdim=True) + 1e-6)

    def forward(self, content_embeds: torch.Tensor, side_embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 多级分解
        content_low, content_high = self.multi_level_decomp(content_embeds)
        side_low, side_high = self.multi_level_decomp(side_embeds)
        fusion_low, fusion_high = self.multi_level_decomp(content_embeds + side_embeds)
        
        # 门控融合
        low_fused = self.gate_fusion(
            [self.low_enhance(content_low[-1]), 
             self.low_enhance(side_low[-1]),
             fusion_low[-1]],
            self.low_gate)
        
        high_fused = self.gate_fusion(
            [self.high_enhance(content_high[-1]), 
             self.high_enhance(side_high[-1]),
             fusion_high[-1]],
            self.high_gate)
        
        # 动态频率注意力
        attention_weights = self.freq_attention(
            torch.cat([low_fused, high_fused], dim=-1))
        output = self.multi_level_recon(
            [attention_weights[:,0].unsqueeze(-1) * low_fused],
            [attention_weights[:,1].unsqueeze(-1) * high_fused]
        )
        
        # 残差连接
        output = self.norm(output + self.res_proj(content_embeds + side_embeds))
        
        # 对比学习特征
        low_feat = self.contrast_proj(low_fused.detach())
        high_feat = self.contrast_proj(high_fused.detach())
        
        return output, low_feat, high_feat
    
    
def contrastive_loss(low_feat: torch.Tensor, 
                    high_feat: torch.Tensor,
                    temperature: float = 0.1) -> torch.Tensor:
    """
    改进的对比损失函数
    Args:
        low_feat: [N,D] 低频特征
        high_feat: [N,D] 高频特征
        temperature: 温度系数
    """
    # 特征归一化
    low_norm = F.normalize(low_feat, p=2, dim=1)
    high_norm = F.normalize(high_feat, p=2, dim=1)
    # 相似度矩阵
    sim_matrix = torch.mm(low_norm, high_norm.T) / temperature  # [N,N]
    # 对称损失计算
    labels = torch.arange(len(low_feat)).to(low_feat.device)
    loss = (F.cross_entropy(sim_matrix, labels) +  F.cross_entropy(sim_matrix.T, labels)) / 2
    
    return loss