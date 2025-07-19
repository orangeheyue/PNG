import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse  # 1D小波变换
import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse
import torch.nn.functional as F


import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse  

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1D, IDWT1D

import torch
import torch.nn as nn
from pytorch_wavelets import DWT1D, IDWT1D  # 假设已安装 pytorch_wavelets 库

# class MultiModalWaveletInterestAttention(nn.Module):
#     def __init__(self, embed_dim, wavelet_name='db1', decomp_level=1):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.wavelet_name = wavelet_name
#         self.decomp_level = decomp_level  # 显式保存分解层数
        
#         # 小波变换（正向+逆向），使用对称填充保留边缘信息
#         self.dwt = DWT1D(wave=wavelet_name, mode='symmetric', J=decomp_level)
#         self.idwt = IDWT1D(wave=wavelet_name, mode='symmetric')
        
#         # 轻量级注意力机制，减少中间维度
#         self.attention = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim//8),  # 降维减少计算量
#             nn.LayerNorm(embed_dim//8),
#             nn.GELU(),
#             nn.Linear(embed_dim//8, 2),          # 直接输出2个权重（低频/高频）
#             nn.Softmax(dim=-1)                   # 权重和为1，平衡两种兴趣
#         )
        
#     def wavelet_decomp(self, x):
#         """可微分小波分解，返回低频和高频系数（保留完整分解层级）"""
#         # 输入形状: [B, D] -> 转换为 [B, 1, D]（适配小波变换的通道维度）
#         x = x.unsqueeze(1)  # [B, 1, D]
#         # 执行多层分解，返回低频和所有层级的高频系数
#         # 对于单层分解（J=1），high_freqs 是包含1个元素的列表
#         low_freq, high_freqs = self.dwt(x)
#         # 提取最后一层的低频和高频系数（多尺度分解时取最底层）
#         final_low_freq = low_freq  # [B, 1, D/2^J]
#         final_high_freq = high_freqs[-1]  # [B, 1, D/2^J]
        
#         # 压缩通道维度，形状: [B, D/2^J]
#         return final_low_freq.squeeze(1), final_high_freq.squeeze(1)
    
#     def wavelet_recon(self, low_freq, high_freq):
#         """可微分小波重构，从低频和单层级高频系数恢复信号"""
#         # 恢复通道维度: [B, D/2^J] -> [B, 1, D/2^J]
#         low_freq = low_freq.unsqueeze(1)
#         high_freq = high_freq.unsqueeze(1)  # 包装为单元素列表适配接口
        
#         # 执行逆变换，形状: [B, 1, D] -> [B, D]
#         recon_x = self.idwt((low_freq, [high_freq]))
#         return recon_x.squeeze(1)
    
#     def forward(self, content_embeds, side_embeds):
#         """
#         Args:
#             content_embeds: 用户-物品交互特征 [B, D]
#             side_embeds: 多模态辅助特征 [B, D]
#         Returns:
#             reconstructed: 多尺度兴趣感知特征 [B, D]
#             low_freq_interest: 低频（热门）兴趣特征 [B, D]
#             high_freq_interest: 高频（小众）兴趣特征 [B, D]
#         """
#         # 1. 融合原始特征（减少冗余分解）
#         fused_embeds = content_embeds + side_embeds  # [B, D]
        
#         # 2. 小波分解获取低频/高频兴趣（单层分解示例）
#         fusion_cA, fusion_cD = self.wavelet_decomp(fused_embeds)  # [B, D/2], [B, D/2]
        
#         # 3. 直接使用分解后的系数作为对比学习特征（避免重构计算）
#         #    注：若需重构特征，可取消注释下方代码
#         # low_freq_interest = self.wavelet_recon(fusion_cA, torch.zeros_like(fusion_cD))
#         # high_freq_interest = self.wavelet_recon(torch.zeros_like(fusion_cA), fusion_cD)
#         low_freq_interest, high_freq_interest = fusion_cA, fusion_cD  # 直接返回系数（节省显存）
        
#         # 4. 注意力机制：对原始融合特征加权（避免使用中间拼接特征）
#         weights = self.attention(fused_embeds)  # [B, 2]，自动分为alpha（低频权重）和 beta（高频权重）
#         alpha, beta = weights.chunk(2, dim=-1)  # [B, 1], [B, 1]
        
#         # 5. 加权融合低频/高频系数（显存友好的广播机制）
#         weighted_cA = alpha * fusion_cA  # [B, D/2]
#         weighted_cD = beta * fusion_cD    # [B, D/2]
        
#         # 6. 小波重构恢复原始维度（D/2 -> D）
#         reconstructed = self.wavelet_recon(weighted_cA, weighted_cD)  # [B, D]
        
#         return reconstructed, low_freq_interest, high_freq_interest


class MultiModalWaveletInterestAttention1(nn.Module):
    '''
        Desc: 多模态小波兴趣注意力机制
            必要性:当前捕获用户兴趣偏好的方法无法分离用户对物品的Hot兴趣和小众兴趣,  同时融合兴趣的方式是content_embeds + side_embeds
            -------------------------------------------
            优点：
                小波变换能有效分离频域特征，适合兴趣分解
                低频/高频的区分符合热门/小众兴趣的特性
                Attention加权可以动态平衡两种兴趣
            -------------------------------------------
            基于小波变换的，多模态兴趣偏好感知, 初步思路是：
            1. 兴趣分离:分别对content_embeds和fusion_embeds以及融合兴趣(content_embeds + side_embeds)进行小波分解，分离出在低频(对多模态公共兴趣，热门兴趣)和高频特征(个性化兴趣，小众兴趣)中进行用户兴趣。
            2. 分别在低频和高频域进行content_embeds和 side_embeds进行兴趣偏好融合，目前使用的是element-wise add,用简单高效优雅的方式(还有哪些方式)。
            然后，将融合后的低频特征(热门兴趣) 和 高频特征(小众兴趣) 设计一个attention进行加权
            3. 多尺度兴趣感知:最后再小波逆变换，得到多尺度用户兴趣感知embeeding 以及高频兴趣和低频兴趣(用于对比学习)
        Args:
            content_embeds.shape: torch.Size([26495, 64])  用户和物品的embedding
            side_embeds.shape: torch.Size([26495, 64])    多模态融合的embeeding
        Returns:
            mm_interest_prefer_aware_embeds: torch.Size([26495, 64]) ，低频注意力兴趣 torch.Size([26495, 64]) ，高频注意力兴趣 torch.Size([26495, 64]) 
    '''
    def __init__(self, embed_dim, wavelet_name='db1', decomp_level=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.wavelet_name = wavelet_name
        
        # 小波变换与逆变换（支持自动梯度计算）
        self.dwt = DWT1DForward(wave=wavelet_name, J=decomp_level)
        self.idwt = DWT1DInverse(wave=wavelet_name)
        
        # 增强的注意力机制
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//8),
            nn.GELU(),
            nn.Softmax(dim=-1)
        )
        
        # 低频融合投影层
        self.low_fusion = nn.Sequential(
            nn.Linear(embed_dim//2 , embed_dim//2),
            nn.LeakyReLU(),
        )
        
        # 高频融合投影层
        self.high_fusion = nn.Sequential(
            nn.Linear(embed_dim//2 , embed_dim//2),
            nn.BatchNorm1d(embed_dim//2),
            nn.LeakyReLU(),
        )
        
        # 残差归一化
        self.norm = nn.LayerNorm(embed_dim)

        # 新增：低频门控（调节content/side/fusion的低频贡献）
        self.low_gate = nn.Sequential(
            nn.Linear( embed_dim//2 * 3 , 3),  # 3个输入源（content/side/fusion）的低频权重
            nn.Softmax(dim=-1)
        )
        # 新增：高频门控（调节content/side/fusion的高频贡献）
        self.high_gate = nn.Sequential(
            nn.Linear( embed_dim//2 * 3, 3),  # 3个输入源的高频权重
            nn.Softmax(dim=-1)
        )
    def wavelet_decomp(self, x):
        """可微分小波分解"""
        x = x.unsqueeze(1)  # [B,1,D]
        cA, cD = self.dwt(x)
        return cA.squeeze(1), cD[0].squeeze(1)
    
    def wavelet_recon(self, cA, cD):
        """可微分小波重构"""
        cA = cA.unsqueeze(1)
        cD = [cD.unsqueeze(1)]
        return self.idwt((cA, cD)).squeeze(1)
    
    def forward(self, content_embeds, side_embeds):
        '''
        content_embeds.shape: torch.Size([26495, 64])  用户和物品的embedding
        side_embeds.shape: torch.Size([26495, 64])    多模态融合的embeeding
        '''
        # 多尺度小波兴趣分解
        content_cA, content_cD = self.wavelet_decomp(content_embeds) # 用户与物品信息 # 
        side_cA, side_cD = self.wavelet_decomp(side_embeds) # 多模态辅助信息
        fusion_cA, fusion_cD  = self.wavelet_decomp(content_embeds + side_embeds) # 融合信息
        #print("content_cA.shape:", content_cA.shape, "content_cD.shape:", content_cD.shape, "fusion_cA.shape:", fusion_cA.shape, "fusion_cD.shape:", fusion_cD.shape)
        # 低频、高频兴趣融合
        # low_fused = (content_cA + side_cA + fusion_cA) / 3
        # ------------------------ 低频门控融合 ------------------------
        # 拼接3个低频源特征 [B, 3*(D/2)]
        low_sources = torch.cat([content_cA, side_cA, fusion_cA], dim=-1) #96
        low_weights = self.low_gate(low_sources)
        low_fused = (
            low_weights[:, 0:1] * content_cA + low_weights[:, 1:2] * side_cA + low_weights[:, 2:3] * fusion_cA
        )
        #low_fused = self.low_fusion(low_fused)  # 投影层增强表达
        # ------------------------ 高频门控融合 ------------------------
        # 拼接3个高频源特征 [B, 3*(D/2)]
        high_sources = torch.cat([content_cD, side_cD, fusion_cD], dim=-1)
        high_weights = self.high_gate(high_sources)
        high_fused = (high_weights[:, 0:1] * content_cD +high_weights[:, 1:2] * side_cD + high_weights[:, 2:3] * fusion_cD) 
        #high_fused = self.high_fusion(high_fused)  # 投影层增强表达
        # 原始融合兴趣特征
        combined = content_embeds + side_embeds
        weights = self.attention(torch.concat([low_fused, high_fused], dim=-1))
        #weights = self.attention(combined)
        # 加权融合
        # reconstructed = self.wavelet_recon(
        #     weights[:, 0:1] * low_fused,
        #     weights[:, 1:2] * high_fused
        # )
        reconstructed = self.wavelet_recon(
            low_fused,
            high_fused
        )
        low_freq_interest = self.wavelet_recon(low_fused, fusion_cA)
        high_freq_interest = self.wavelet_recon(high_fused, fusion_cD)

        return reconstructed, low_freq_interest, high_freq_interest


# def contrastive_loss(low_feat: torch.Tensor, 
#                     high_feat: torch.Tensor,
#                     temperature: float = 0.1) -> torch.Tensor:
#     """
#     改进的对比损失函数
#     Args:
#         low_feat: [N,D] 低频特征
#         high_feat: [N,D] 高频特征
#         temperature: 温度系数
#     """
#     # 特征归一化
#     low_norm = F.normalize(low_feat, p=2, dim=1)
#     high_norm = F.normalize(high_feat, p=2, dim=1)
#     # 相似度矩阵
#     sim_matrix = torch.mm(low_norm, high_norm.T) / temperature  # [N,N]
#     # 对称损失计算
#     labels = torch.arange(len(low_feat)).to(low_feat.device)
#     loss = (F.cross_entropy(sim_matrix, labels) +  F.cross_entropy(sim_matrix.T, labels)) / 2
    
#     return loss

# class MultiModalWaveletInterestAttention(nn.Module):
#     '''
#             Desc: 多模态小波兴趣注意力机制
#                 必要性:当前捕获用户兴趣偏好的方法无法分离用户对物品的Hot兴趣和小众兴趣,  同时融合兴趣的方式是content_embeds + side_embeds
#                 -------------------------------------------
#                 优点：
#                     小波变换能有效分离频域特征，适合兴趣分解
#                     低频/高频的区分符合热门/小众兴趣的特性
#                     Attention加权可以动态平衡两种兴趣
#                 -------------------------------------------
#                 基于小波变换的，多模态兴趣偏好感知, 初步思路是：
#                 1. 兴趣分离:分别对content_embeds和fusion_embeds以及融合兴趣(content_embeds + side_embeds)进行小波分解，分离出在低频(对多模态公共兴趣，热门兴趣)和高频特征(个性化兴趣，小众兴趣)中进行用户兴趣。
#                 2. 分别在低频和高频域进行content_embeds和 side_embeds进行兴趣偏好融合，目前使用的是element-wise add,用简单高效优雅的方式(还有哪些方式)。
#                 然后，将融合后的低频特征(热门兴趣) 和 高频特征(小众兴趣) 设计一个attention进行加权
#                 3. 多尺度兴趣感知:最后再小波逆变换，得到多尺度用户兴趣感知embeeding 以及高频兴趣和低频兴趣(用于对比学习)
#     '''
#     def __init__(self, embed_dim, wavelet_name='db1', decomp_level=1):
#         super().__init__()
#         self.embed_dim = embed_dim
        
#         # 小波变换组件
#         self.dwt = DWT1DForward(wave=wavelet_name, J=decomp_level)
#         self.idwt = DWT1DInverse(wave=wavelet_name)
        
#         # 频域特征融合组件
#         self.low_fusion = nn.Sequential(
#             nn.Linear(embed_dim*3//2, embed_dim//2),  # 融合三个低频分量
#             nn.GELU()
#         )
        
#         self.high_fusion = nn.Sequential(
#             nn.Linear(embed_dim*3//2, embed_dim//2),  # 融合三个高频分量
#             nn.GELU()
#         )
        
#         # 动态权重注意力
#         self.attention = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim//8),
#             nn.LayerNorm(embed_dim//8),
#             nn.GELU(),
#             nn.Linear(embed_dim//8, 2),
#             nn.Softmax(dim=-1)
#         )
        
#         # 残差连接
#         self.res_connect = nn.Linear(embed_dim, embed_dim)
#         self.norm = nn.LayerNorm(embed_dim)

#     def wavelet_decomp(self, x):
#         """多尺度特征分解"""
#         x = x.unsqueeze(1)  # [B,1,D]
#         cA, cD = self.dwt(x)
#         return cA.squeeze(1), cD[0].squeeze(1)

#     def wavelet_recon(self, cA, cD):
#         """特征重构"""
#         cA = cA.unsqueeze(1)
#         cD = [cD.unsqueeze(1)]
#         return self.idwt((cA, cD)).squeeze(1)

#     def forward(self, content_embeds, side_embeds):
#         # 原始特征分解
#         content_low, content_high = self.wavelet_decomp(content_embeds)
#         side_low, side_high = self.wavelet_decomp(side_embeds)
#         fusion_low, fusion_high = self.wavelet_decomp(content_embeds + side_embeds)
        
#         # 跨模态特征融合
#         low_fused = self.low_fusion(
#             torch.cat([content_low, side_low, fusion_low], dim=-1))
#         high_fused = self.high_fusion(
#             torch.cat([content_high, side_high, fusion_high], dim=-1))
        
#         # 动态频域注意力
#         attention_weights = self.attention(
#             torch.cat([low_fused, high_fused], dim=-1))  # [B, 2]
#         low_weight = attention_weights[:, 0].unsqueeze(-1)
#         high_weight = attention_weights[:, 1].unsqueeze(-1)
        
#         # 加权重构
#         reconstructed = self.wavelet_recon(
#             low_weight * low_fused,
#             high_weight * high_fused
#         )
#         #print("reconstructed.shape:", reconstructed.shape)

#         # 残差连接与归一化
#         residual = self.res_connect(content_embeds + side_embeds)
#         #print("residual.shape:", residual.shape)
#         # output = self.norm(reconstructed + residual)
#         output = reconstructed 
#         return output, low_fused, high_fused




class MultiModalWaveletInterestAttention(nn.Module):
    '''
        Desc: 多模态小波兴趣注意力机制
            必要性:当前捕获用户兴趣偏好的方法无法分离用户对物品的Hot兴趣和小众兴趣,  同时融合兴趣的方式是content_embeds + side_embeds
            -------------------------------------------
            优点：
                小波变换能有效分离频域特征，适合兴趣分解
                低频/高频的区分符合热门/小众兴趣的特性
                Attention加权可以动态平衡两种兴趣
            -------------------------------------------
            基于小波变换的，多模态兴趣偏好感知, 初步思路是：
            1. 兴趣分离:分别对content_embeds和fusion_embeds以及融合兴趣(content_embeds + side_embeds)进行小波分解，分离出在低频(对多模态公共兴趣，热门兴趣)和高频特征(个性化兴趣，小众兴趣)中进行用户兴趣。
            2. 分别在低频和高频域进行content_embeds和 side_embeds进行兴趣偏好融合，目前使用的是element-wise add,用简单高效优雅的方式(还有哪些方式)。
            然后，将融合后的低频特征(热门兴趣) 和 高频特征(小众兴趣) 设计一个attention进行加权
            3. 多尺度兴趣感知:最后再小波逆变换，得到多尺度用户兴趣感知embeeding 以及高频兴趣和低频兴趣(用于对比学习)
        Args:
            content_embeds.shape: torch.Size([26495, 64])  用户和物品的embedding
            side_embeds.shape: torch.Size([26495, 64])    多模态融合的embeeding
        Returns:
            mm_interest_prefer_aware_embeds: torch.Size([26495, 64]) ，低频注意力兴趣 torch.Size([26495, 64]) ，高频注意力兴趣 torch.Size([26495, 64]) 
    '''
    def __init__(self, embed_dim, wavelet_name='db1', decomp_level=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.wavelet_name = wavelet_name
        
        # 小波变换与逆变换（支持自动梯度计算）
        self.dwt = DWT1DForward(wave=wavelet_name, J=decomp_level)
        self.idwt = DWT1DInverse(wave=wavelet_name)
        
        # 增强的注意力机制
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//8),
            nn.LayerNorm(embed_dim//8),
            nn.GELU(),
            nn.Softmax(dim=-1)
        )
        
        # 低频融合投影层
        self.low_fusion = nn.Sequential(
            nn.Linear(embed_dim//2 , embed_dim//2),
            nn.LeakyReLU(),
        )
        
        # 高频融合投影层
        self.high_fusion = nn.Sequential(
            nn.Linear(embed_dim//2 , embed_dim//2),
            nn.BatchNorm1d(embed_dim//2),
            nn.LeakyReLU(),
        )
        
        # 残差归一化
        self.norm = nn.LayerNorm(embed_dim)

        # 新增：可学习的低频融合权重
        self.low_weights = nn.Parameter(torch.ones(3) / 3)
        
        # 新增：可学习的高频融合权重
        self.high_weights = nn.Parameter(torch.ones(3) / 3)
    def wavelet_decomp(self, x):
        """可微分小波分解"""
        x = x.unsqueeze(1)  # [B,1,D]
        cA, cD = self.dwt(x)
        return cA.squeeze(1), cD[0].squeeze(1)
    
    def wavelet_recon(self, cA, cD):
        """可微分小波重构"""
        cA = cA.unsqueeze(1)
        cD = [cD.unsqueeze(1)]
        return self.idwt((cA, cD)).squeeze(1)
    
    def forward(self, content_embeds, side_embeds, fusion_embeds):
        '''
        content_embeds.shape: torch.Size([26495, 64])  用户和物品的embedding
        side_embeds.shape: torch.Size([26495, 64])    多模态融合的embeeding
        '''
        # 多尺度小波兴趣分解
        content_cA, content_cD = self.wavelet_decomp(content_embeds) # 用户与物品信息
        side_cA, side_cD = self.wavelet_decomp(side_embeds) # 多模态辅助信息
        fusion_cA, fusion_cD  = self.wavelet_decomp(fusion_embeds) # 融合信息
        #print("content_cA.shape:", content_cA.shape, "content_cD.shape:", content_cD.shape, "fusion_cA.shape:", fusion_cA.shape, "fusion_cD.shape:", fusion_cD.shape)
        
        # 低频、高频兴趣融合
        # 低频：
        low_weights = torch.softmax(self.low_weights, dim=0)  # 确保权重和为1
        low_fused = (
            low_weights[0] * content_cA +
            low_weights[1] * side_cA +
            low_weights[2] * fusion_cA
        )
        # low_fused = (content_cA + side_cA + fusion_cA) / 3
        #print("low_fused.shape:", low_fused.shape)
        # 高频：
        # 高频兴趣融合 - 使用可学习的权重
        high_weights = torch.softmax(self.high_weights, dim=0)  # 确保权重和为1
        high_fused = (
            high_weights[0] * content_cD +
            high_weights[1] * side_cD +
            high_weights[2] * fusion_cD
        )
        # high_fused = (content_cD + side_cD + fusion_cD) / 3
        #print("high_fused.shape")
        # 原始融合兴趣特征
        combined = content_embeds + side_embeds
        # combined = torch.multiply(content_embeds, fusion_embeds)
        # combined = fusion_embeds
        # weights = self.attention(torch.concat([low_fused, high_fused], dim=-1))
        weights = self.attention(combined)
        
        # 加权融合
        # reconstructed = self.wavelet_recon(
        #     weights[:, 0:1] * low_fused,
        #     weights[:, 1:2] * high_fused
        # )
        reconstructed = self.wavelet_recon(
            low_fused,
            high_fused
        )
        low_freq_interest = self.wavelet_recon(low_fused, fusion_cA)
        high_freq_interest = self.wavelet_recon(high_fused, fusion_cD)

        return reconstructed, low_freq_interest, high_freq_interest


def contrastive_loss(low_feat, high_feat, temperature=0.1):
    """改进的对比损失"""
    # 特征归一化
    low_norm = F.normalize(low_feat, p=2, dim=1)
    high_norm = F.normalize(high_feat, p=2, dim=1)
    
    # 相似度矩阵
    sim_matrix = torch.mm(low_norm, high_norm.T) / temperature
    
    # 对称式损失
    labels = torch.arange(len(low_feat)).to(low_feat.device)
    loss = (F.cross_entropy(sim_matrix, labels) + 
            F.cross_entropy(sim_matrix.T, labels)) / 2
    
    # 正交正则项
    orth_reg = torch.norm(torch.mm(low_norm.T, high_norm)) / len(low_feat)
    
    return loss + 0.1 * orth_reg  # 总损失

# class MultiModalWaveletInterestAttention(nn.Module):
#     '''
#         Desc: 多模态小波兴趣注意力机制
#             必要性:当前捕获用户兴趣偏好的方法无法分离用户对物品的Hot兴趣和小众兴趣,  同时融合兴趣的方式是content_embeds + side_embeds
#             -------------------------------------------
#             优点：
#                 小波变换能有效分离频域特征，适合兴趣分解
#                 低频/高频的区分符合热门/小众兴趣的特性
#                 Attention加权可以动态平衡两种兴趣
#             -------------------------------------------
#             基于小波变换的，多模态兴趣偏好感知, 初步思路是：
#             1. 兴趣分离:分别对content_embeds和fusion_embeds进行小波分解，分离出在低频(对多模态公共兴趣，热门兴趣)和高频特征(个性化兴趣，小众兴趣)中进行用户兴趣。
#             2. 分别在低频和高频域进行content_embeds和 side_embeds进行兴趣偏好融合，目前使用的是element-wise add,用简单高效优雅的方式(有哪些方式)。
#             然后，将融合后的低频特征(热门兴趣) 和 高频特征(小众兴趣) 设计一个attention进行加权
#             3. 多尺度兴趣感知:最后再小波逆变换。
#         Args:
#             content_embeds.shape: torch.Size([26495, 64])  用户和物品的embedding
#             side_embeds.shape: torch.Size([26495, 64])    多模态融合的embeeding
#         Returns:
#             mm_interest_prefer_aware_embeds: torch.Size([26495, 64]) ，低频注意力兴趣 torch.Size([26495, 64]) ，高频注意力兴趣 torch.Size([26495, 64]) 
#     '''
#     def __init__(self, embed_dim, wavelet_name='db1', decomp_level=1):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.wavelet_name = wavelet_name
        
#         # 小波变换与逆变换（支持自动梯度计算）
#         self.dwt = DWT1DForward(wave=wavelet_name, J=decomp_level)
#         self.idwt = DWT1DInverse(wave=wavelet_name)
        
#         # 增强的注意力机制
#         self.attention = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.LayerNorm(embed_dim),
#             nn.GELU(),
#             nn.Linear(embed_dim, embed_dim//2),
#             nn.GELU(),
#             nn.Linear(embed_dim//2, 2),
#             nn.Softmax(dim=-1)
#         )
        
#         # 低频融合投影层
#         self.low_fusion = nn.Sequential(
#             nn.Linear(embed_dim//2 , embed_dim//2),
#             nn.LeakyReLU(),
#         )
        
#         # 高频融合投影层
#         self.high_fusion = nn.Sequential(
#             nn.Linear(embed_dim//2 , embed_dim//2),
#             nn.BatchNorm1d(embed_dim//2),
#             nn.LeakyReLU(),
#         )
        
#         # 残差归一化
#         self.norm = nn.LayerNorm(embed_dim)
        
#     def wavelet_decomp(self, x):
#         """可微分小波分解"""
#         x = x.unsqueeze(1)  # [B,1,D]
#         cA, cD = self.dwt(x)
#         return cA.squeeze(1), cD[0].squeeze(1)
    
#     def wavelet_recon(self, cA, cD):
#         """可微分小波重构"""
#         cA = cA.unsqueeze(1)
#         cD = [cD.unsqueeze(1)]
#         return self.idwt((cA, cD)).squeeze(1)
    
#     def forward(self, content_embeds, side_embeds):
#         '''
#         content_embeds.shape: torch.Size([26495, 64])  用户和物品的embedding
#         side_embeds.shape: torch.Size([26495, 64])    多模态融合的embeeding
#         '''
#         # 小波分解
#         content_cA, content_cD = self.wavelet_decomp(content_embeds)
#         side_cA, side_cD = self.wavelet_decomp(side_embeds)
#         fusion_cA, fusion_cD  = self.wavelet_decomp(content_embeds + side_embeds)
#         #print("content_cA.shape:", content_cA.shape, "content_cD.shape:", content_cD.shape, "fusion_cA.shape:", fusion_cA.shape, "fusion_cD.shape:", fusion_cD.shape)
        
#         # 多模态特征融合（增强版）
#         # 低频：跨模态特征拼接+投影
#         #low_fused = self.low_fusion(content_cA * fusion_cA)
#         #low_fused = self.low_fusion(torch.cat([content_cA * fusion_cA, content_cA + fusion_cA], dim=-1)) # torch.Size([26495, 64])
#         #print("torch.cat([content_cA * fusion_cA, content_cA + fusion_cA], dim=-1).shape:", torch.cat([content_cA * fusion_cA, content_cA + fusion_cA], dim=-1).shape)
#         low_fused = content_cA + fusion_cA
#         #print("low_fused.shape:", low_fused.shape)
#         #low_fused = torch.log(content_cA * fusion_cA) * (content_cA * fusion_cA)
#         # 高频：跨模态交互特征
#         #high_fused = self.high_fusion(content_cD * fusion_cD)
#         #high_fused = self.high_fusion(torch.cat([content_cD * fusion_cD, content_cD + fusion_cD], dim=-1))
#         high_fused = content_cD + fusion_cD
#         #print("high_fused.shape")
    
#         # 动态权重学习（基于原始特征）
#         combined = content_embeds + side_embeds
#         # combined = torch.multiply(content_embeds, fusion_embeds)
#         # combined = fusion_embeds
#         # weights = self.attention(torch.concat([low_fused, high_fused], dim=-1))
#         weights = self.attention(combined)
        
#         # 加权融合
#         # reconstructed = self.wavelet_recon(
#         #     weights[:, 0:1] * low_fused,
#         #     weights[:, 1:2] * high_fused
#         # )
#         reconstructed = self.wavelet_recon(
#             low_fused,
#             high_fused
#         )
#         low_freq_interest = self.wavelet_recon(content_cA, fusion_cA)
#         high_freq_interest = self.wavelet_recon(content_cD, content_cD)
#         # low_freq_interest = self.wavelet_recon(low_fused, content_cA)
#         # high_freq_interest = self.wavelet_recon(high_fused, content_cD)
#         # 残差连接与归一化
#         #return self.norm(weights[:, 0:1] * reconstructed + weights[:, 1:2] * combined + combined ) 
#         # return self.norm(reconstructed) + combined, low_freq_interest, high_freq_interest
#         return reconstructed, low_freq_interest, high_freq_interest

        

# class MultiModalWaveletInterestAttention(nn.Module):
#     def __init__(self, embed_dim, wavelet_name='db1'):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.wavelet_name = wavelet_name
        
#         # 小波变换与逆变换（支持自动梯度计算）
#         self.dwt = DWT1DForward(wave=wavelet_name, J=1)
#         self.idwt = DWT1DInverse(wave=wavelet_name)
        
#         # 增强的注意力机制
#         self.attention = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.LayerNorm(embed_dim),
#             nn.GELU(),
#             nn.Linear(embed_dim, embed_dim//2),
#             nn.GELU(),
#             nn.Linear(embed_dim//2, 2),
#             nn.Softmax(dim=-1)
#         )
        
#         # 低频融合投影层
#         self.low_fusion = nn.Sequential(
#             nn.Linear(embed_dim//2 * 2, embed_dim//2),
#             nn.GELU()
#         )
        
#         # 高频融合投影层
#         self.high_fusion = nn.Sequential(
#             nn.Linear(embed_dim//2 * 2, embed_dim//2),
#             nn.GELU()
#         )
        
#         # 残差归一化
#         self.norm = nn.LayerNorm(embed_dim)
        
#     def wavelet_decomp(self, x):
#         """可微分小波分解"""
#         x = x.unsqueeze(1)  # [B,1,D]
#         cA, cD = self.dwt(x)
#         return cA.squeeze(1), cD[0].squeeze(1)
    
#     def wavelet_recon(self, cA, cD):
#         """可微分小波重构"""
#         cA = cA.unsqueeze(1)
#         cD = [cD.unsqueeze(1)]
#         return self.idwt((cA, cD)).squeeze(1)
    
#     def forward(self, content_embeds, fusion_embeds):
#         # 小波分解
#         content_cA, content_cD = self.wavelet_decomp(content_embeds)
#         fusion_cA, fusion_cD = self.wavelet_decomp(fusion_embeds)
#         #print("content_cA.shape:", content_cA.shape, "content_cD.shape:", content_cD.shape, "fusion_cA.shape:", fusion_cA.shape, "fusion_cD.shape:", fusion_cD.shape)
        
#         # 多模态特征融合（增强版）
#         # 低频：跨模态特征拼接+投影
#         #low_fused = self.low_fusion(torch.cat([content_cA * fusion_cA, content_cA + fusion_cA], dim=-1))
#         # print("torch.cat([content_cA * fusion_cA, content_cA + fusion_cA], dim=-1).shape:", torch.cat([content_cA * fusion_cA, content_cA + fusion_cA], dim=-1).shape)
#         low_fused = content_cA * fusion_cA
#         #low_fused = torch.log(content_cA * fusion_cA) * (content_cA * fusion_cA)
#         # 高频：跨模态交互特征
#         #high_fused = self.high_fusion(torch.cat([content_cD * fusion_cD, content_cD + fusion_cD], dim=-1))
#         high_fused = content_cD * fusion_cD
    
#         # 动态权重学习（基于原始特征）
#         combined = content_embeds + fusion_embeds
#         # combined = torch.multiply(content_embeds, fusion_embeds)
#         # combined = fusion_embeds
#         weights = self.attention(combined)
        
#         # 加权融合
#         # reconstructed = self.wavelet_recon(
#         #     weights[:, 0:1] * low_fused,
#         #     weights[:, 1:2] * high_fused
#         # )
#         reconstructed = self.wavelet_recon(
#             low_fused,
#             high_fused
#         )

#         low_freq_interest = self.wavelet_recon(content_cA, fusion_cA)
#         high_freq_interest = self.wavelet_recon(content_cD, content_cD)
#         # 残差连接与归一化
#         #return self.norm(weights[:, 0:1] * reconstructed + weights[:, 1:2] * combined + combined ) 
#         return reconstructed + combined, low_freq_interest, high_freq_interest
