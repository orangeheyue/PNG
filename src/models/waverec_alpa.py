"""
Author: orangeheyue@gmail
Paper Reference:
    ACM CIKM 2026: WaveRec: Wavelet Learning for Multimodal Recommendation
Sourece Code:
    https://github.com/orangeai-research/WaveRec
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
import pywt
from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph
from models.MMWA import MultiModalWaveletInterestAttention

class WaveRec(GeneralRecommender):
    '''
        paper: ACM CIKM 2026: WaveRec: Wavelet Learning for Multimodal Recommendation
        code: https://github.com/orangeai-research/WaveRec
    '''
    def __init__(self, config, dataset):
        super(WaveRec, self).__init__(config, dataset)
        self.sparse = True
        self.cl_loss1 = config['cl_loss1']
        self.cl_loss2 = config['cl_loss2']
        self.cl_loss3 = config['cl_loss3']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.image_knn_k = config['image_knn_k']
        self.text_knn_k = config['text_knn_k']
        # print("self.text_knn_k:", self.text_knn_k)
        self.fusion_knn_k = config['fusion_knn_k']
        self.dropout_rate = config['dropout_rate']
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.MMWA = config['MMWA']
        self.MMWI = config['MMWI']
        self.MMCL = config['MMCL']

        # 多模态小波兴趣感知
        # self.mm_wavelet_interest_aware = MultiModalWaveletInterestAttention(embed_dim=self.embedding_dim)
        self.mm_wavelet_interest_aware = MultiModalWaveletInterestAttention(embed_dim=self.embedding_dim, wavelet_name='db4')

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        #print("self.image_knn_k:", self.image_knn_k)
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.image_knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.text_knn_k, self.sparse))
        self.fusion_adj_file = os.path.join(dataset_path, 'fusion_adj_{}_{}.pt'.format(self.fusion_knn_k, self.sparse))

        self.norm_adj = self.get_adj_mat()
        self.R_sprse_mat = self.R
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.image_knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.text_knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda() 

        # self.fusion_adj = self.max_pool_fusion()



        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.query_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        )
        self.query_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        )

        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_f = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_fusion_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.image_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))
        self.text_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))
        self.fusion_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))
        

    # def pre_epoch_processing(self):
    #     pass

    # def max_pool_fusion(self):
    #     image_adj = self.image_original_adj.coalesce()
    #     text_adj = self.text_original_adj.coalesce()

    #     image_indices = image_adj.indices().to(self.device)
    #     image_values = image_adj.values().to(self.device)
    #     text_indices = text_adj.indices().to(self.device)
    #     text_values = text_adj.values().to(self.device)

    #     combined_indices = torch.cat((image_indices, text_indices), dim=1)
    #     combined_indices, unique_idx = torch.unique(combined_indices, dim=1, return_inverse=True)

    #     combined_values_image = torch.full((combined_indices.size(1),), float('-inf')).to(self.device)
    #     combined_values_text = torch.full((combined_indices.size(1),), float('-inf')).to(self.device)

    #     combined_values_image[unique_idx[:image_indices.size(1)]] = image_values
    #     combined_values_text[unique_idx[image_indices.size(1):]] = text_values
    #     combined_values, _ = torch.max(torch.stack((combined_values_image, combined_values_text)), dim=0)

    #     fusion_adj = torch.sparse.FloatTensor(combined_indices, combined_values, image_adj.size()).coalesce()

    #     return fusion_adj

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    # def spectrum_convolution(self, image_embeds, text_embeds):
    #     """
    #     Modality Denoising & Cross-Modality Fusion
    #     """
    #     image_fft = torch.fft.rfft(image_embeds, dim=1, norm='ortho')           
    #     text_fft = torch.fft.rfft(text_embeds, dim=1, norm='ortho')

    #     image_complex_weight = torch.view_as_complex(self.image_complex_weight)   
    #     text_complex_weight = torch.view_as_complex(self.text_complex_weight)
    #     fusion_complex_weight = torch.view_as_complex(self.fusion_complex_weight)

    #     #   Uni-modal Denoising
    #     image_conv = torch.fft.irfft(image_fft * image_complex_weight, n=image_embeds.shape[1], dim=1, norm='ortho')    
    #     text_conv = torch.fft.irfft(text_fft * text_complex_weight, n=text_embeds.shape[1], dim=1, norm='ortho')

    #     #   Cross-modality fusion
    #     fusion_conv = torch.fft.irfft(text_fft * image_fft * fusion_complex_weight, n=text_embeds.shape[1], dim=1, norm='ortho') 
    #     # print("image_conv.shape:", image_conv.shape, "text_conv.shape:", text_conv.shape, "fusion_conv.shape:", fusion_conv.shape)
    #     return image_conv, text_conv, fusion_conv

    def multimodal_multiscale_wavelet_align(self, image_embedding, text_embedding):

        '''
        Desc: 多模态多尺度小波变换对齐
        Args:
            image_embedding: 图像embedding, image_embedding.shape: torch.Size([7050, 64]) 
            text_embedding: 文本embedding, text_embedding.shape: torch.Size([7050, 64]), 7050个物品,每个物品编码为64维向量
        Function:
            对图像和文本嵌入进行多模态多尺度小波变换对齐 ：
            Steps1: 选取模态领域适应的合适小波基['harr', 'db1-20', 'bior1.3'], 选择高通滤波器的下采样分级尺度level,本文默认选择level3, 注意输入的特征向量是一维度，因此暂时不考虑水平、垂直、对角。
            Steps2: Multimodal Wavelet Transform Learning Project： 获取图像模态特征的image_coeffs（一个低通分量，三个高通分量）；文本模态特征的text_coeffs（一个低通分量，三个高通分量）
            Steps3: 多模态多尺度频域空间对齐 Multimodal Multi-Scale Frequency Domain Align：
                    3.1低频信号对齐
                        1.对于低频信号（第 3 级水平低频分量），首先对图像模态和文本模态下的低频信号分别进行LoRA(奇异值分解（SVD）),用于去除冗余信息，保留主要特征。 
                        2.然后,再分别对进行归一化，对图像低频信号，可将其像素值归一化到特定范围，如 [0, 1] 或 [-1, 1]，以消除不同图像之间的亮度差异等影响。 对于文本低频信号，可对词向量或特征向量进行归一化，使不同文本的特征具有可比性，例如采用 L2 归一化。
                        3.然后再将LoRa和归一化后的两个模态的特征向量进行对齐，对齐方式采用语义注意力机制， 生成一个新的融合模态第 3 级水平低频分量
                    3.2 高频信号多尺度对齐 TODO：高频信号不同等级能量是否才用不同对齐方式？
                        1.对图像和文本的第 3 级水平高频分量进行对齐，对齐方式采用频域能量对齐，生成一个新的融合模态第 3 级水平高频分量
                        2.对图像和文本的第 2 级水平高频分量进行对齐， 生成一个新的融合模态第 2 级水平高频分量
                        3.对图像和文本的第 1 级水平高频分量分别先进行去噪，再进行对齐， 生成一个新的融合模态第 1 级水平高频分量
            Steps4： Multimodal Wavelet Inverse Transform 多模态小波重建

                    对融合模态的低频分量和3个高频分量进行小波逆变换，重建为一个多模态频域对齐以及语义域融合的特征向量fusion_wave
                    对图像模态、文本模态经过上述变换后，进行小波逆变换，重建为新的图像image_embedding_wave、文本模态特征向量text_embedding_wave。
        Returns:
            image_embedding_wave.shape: torch.Size([7050, 64]) text_embedding_wave.shape: torch.Size([7050, 64]) fusion_wave.shape: torch.Size([7050, 64])
        '''
        device = image_embedding.device
        # print("device:", device)
        # print("image_embedding:", image_embedding.shape)
        # 转换为numpy数组进行小波变换
        # image_np = image_embedding.detach().cpu().numpy()
        # text_np = text_embedding.detach().cpu().numpy()

        # 定义小波类型
        wavelet = 'db4'
        level = 3 

        # 对图像和文本嵌入进行小波变换
        '''
        原始信号 如果输入是 (256, 256) 的图像，axis=1（水平方向分解），level=3：
    │
        ├── cA3（最低频，最模糊的近似）      主体信号，轮廓 （保留）  (256, 32)  第 3 级水平低频分量
        │
        ├── cD3（第3级细节，较大尺度的高频） 大尺度边缘 最精细的细节信息 (256, 32) 第 3 级水平高频分量
        │
        ├── cD2（第2级细节，中等尺度的高频） 中尺度细节，更细微的边缘和纹理变化 (256, 64) 第 2 级水平高频分量
        │
        └── cD1（第1级细节，最细尺度的高频） 高频噪声->去噪 (256, 128) 第 1 级水平高频分量
        '''
        image_coeffs = self.wavelet_decompose(image_embedding, wavelet, level=level, axis=1, device=device)
        text_coeffs = self.wavelet_decompose(text_embedding, wavelet, level=level, axis=1, device=device)

        '''
        image_coeffs shapes:
            wavelet_decompose Level 4: (7050, 14)
            wavelet_decompose Level 3: (7050, 14)
            wavelet_decompose Level 2: (7050, 21)
            wavelet_decompose Level 1: (7050, 35)
        text_coeffs shapes:
            wavelet_decompose Level 4: (7050, 14)
            wavelet_decompose Level 3: (7050, 14)
            wavelet_decompose Level 2: (7050, 21)
            wavelet_decompose Level 1: (7050, 35)
        '''
        # len(image_coeffs): 4 len(text_coeffs): 4
        # print("len(image_coeffs):", len(image_coeffs), "len(text_coeffs):", len(text_coeffs))
        # 对每一级系数进行对齐和融合
        # ===================== 多模态多尺度频域对齐 =====================
        fused_coeffs = []
        img_coeffs_proc = []
        txt_coeffs_proc = []

        for i, (img_coeff, txt_coeff) in enumerate(zip(image_coeffs, text_coeffs)):        
            # 低频分量
            if i == 0: 
                img_proc = self.process_low(img_coeff, modality='image')
                txt_proc = self.process_low(txt_coeff, modality='text')
                # 融合
                fused = self.fuse_low(img_proc, txt_proc)
            # 高频分量
            else:       
                level_type = len(image_coeffs) - i  # 计算当前层级(3,2,1)
                # 直接传递原始系数用于重建
                img_proc = img_coeff  
                txt_proc = txt_coeff
                # 融合处理
                fused = self.fuse_high(img_coeff, txt_coeff, level_type)
            # 保存处理结果
            img_coeffs_proc.append(img_proc)
            txt_coeffs_proc.append(txt_proc)
            fused_coeffs.append(fused)


        # 进行小波逆变换
        image_embedding_wave = self.wavelet_reconstruct(img_coeffs_proc, wavelet, axis=1, device=device)
        text_embedding_wave = self.wavelet_reconstruct(txt_coeffs_proc, wavelet, axis=1, device=device)
        fusion_wave = self.wavelet_reconstruct(fused_coeffs, wavelet, axis=1, device=device)

        return image_embedding_wave, text_embedding_wave, fusion_wave

    def wavelet_decompose(self, x, wavelet='db4', level=3, axis=1, device='cuda'):
        '''
            小波变换分解
            Steps1: 选取模态领域适应的合适小波基['harr', 'db1-20', 'bior1.3'], 选择高通滤波器的下采样分级尺度level,本文默认选择level3, 注意输入的特征向量是一维度，因此暂时不考虑水平、垂直、对角。

        '''
        x_np = x.detach().cpu().numpy()
        coeffs = pywt.wavedec(x_np, wavelet, level=level, axis=axis)
        return [torch.tensor(c, device=device, dtype=torch.float32) for c in coeffs]
        # return pywt.wavedec(x, wavelet, level=level, axis=axis)

        # ===================== 小波重建 =====================
    def wavelet_reconstruct(self, coeffs, wavelet='db4', axis=1, device='cuda'):
        coeffs_np = [c.detach().cpu().numpy() for c in coeffs]
        rec = pywt.waverec(coeffs_np, wavelet, axis=1)
        return torch.tensor(rec, device=device, dtype=torch.float32)

    def process_low(self, coeff, modality='image'):
        """
            低频信号: SVD降维+归一化
        """
        # print("coeff:", coeff.shape)
        # SVD降维
        U, S, V = torch.svd_lowrank(coeff, q=min(coeff.shape)//2)
        recon = U @ torch.diag(S) @ V.T
        # print("recon:", recon.shape)
        # 归一化
        if modality == 'image':
            recon = 2 * (recon - recon.min())/(recon.max() - recon.min() + 1e-8) - 1
        else:
            recon = torch.nn.functional.normalize(recon, p=2, dim=1)
        return recon

    def fuse_low(self, img_low, txt_low):
        """
            低频信号对齐: 基于语义的attention
        """
        # 语义注意力机制
        attn = torch.stack([img_low, txt_low], dim=1)  # [N, 2, D]
        attn_weights = torch.softmax(attn @ attn.transpose(1, 2), dim=1)  # [N, 2, 2]
        # 加权融合
        w_img = attn_weights[:, 0, 0].unsqueeze(1)
        w_txt = attn_weights[:, 0, 1].unsqueeze(1)
        return w_img * img_low + w_txt * txt_low


    def fuse_high(self, img_high, txt_high, level_type):
        """
            低频信号对齐: 基于语义的attention
        """
        eps = 1e-8  # 防止除以零
        
        if level_type == 3 :  # 能量对齐
            energy_img = torch.norm(img_high, dim=1, keepdim=True)
            energy_txt = torch.norm(txt_high, dim=1, keepdim=True)
            return (energy_img*img_high + energy_txt*txt_high)/(energy_img + energy_txt + eps)
            # return (img_high + txt_high) / 2
            
        elif level_type == 2:  # 直接平均
            '''
                simple fusion TODO: 
            '''
            # return (img_high + txt_high) / 2
            return self.process_low(coeff=img_high, modality='image') + self.process_low(coeff=txt_high, modality='text') / 2
            # def denoise(x):
            #     threshold = torch.median(torch.abs(x)) / 0.6745
            #     return torch.sign(x) * torch.relu(torch.abs(x) - threshold)
            # return (denoise(img_high) + denoise(txt_high)) / 2
            
        elif level_type == 1:  # 去噪后融合
            # def denoise(x):
            #     threshold = torch.median(torch.abs(x)) / 0.6745
            #     return torch.sign(x) * torch.relu(torch.abs(x) - threshold)
            # return (denoise(img_high) + denoise(txt_high)) / 2
            return self.process_low(coeff=img_high, modality='image') + self.process_low(coeff=txt_high, modality='text') / 2
            #return (img_high + txt_high) / 2

        
    def build_item_item_fusion_graph(self, fusion_embedding):
        '''
            构造item-item fusion graph
        '''
        if os.path.exists(self.fusion_adj_file):
            fusion_adj = torch.load(self.fusion_adj_file)
        else:
            fusion_adj = build_sim(fusion_embedding.detach())
            fusion_adj = build_knn_normalized_graph(fusion_adj, topk=self.fusion_knn_k, is_sparse=self.sparse, norm_type='sym')
            torch.save(fusion_adj, self.fusion_adj_file)
        
        return fusion_adj.cuda() 

    def forward(self, adj, train=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
        
        # (Multimodal Multi-Sacle Wavelet align, MMWA)
        if self.MMWA:
            image_embedding, text_embedding, fusion_embedding = self.multimodal_multiscale_wavelet_align(image_feats, text_feats)
        else:
            image_embedding = image_feats
            text_embedding = text_feats

        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_embedding))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_embedding))
        fusion_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_f(fusion_embedding))


        # load or build fusion item-item graph
        # self.fusion_adj = self.build_item_item_fusion_graph(fusion_embedding)
        #   User-Item (Behavioral) View
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]

        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings
        # print("content_embeds.shape:", content_embeds.shape) # content_embeds.shape: torch.Size([26495, 64])

        #   Item-Item Modality Specific and Fusion views
        #   Image-view
        if self.sparse:
            for i in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        else:
            for i in range(self.n_layers):
                image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)
 
        #   Text-view
        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        #  Faster Multimodal Fusion view
        self.fusion_adj = torch.sqrt(self.text_original_adj * self.text_original_adj + self.image_original_adj * self.image_original_adj) / 2
        if self.sparse:
            for i in range(self.n_layers): 
                fusion_item_embeds = torch.sparse.mm(self.fusion_adj, fusion_item_embeds)
        else:
            for i in range(self.n_layers):
                fusion_item_embeds = torch.mm(self.fusion_adj, fusion_item_embeds)
        fusion_user_embeds = torch.sparse.mm(self.R, fusion_item_embeds)
        fusion_embeds = torch.cat([fusion_user_embeds, fusion_item_embeds], dim=0)

        #   Modality-aware Preference Module
        fusion_att_v, fusion_att_t = self.query_v(fusion_embeds), self.query_t(fusion_embeds)
        fusion_soft_v = self.softmax(fusion_att_v)
        agg_image_embeds = fusion_soft_v * image_embeds

        fusion_soft_t = self.softmax(fusion_att_t)
        agg_text_embeds = fusion_soft_t * text_embeds

        # 模态偏好感知
        image_prefer = self.gate_image_prefer(content_embeds)
        text_prefer = self.gate_text_prefer(content_embeds)

        fusion_prefer = self.gate_fusion_prefer(content_embeds)
        image_prefer, text_prefer, fusion_prefer = self.dropout(image_prefer), self.dropout(text_prefer), self.dropout(fusion_prefer)
        # print("image_prefer.shape:", image_prefer.shape, "text_prefer.shape:", text_prefer.shape, "fusion_prefer.shape:", fusion_prefer.shape)
        # print("agg_image_embeds.shape:", agg_image_embeds.shape, "agg_text_embeds.shape:", agg_text_embeds.shape, "fusion_embeds.shape:", fusion_embeds.shape)
        '''
        content_embeds.shape: torch.Size([26495, 64])
        image_prefer.shape: torch.Size([26495, 64]) text_prefer.shape: torch.Size([26495, 64]) fusion_prefer.shape: torch.Size([26495, 64])
        agg_image_embeds.shape: torch.Size([26495, 64]) agg_text_embeds.shape: torch.Size([26495, 64]) fusion_embeds.shape: torch.Size([26495, 64])
        '''
        agg_image_embeds = torch.multiply(image_prefer, agg_image_embeds) # 图像模态偏好感知
        agg_text_embeds = torch.multiply(text_prefer, agg_text_embeds) # 文本模态偏好感知
        fusion_embeds = torch.multiply(fusion_prefer, fusion_embeds) # 兴趣偏好感知
        # 公共兴趣和个性化兴趣偏好感知
        # fusion_embeds = self.mm_wavelet_interest_aware(content_embeds, fusion_embeds)
        #fusion_embeds = self.mm_wavelet_interest_aware(agg_image_embeds, agg_text_embeds)
        #print("fusion_embeds:", fusion_embeds)
        # print("content_embeds.shape:", content_embeds.shape)
        side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds, fusion_embeds]), dim=0) 
        #side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds]), dim=0) 
        # side_embeds = torch.mean(torch.stack([fusion_embeds]), dim=0) 

        # Ablation Study: TODO MMWI
        if self.MMWI:
            all_embeds, low_freq_interest, high_freq_interest = self.mm_wavelet_interest_aware(content_embeds, side_embeds)
        else:
            all_embeds = content_embeds + side_embeds
            low_freq_interest, high_freq_interest = 0, 0
        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds, low_freq_interest, high_freq_interest

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds, low_freq_interest, high_freq_interest = self.forward(
            self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)

        #item-item constractive loss
        cl_loss1 = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(side_embeds_users[users], content_embeds_user[users], 0.2) 
        #user-item constractive loss
        cl_loss2 = self.InfoNCE(u_g_embeddings, content_embeds_items[pos_items], 0.2) + self.InfoNCE(u_g_embeddings, side_embeds_items[pos_items], 0.2)

        # hot-interest cold-interest
        if self.MMWI:
            low_freq_embeds_users, low_freq_embeds_items = torch.split(low_freq_interest, [self.n_users, self.n_items], dim=0)
            high_freq_embeds_user, high_freq_embeds_items = torch.split(high_freq_interest, [self.n_users, self.n_items], dim=0)
            cl_loss3 = self.InfoNCE(low_freq_embeds_items[pos_items], high_freq_embeds_items[pos_items], 0.2) + self.InfoNCE(low_freq_embeds_users[users], high_freq_embeds_user[users], 0.2)
        else:
            cl_loss3 = 0.0

        # return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss1 + self.cl_loss * 0.1 * cl_loss2 + self.cl_loss * 0.1 * cl_loss3
        return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss1 * cl_loss1 + self.cl_loss2  * cl_loss2 + self.cl_loss3 * cl_loss3

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores


# class MultiModalWaveletInterestAttention(nn.Module):
#     '''
#     Desc:
#         当前的方法无法分离用户对物品的大众兴趣和小众兴趣，
#         -------------------------------------------
#         优点：

#             小波变换能有效分离频域特征，适合兴趣分解

#             低频/高频的区分符合热门/小众兴趣的特性

#             Attention加权可以动态平衡两种兴趣
#         -------------------------------------------
#         基于小波变换的，多模态兴趣偏好感知, 初步思路是：
#         分别对content_embeds和fusion_embeds进行小波分解，然后在低频(对多模态公共兴趣，热门兴趣)和高频特征(个性化兴趣，小众兴趣)中进行用户兴趣感，可以用torch.multiply或者其他优雅的发方式， 
#         然后，将融合后的低频特征(热门兴趣) 和 高频特征(小众兴趣) 设计一个attention进行加权
#         最后再小波逆变换。
#     Args:
#         content_embeds.shape: torch.Size([26495, 64])  用户和物品的embedding
#         fusion_embeds.shape: torch.Size([26495, 64])   多模态融合的embeeding
#     Returns:
#         mm_interest_prefer_aware_embeds: torch.Size([26495, 64]) 
#     '''
#     def __init__(self, embed_dim, wavelet_name='db1'):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.wavelet_name = wavelet_name
        
#         # 可学习的attention权重
#         self.attention = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim //2),
#             nn.BatchNorm1d(embed_dim //2),
#             nn.LeakyReLU(),
#             nn.Linear(embed_dim // 2, 2),
#             nn.Softmax(dim=-1)
#         )
        
#     def wavelet_decomp(self, x):
#         """小波分解为低频和高频成分"""
#         coeffs = pywt.wavedec(x.detach().cpu().numpy(), self.wavelet_name, level=1, axis=1)
#         cA = torch.from_numpy(coeffs[0]).to(x.device)  # 低频
#         cD = torch.from_numpy(coeffs[1]).to(x.device)  # 高频
#         # =print("coeffs[0].shape:", coeffs[0].shape, "coeffs[1].shape:", coeffs[0].shape)
#         # if coeffs[0].shape == coeffs[1].shape:
#         #     self.wavelet_decomp_dim = coeffs[0].shape[1]
#         return cA, cD
    
#     def wavelet_recon(self, cA, cD):
#         """小波重构"""
#         coeffs = [cA.detach().cpu().numpy(), cD.detach().cpu().numpy()]
#         return torch.from_numpy(pywt.waverec(coeffs, self.wavelet_name, axis=1)).to(cA.device)
    
#     def forward(self, content_embeds, fusion_embeds):
#         # 小波分解
#         content_cA, content_cD = self.wavelet_decomp(content_embeds)
#         fusion_cA, fusion_cD = self.wavelet_decomp(fusion_embeds)
        
#         # 兴趣感知融合
#         # 低频部分(热门兴趣)用元素相乘增强共享特征
#         low_freq = content_cA * fusion_cA # low_freq.shape: torch.Size([26495, 32])
#         # print("low_freq.shape:", low_freq.shape)
#         # 高频部分(个性化兴趣)用加权平均
#         high_freq = (content_cD * fusion_cD) # high_freq.shape: torch.Size([26495, 32])
#         # 动态权重学习
#         combined = torch.cat([low_freq, high_freq], dim=-1) # combined.shape torch.Size([26495, 64])
#         # print("combined.shape", combined.shape)
#         weights = self.attention(combined)  # [batch, 2]
#         # print("weights:", weights)
#         # 加权融合  
#         # fused = weights[:, 0:1] * low_freq + weights[:, 1:2] * high_freq
#         low_freq_fused, high_freq_fused = weights[:, 0:1] * low_freq, weights[:, 1:2] * high_freq

#         # low_freq_fused, high_freq_fused = self.optimized_fusion(content_cA, content_cD, fusion_cA, fusion_cD)
#         # 小波重构
#         output = self.wavelet_recon(low_freq_fused, high_freq_fused)
#         # # 残差连接
#         return output + content_embeds
#         # print("output.shape:", output.shape)
#         return output

#     def optimized_fusion(self, content_cA, content_cD, fusion_cA, fusion_cD):
#         # 低频：门控双线性融合
#         gate = torch.sigmoid(
#             torch.sum(content_cA * fusion_cA, dim=-1, keepdim=True) / 
#             (torch.norm(content_cA, dim=-1, keepdim=True) * 
#             torch.norm(fusion_cA, dim=-1, keepdim=True) + 1e-6)
#         )
#         low_freq = gate * (content_cA * fusion_cA)
        
#         # 高频：自适应残差融合
#         residual = fusion_cD - content_cD
#         adapt_gate = torch.sigmoid(
#             nn.Linear(content_cD.shape[-1], 1)(residual)
#         )
#         high_freq = content_cD + adapt_gate * residual
        
#         return low_freq, high_freq

# class MultiModalWaveletInterestAttention(nn.Module):
#     def __init__(self, embed_dim, wavelet_name='db1'):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.wavelet_name = wavelet_name
        
#         # 可学习的attention权重
#         self.attention = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim // 2),
#             nn.BatchNorm1d(embed_dim // 2),
#             nn.LeakyReLU(),
#             nn.Linear(embed_dim // 2, 2),
#             nn.Softmax(dim=-1)
#         )
        
#         # 高频融合的门控网络
#         self.adapt_gate = nn.Sequential(
#             nn.Linear(embed_dim // 2, embed_dim // 2),
#             nn.LeakyReLU(),
#             nn.Linear(embed_dim // 2, 1),
#             nn.Sigmoid()
#         )
        
#     def wavelet_decomp(self, x):
#         """设备安全的小波分解"""
#         # 确保在CPU上计算
#         x_np = x.detach().cpu().numpy()
        
#         # 处理奇数长度
#         if x_np.shape[1] % 2 != 0:
#             x_np = np.pad(x_np, [(0,0), (0,1)], mode='edge')
            
#         coeffs = pywt.wavedec(x_np, self.wavelet_name, level=1, axis=1)
#         cA = torch.from_numpy(coeffs[0]).float().to(x.device)  # 低频
#         cD = torch.from_numpy(coeffs[1]).float().to(x.device)  # 高频
#         return cA, cD
    
#     def wavelet_recon(self, cA, cD, target_dim):
#         """设备安全的小波重构"""
#         # 确保在CPU上计算
#         cA_np = cA.detach().cpu().numpy()
#         cD_np = cD.detach().cpu().numpy()
        
#         coeffs = [cA_np, cD_np]
#         recon = pywt.waverec(coeffs, self.wavelet_name, axis=1)
        
#         # 调整维度
#         if recon.shape[1] > target_dim:
#             recon = recon[:, :target_dim]
#         elif recon.shape[1] < target_dim:
#             recon = np.pad(recon, [(0,0), (0, target_dim - recon.shape[1])], mode='constant')
            
#         return torch.from_numpy(recon).float().to(cA.device)
    
#     def optimized_fusion(self, content_cA, content_cD, fusion_cA, fusion_cD):
#         """设备安全的优化融合"""
#         # 低频：门控双线性融合
#         norm_product = (torch.norm(content_cA, dim=-1, keepdim=True) * 
#                        torch.norm(fusion_cA, dim=-1, keepdim=True) + 1e-6)
#         gate = torch.sigmoid(
#             torch.sum(content_cA * fusion_cA, dim=-1, keepdim=True) / norm_product
#         )
#         low_freq = gate * (content_cA * fusion_cA)
        
#         # 高频：自适应残差融合
#         residual = fusion_cD - content_cD
#         adapt_gate = self.adapt_gate(residual)
#         high_freq = content_cD + adapt_gate * residual
        
#         return low_freq, high_freq
    
#     def forward(self, content_embeds, fusion_embeds):
#         # 输入验证
#         assert content_embeds.device == fusion_embeds.device
#         original_dim = content_embeds.size(1)
        
#         # 小波分解
#         content_cA, content_cD = self.wavelet_decomp(content_embeds)
#         fusion_cA, fusion_cD = self.wavelet_decomp(fusion_embeds)
        
#         # 优化融合
#         low_freq, high_freq = self.optimized_fusion(
#             content_cA, content_cD, fusion_cA, fusion_cD
#         )
        
#         # 动态权重学习
#         combined = torch.cat([low_freq, high_freq], dim=-1)
#         weights = self.attention(combined)
        
#         # 加权融合
#         fused_low = weights[:, 0:1] * low_freq
#         fused_high = weights[:, 1:2] * high_freq
        
#         # 小波重构
#         output = self.wavelet_recon(fused_low, fused_high, original_dim)
        
#         return output + content_embeds  # 残差连接