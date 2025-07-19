from pytorch_wavelets import DWT1D, IDWT1D
import torch
import torch.nn as nn

class MMWaveletAlignModule(nn.Module):
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
    def __init__(self, wavelet='db4', level=3):
        """
        多模态小波对齐模块（可微分）
        Args:
            wavelet: 小波基名称（如'db4'）
            level: 分解层数（默认3层）
        """
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        
        # 初始化一维小波分解/重建器（支持PyTorch自动微分）
        self.dwt = DWT1D(wave=wavelet, mode='symmetric', J=self.level)  # 对称填充保持边缘信息
        self.idwt = IDWT1D(wave=wavelet, mode='symmetric')

    def wavelet_decompose(self, x):
        """
        一维小波分解（可微分）
        Args:
            x: 输入张量，形状 (N, L) 或 (N, C, L)（N:样本数，L:序列长度）
        Returns:
            coeffs: 列表，[cA_level, cD_level, cD_level-1, ..., cD_1]（低频+各层高频系数）
        """
        # 调整输入维度为 (N, C, L)，C=1（单通道）
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (N, L) -> (N, 1, L)
        
        # 分解得到低频系数cA和高频系数列表cD（顺序：cD_level, cD_level-1, ..., cD_1）
        cA, cD_list = self.dwt(x)  # cA形状: (N, 1, L//2^level), cD_list各元素形状: (N, 1, L//2^i)
        
        # 重组系数列表为 [cA_level, cD_level, cD_level-1, ..., cD_1]（与原pywt顺序一致）
        coeffs = [cA.squeeze(1)] + [cd.squeeze(1) for cd in cD_list]
        return coeffs

    def wavelet_reconstruct(self, coeffs):
        """
        一维小波重建（可微分）
        Args:
            coeffs: 列表，[cA_level, cD_level, cD_level-1, ..., cD_1]（与分解输出顺序一致）
        Returns:
            rec: 重建张量，形状 (N, L)（与输入x形状一致）
        """
        # 分离低频和高频系数（注意cD顺序需调整为从高层到低层）
        cA = coeffs[0].unsqueeze(1)  # (N, 1, L//2^level)
        cD_list = [cd.unsqueeze(1) for cd in coeffs[1:]]  # 高频系数列表（顺序：cD_level, cD_level-1, ..., cD_1）
        
        # 重建（输出形状与输入x一致）
        rec = self.idwt((cA, cD_list))
        return rec.squeeze(1)  # (N, 1, L) -> (N, L)

    def process_low(self, coeff, modality='image'):
        """
        低频信号处理：SVD降维+归一化（可微分）
        """
        # SVD降维（保留主要特征）
        U, S, V = torch.svd_lowrank(coeff, q=min(coeff.shape)//2)
        recon = U @ torch.diag(S) @ V.T  # 形状保持 (N, D)
        
        # 归一化（不同模态不同策略）
        if modality == 'image':
            # 图像：归一化到[-1, 1]
            recon = 2 * (recon - recon.min(dim=1, keepdim=True)[0]) / (
                recon.max(dim=1, keepdim=True)[0] - recon.min(dim=1, keepdim=True)[0] + 1e-8) - 1
        else:
            # 文本：L2归一化
            recon = torch.nn.functional.normalize(recon, p=2, dim=1)
        return recon

    def fuse_low(self, img_low, txt_low):
        """
        低频信号语义注意力融合（可微分）
        """
        # print("img_low.shape:", img_low.shape, "txt_low.shape:", txt_low.shape)
        attn = torch.stack([img_low, txt_low], dim=1)   # img_low.shape: torch.Size([7050, 14]) txt_low.shape: torch.Size([7050, 14])
        #print("attn.shape:", attn.shape) # attn.shape: torch.Size([7050, 2, 14])
        attn_weights = torch.softmax(attn @ attn.transpose(1, 2), dim=1)   # attn_weights.shape: torch.Size([7050, 2, 2])
        # print("attn_weights.shape:", attn_weights.shape)  attn_weights.shape: torch.Size([7050, 2, 2])
        # 加权融合（保留模态特异性）
        w_img = attn_weights[:, 0:1, 0]  # [N, 1]
        w_txt = attn_weights[:, 0:1, 1]  # [N, 1]
        fused = w_img * img_low + w_txt * txt_low  # [N, D]
        # print("fused.shape:", fused.shape)  # fused.shape: torch.Size([7050, 14])
        return fused

    def fuse_high(self, img_high, txt_high, level_type):
        """
        高频信号多尺度融合（可微分）
        """
        eps = 1e-8
        if level_type == 3:  # 第3级高频（大尺度边缘）：能量对齐
            energy_img = torch.norm(img_high, dim=1, keepdim=True)  # [N, 1]
            energy_txt = torch.norm(txt_high, dim=1, keepdim=True)  # [N, 1]
            fused = (energy_img * img_high + energy_txt * txt_high) / (energy_img + energy_txt + eps)
        elif level_type == 2:  # 第2级高频（中尺度细节）：平均融合
            fused = (img_high + txt_high) / 2
        else:  # 第1级高频（小尺度噪声）：去噪后融合
            # 软阈值去噪（可微分）
            def soft_denoise(x):
                threshold = torch.median(torch.abs(x), dim=1, keepdim=True)[0] / 0.6745  # 鲁棒阈值
                return torch.sign(x) * torch.relu(torch.abs(x) - threshold)
            denoised_img = soft_denoise(img_high)
            denoised_txt = soft_denoise(txt_high)
            fused = (denoised_img + denoised_txt) / 2
            # denoised_img = self.process_low(img_high, modality='image')
            # denoised_txt = self.process_low(txt_high, modality='text')
            # fused = (img_high + txt_high) / 2
        return fused

    def forward(self, image_embedding, text_embedding):
        """
        多模态多尺度小波对齐主流程
        Args:
            image_embedding: 图像模态嵌入，形状 (N, D)
            text_embedding: 文本模态嵌入，形状 (N, D)
        Returns:
            image_embedding_wave: 图像模态小波重建嵌入，(N, D)
            text_embedding_wave: 文本模态小波重建嵌入，(N, D)
            fusion_wave: 融合模态小波重建嵌入，(N, D)
        """
        # 步骤1：小波分解（可微分）
        image_coeffs = self.wavelet_decompose(image_embedding)  # [cA3, cD3, cD2, cD1]
        text_coeffs = self.wavelet_decompose(text_embedding)    # [cA3, cD3, cD2, cD1]

        # 步骤2：多尺度对齐与融合
        fused_coeffs = []
        img_coeffs_proc = []
        txt_coeffs_proc = []
        LORA = False
        for i, (img_coeff, txt_coeff) in enumerate(zip(image_coeffs, text_coeffs)):
            if i == 0:  # 低频分量（cA3）

                if LORA:
                    img_proc = self.process_low(img_coeff, modality='image')
                    txt_proc = self.process_low(txt_coeff, modality='text')

                    fused = self.fuse_low(img_proc, txt_proc)
                else:
                    img_proc = img_coeff
                    txt_proc = txt_coeff
                    fused = self.fuse_low(img_proc, txt_proc)
            else:  # 高频分量（cD3, cD2, cD1）
                level_type = self.level - (i - 1)  # 计算层级（3,2,1）
                img_proc = img_coeff  # 图像模态保留原始系数（或按需处理）
                txt_proc = txt_coeff  # 文本模态保留原始系数（或按需处理）
                fused = self.fuse_high(img_coeff, txt_coeff, level_type)
            img_coeffs_proc.append(img_proc)
            txt_coeffs_proc.append(txt_proc)
            fused_coeffs.append(fused)

        # 步骤3：小波重建（可微分）
        image_embedding_wave = self.wavelet_reconstruct(img_coeffs_proc)
        text_embedding_wave = self.wavelet_reconstruct(txt_coeffs_proc)
        fusion_wave = self.wavelet_reconstruct(fused_coeffs)


        return image_embedding_wave, text_embedding_wave, fusion_wave