"""
Author: orangeheyue@gmail
Paper Reference:
	IEEE AAAI 2026: PNGRec: Popular-Niche Wavelet Graph Learning for Multimodal Recommendation
Sourece Code:
	https://github.com/orangeai-research/PNGRec.git
	https://github.com/orangeheyue/PNGRec.git
"""


import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph

# from models.uig import UIG
# from models.UIG import UIG 
# from models.UIG2 import GraphWaveletDecomposer
# from models.UIGV1 import UIG
# from models.UIGV2 import UIG
# from models.UIGV3 import UIG
# from models.mmp import WaveletInterestNet

from models.png import PNG

class PNGRec(GeneralRecommender):
	def __init__(self, config, dataset):
		super(PNGRec, self).__init__(config, dataset)
		self.sparse = True
		self.cl_loss = config['cl_loss']
		self.n_ui_layers = config['n_ui_layers']
		self.embedding_dim = config['embedding_size']
		self.n_layers = config['n_layers']
		self.reg_weight = config['reg_weight']
		self.image_knn_k = config['image_knn_k']
		self.text_knn_k = config['text_knn_k']
		self.dropout_rate = config['dropout_rate']
		self.dropout = nn.Dropout(p=self.dropout_rate)

		self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

		self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
		self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
		nn.init.xavier_uniform_(self.user_embedding.weight)
		nn.init.xavier_uniform_(self.item_id_embedding.weight)

		dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
		image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.image_knn_k, self.sparse))
		text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.text_knn_k, self.sparse))

		self.norm_adj = self.get_adj_mat()
		self.R_sprse_mat = self.R
		self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
		self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)
		
		# 初始化模型
		# model = UIG(self.R)
		# # 前向传播
		# self.UIPG, self.UING = model.forward()
		model = PNG(self.R, wavelet='db5', level=3)
		self.UIPG, self.UING, self.quads = model.forward()
		# num_users, num_items = self.R.shape[0], self.R.shape[1]
		# model = UIG(self.norm_adj, num_users, num_items)
		# self.UIPG, self.UING, self.UIPPG, self.UIPNG, self.UINPG, self.UINNG = model.forward()
		# UIG_Model = UIG(self.norm_adj, wavelet='haar', level=1)
		# self.UIPG, self.UING = UIG_Model.forward() 
	
		# 初始化分解器
		#decomposer = GraphWaveletDecomposer(wavelet_name='db4', threshold=0.05)
		# 方法1: 基本分解
		#self.UIPG, self.UING  = decomposer.decompose_graph(self.norm_adj, levels=3)
		# 方法2: 自适应分解(考虑用户和物品流行度)
		#self.UIPG, self.UING = decomposer.adaptive_decompose(self.norm_adj, levels=3)
		# model = UIG(self.norm_adj, rank=300)
		# self.UIPG, self.UING = model.forward()

		# 初始化跨模态注意力
		self.cross_mm_attentoin = CrossModalAttention(self.embedding_dim)


		# self.png_model = WaveletInterestNet().cuda()

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

		self.fusion_adj = self.max_pool_fusion()

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
		

	def pre_epoch_processing(self):
		pass

	def max_pool_fusion(self):
		image_adj = self.image_original_adj.coalesce()
		text_adj = self.text_original_adj.coalesce()

		image_indices = image_adj.indices().to(self.device)
		image_values = image_adj.values().to(self.device)
		text_indices = text_adj.indices().to(self.device)
		text_values = text_adj.values().to(self.device)

		combined_indices = torch.cat((image_indices, text_indices), dim=1)
		combined_indices, unique_idx = torch.unique(combined_indices, dim=1, return_inverse=True)

		combined_values_image = torch.full((combined_indices.size(1),), float('-inf')).to(self.device)
		combined_values_text = torch.full((combined_indices.size(1),), float('-inf')).to(self.device)

		combined_values_image[unique_idx[:image_indices.size(1)]] = image_values
		combined_values_text[unique_idx[image_indices.size(1):]] = text_values
		combined_values, _ = torch.max(torch.stack((combined_values_image, combined_values_text)), dim=0)

		fusion_adj = torch.sparse.FloatTensor(combined_indices, combined_values, image_adj.size()).coalesce()

		return fusion_adj

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

	def spectrum_convolution(self, image_embeds, text_embeds):
		"""
		Modality Denoising & Cross-Modality Fusion
		"""
		image_fft = torch.fft.rfft(image_embeds, dim=1, norm='ortho')           
		text_fft = torch.fft.rfft(text_embeds, dim=1, norm='ortho')

		image_complex_weight = torch.view_as_complex(self.image_complex_weight)   
		text_complex_weight = torch.view_as_complex(self.text_complex_weight)
		fusion_complex_weight = torch.view_as_complex(self.fusion_complex_weight)

		#   Uni-modal Denoising
		image_conv = torch.fft.irfft(image_fft * image_complex_weight, n=image_embeds.shape[1], dim=1, norm='ortho')    
		text_conv = torch.fft.irfft(text_fft * text_complex_weight, n=text_embeds.shape[1], dim=1, norm='ortho')

		#   Cross-modality fusion
		fusion_conv = torch.fft.irfft(text_fft * image_fft * fusion_complex_weight, n=text_embeds.shape[1], dim=1, norm='ortho') 
		
		return image_conv, text_conv, fusion_conv
	
	# # 轻量优化版本
	# def forward_ui_gcn(self, adj):
	# 	item_embeds = self.item_id_embedding.weight
	# 	user_embeds = self.user_embedding.weight
	# 	# 初始嵌入（用户+物品）
	# 	ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
	# 	ego_embeddings2 = torch.cat([user_embeds, item_embeds], dim=0)
	# 	ego_embeddings3 = torch.cat([user_embeds, item_embeds], dim=0)
	# 	all_embeddings = [ego_embeddings]
		
	# 	# 合并邻接矩阵（确保在同一设备）
	# 	adj = adj 
	# 	adj = adj.to(ego_embeddings.device)  # 显式指定邻接矩阵设备
		
	# 	# 可学习的层权重（关键修复：创建时就放到嵌入所在设备）
	# 	if not hasattr(self, 'layer_weights'):  # 避免重复初始化
	# 		self.layer_weights = nn.Parameter(
	# 			torch.ones(self.n_ui_layers + 1, device=ego_embeddings.device)  # 直接在GPU上创建
	# 		)
		
	# 	# GCN消息传递
	# 	for i in range(self.n_ui_layers):
	# 		ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
	# 		all_embeddings.append(ego_embeddings)
	# 	for i in range(self.n_ui_layers):
	# 		ego_embeddings2 = torch.sparse.mm(self.UIPG, ego_embeddings2)
	# 		all_embeddings.append(ego_embeddings2)
	# 	for i in range(self.n_ui_layers):
	# 		ego_embeddings3 = torch.sparse.mm(self.UING, ego_embeddings3)
	# 		all_embeddings.append(ego_embeddings3)

		
		
	# 	# 动态加权融合（确保所有张量在同一设备）
	# 	all_embeddings = torch.stack(all_embeddings, dim=1)  # [N, L+1, D]
	# 	attn = torch.softmax(self.layer_weights, dim=0)  # 层权重归一化（已在GPU）
	# 	# 扩展维度时保持设备一致
	# 	attn = attn.unsqueeze(0).unsqueeze(-1)  # [1, L+1, 1]
	# 	content_embeds = torch.sum(all_embeddings * attn, dim=1)  # [N, D]
		
	# 	return content_embeds
	
	def forward_ui_gcn(self, adj):
		item_embeds = self.item_id_embedding.weight
		user_embeds = self.user_embedding.weight
		# 初始嵌入（用户+物品）
		ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
		ego_embeddings2 = ego_embeddings.clone()  # 使用克隆而不是重复拼接
		ego_embeddings3 = ego_embeddings.clone()
		
		# 为三个GCN分别创建嵌入列表
		all_embeddings1 = [ego_embeddings]
		all_embeddings2 = [ego_embeddings2]
		all_embeddings3 = [ego_embeddings3]
		
		# 合并邻接矩阵（确保在同一设备）
		adj = adj.to(ego_embeddings.device)  # 显式指定邻接矩阵设备
		
		# 可学习的层权重（关键修复：调整维度以匹配总层数）
		total_layers = (self.n_ui_layers + 1) * 3  # 三个GCN，每个有n_ui_layers+1层
		if not hasattr(self, 'layer_weights') or self.layer_weights.size(0) != total_layers:
			self.layer_weights = nn.Parameter(
				torch.ones(total_layers, device=ego_embeddings.device)
			)
		
		# GCN消息传递
		for i in range(self.n_ui_layers):
			ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
			all_embeddings1.append(ego_embeddings)
		
		for i in range(self.n_ui_layers):
			ego_embeddings2 = torch.sparse.mm(self.UIPG, ego_embeddings2)
			all_embeddings2.append(ego_embeddings2)
		
		for i in range(self.n_ui_layers):
			ego_embeddings3 = torch.sparse.mm(self.UING, ego_embeddings3)
			all_embeddings3.append(ego_embeddings3)
		
		# 合并所有嵌入
		all_embeddings = all_embeddings1 + all_embeddings2 + all_embeddings3
		
		# 动态加权融合（确保所有张量在同一设备）
		all_embeddings = torch.stack(all_embeddings, dim=1)  # [N, total_layers, D]
		attn = torch.softmax(self.layer_weights, dim=0)  # 层权重归一化
		attn = attn.unsqueeze(0).unsqueeze(-1)  # [1, total_layers, 1]
		content_embeds = torch.sum(all_embeddings * attn, dim=1)  # [N, D]
		
		return content_embeds

	def user_item_gcn_layer(self,
							adj: torch.Tensor, 
							n_ui_layers: int) -> torch.Tensor:
		"""
		用户-物品行为视图的嵌入传播函数
		
		参数:
			user_embedding: 用户ID嵌入层 (num_users x embed_dim)
			item_id_embedding: 物品ID嵌入层 (num_items x embed_dim)
			adj: 归一化的用户-物品邻接矩阵 (sparse tensor, (num_users + num_items) x (num_users + num_items))
			n_ui_layers: 图卷积层数
			
		返回:
			content_embeds: 传播后的融合嵌入 (num_users + num_items) x embed_dim)
		"""
		# 获取初始嵌入
		item_embeds = self.item_id_embedding.weight
		user_embeds = self.user_embedding.weight
		
		# 拼接用户和物品嵌入
		ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
		all_embeddings = [ego_embeddings]  # 存储各层嵌入
		
		# 多层图传播
		for _ in range(n_ui_layers):
			side_embeddings = torch.sparse.mm(adj, ego_embeddings)
			ego_embeddings = side_embeddings
			all_embeddings.append(ego_embeddings)
		
		# 合并各层嵌入
		all_embeddings = torch.stack(all_embeddings, dim=1)
		all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
		content_embeds = all_embeddings
		
		return content_embeds

	def item_item_gcn_layer(self, original_adj, item_embeds, sparse, n_layers, R):
		if sparse:
			for i in range(n_layers):
				item_embeds = torch.sparse.mm(original_adj, item_embeds)
		else:
			for i in range(n_layers):
				item_embeds = torch.mm(original_adj, item_embeds)
		user_embeds = torch.sparse.mm(R, item_embeds)
		embeds = torch.cat([user_embeds, item_embeds], dim=0)
		return embeds

	def user_popular_niche_gcn_layer(self, R, item_embeds): 
		'''
			R: 用户-物品邻接矩阵 NxM
			item_embeds: 物品嵌入 MxD 
			n_layers: 图卷积层数
			返回: 用户-物品流行兴趣嵌入 NxD
			注意: 该函数用于计算用户对物品的流行兴趣,通过多层图卷积传播物品嵌入到用户最终得到用户对物品的流行兴趣嵌入
			fusion_user_embeds_UIPG = torch.sparse.mm(self.UIPG, fusion_item_embeds) #  torch.Size([19445, 7050]) x torch.Size([7050, 64]) -> torch.Size([19445, 64])
			fusion_embeds_UIPG = torch.cat([fusion_user_embeds_UIPG, fusion_item_embeds], dim=0) # fusion_embeds_uipg.shape: torch.Size([19445, 64]) + torch.Size([7050, 64]) -
		'''

		# for i in range(n_layers):
		# 	item_embeds = torch.sparse.mm(R, item_embeds)
		user_embeds = torch.sparse.mm(R, item_embeds)
		embeds = torch.cat([user_embeds, item_embeds], dim=0)
		return embeds

	def forward(self, adj, train=False):
		if self.v_feat is not None:
			image_feats = self.image_trs(self.image_embedding.weight)
		if self.t_feat is not None:
			text_feats = self.text_trs(self.text_embedding.weight)

		#   Spectrum Modality Fusion
		# image_conv, text_conv, fusion_conv = self.spectrum_convolution(image_feats, text_feats)
		image_conv, text_conv = image_feats, text_feats 
		# fusion_conv = torch.sqrt(image_conv *image_conv +  text_conv * text_conv) # 融合视图
		fusion_conv = self.cross_mm_attentoin(image_conv, text_conv)
		image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_conv))
		text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_conv))
		fusion_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_f(fusion_conv))

		# User-Item (Behavioral) View
		# item_embeds = self.item_id_embedding.weight
		# user_embeds = self.user_embedding.weight
		# ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
		# all_embeddings = [ego_embeddings]

		# for i in range(self.n_ui_layers):
		# 	side_embeddings = torch.sparse.mm(adj, ego_embeddings)
		# 	ego_embeddings = side_embeddings
		# 	all_embeddings += [ego_embeddings]
		# all_embeddings = torch.stack(all_embeddings, dim=1)
		# all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
		# content_embeds = all_embeddings

		# content_embeds_UIPG = self.forward_ui_gcn(self.UIPG)
		# content_embeds_UING = self.forward_ui_gcn(self.UING)

		# User-Item (Behavioral) View
		content_embeds = self.user_item_gcn_layer(adj=adj, n_ui_layers=self.n_ui_layers) # -> torch.Size([26495, 64])
		image_embeds = self.item_item_gcn_layer(self.image_original_adj, image_item_embeds, self.sparse, self.n_layers, self.R) # -> torch.Size([26495, 64])
		text_embeds = self.item_item_gcn_layer(self.text_original_adj, text_item_embeds, self.sparse, self.n_layers, self.R) #-> torch.Size([26495, 64])
		fusion_embeds = self.item_item_gcn_layer(self.fusion_adj, fusion_item_embeds, self.sparse, self.n_layers, self.R) #-> torch.Size([26495, 64])

		popular_behavior_embeds = self.user_popular_niche_gcn_layer(self.UIPG, fusion_item_embeds) # -> torch.Size([26495, 64]) # 大众兴趣嵌入
		niche_behavior_embeds = self.user_popular_niche_gcn_layer(self.UING, fusion_item_embeds) # -> torch.Size([26495, 64])   # 小众兴趣嵌入

		# 'pop_user_pop_item'  'pop_user_niche_item'  'niche_user_pop_item' 'niche_user_niche_item'
		pop_user_pop_item_embeds = self.user_popular_niche_gcn_layer(self.quads['pop_user_pop_item'] , fusion_item_embeds)  # torch.Size([19445, 64])
		pop_user_niche_item_embeds = self.user_popular_niche_gcn_layer(self.quads['pop_user_niche_item'] , fusion_item_embeds)  # torch.Size([19445, 64])
		niche_user_pop_item_embeds = self.user_popular_niche_gcn_layer(self.quads['niche_user_pop_item']  , fusion_item_embeds) # torch.Size([19445, 64])
		niche_user_niche_item_embeds = self.user_popular_niche_gcn_layer(self.quads['niche_user_niche_item'] , fusion_item_embeds) # torch.Size([19445, 64])

		popular_behavior_embeds = (popular_behavior_embeds - niche_user_pop_item_embeds - niche_user_niche_item_embeds) 
		niche_behavior_embeds = (niche_behavior_embeds - pop_user_pop_item_embeds - pop_user_niche_item_embeds) 


		#content_embeds = (content_embeds + content_embeds_UIPG +  content_embeds_UING)/ 3
		# behavior_embeds = self.png_model(content_embeds) 
		# popular_behavior_embeds = behavior_embeds['mass_emb']  # 大众兴趣嵌入
		# niche_behavior_embeds = behavior_embeds['niche_emb']   	

		# content_embeds_UIPG = self.propagate_user_item_behavior(adj=self.UIPG, n_ui_layers=1)
		# content_embeds_UING = self.propagate_user_item_behavior(adj=self.UING, n_ui_layers=1)

		#   Item-Item Modality Specific and Fusion views GCN Layer
		#   Image-view
		# if self.sparse:
		# 	for i in range(self.n_layers):
		# 		image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
		# else:
		# 	for i in range(self.n_layers):
		# 		image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
		# image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
		# image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

		#   Text-view
		# if self.sparse:
		# 	for i in range(self.n_layers):
		# 		text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
		# else:
		# 	for i in range(self.n_layers):
		# 		text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
		# text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
		# text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

		#   Fusion-view
		# if self.sparse:
		# 	for i in range(self.n_layers):
		# 		fusion_item_embeds = torch.sparse.mm(self.fusion_adj, fusion_item_embeds)
		# else:
		# 	for i in range(self.n_layers):
		# 		fusion_item_embeds = torch.mm(self.fusion_adj, fusion_item_embeds)
		# fusion_user_embeds = torch.sparse.mm(self.R, fusion_item_embeds)
		# fusion_embeds = torch.cat([fusion_user_embeds, fusion_item_embeds], dim=0)


		USE_Fusion_UIPG = False
		# 四象限非零元素: [torch.Size([19445, 7050]), torch.Size([19445, 7050]), torch.Size([19445, 7050]), torch.Size([19445, 7050])]
		if USE_Fusion_UIPG:
			# print("UIPG 类型:", self.UIPG.dtype)
			# print("fusion_item_embeds 类型:", fusion_item_embeds.dtype)
			fusion_user_embeds_UIPG = torch.sparse.mm(self.UIPG, fusion_item_embeds) #  torch.Size([19445, 7050]) x torch.Size([7050, 64]) -> torch.Size([19445, 64])
			fusion_embeds_UIPG = torch.cat([fusion_user_embeds_UIPG, fusion_item_embeds], dim=0) # fusion_embeds_uipg.shape: torch.Size([19445, 64]) + torch.Size([7050, 64]) -> torch.Size([26495, 64])
			fusion_user_embeds_UING = torch.sparse.mm(self.UING, fusion_item_embeds)
			fusion_embeds_UING = torch.cat([fusion_user_embeds_UING, fusion_item_embeds], dim=0)
			fusion_embeds = fusion_embeds * fusion_embeds_UIPG * fusion_embeds_UING + fusion_embeds # + is not good

		#   Modality-aware Preference Module
		fusion_att_v, fusion_att_t = self.query_v(fusion_embeds), self.query_t(fusion_embeds)
		fusion_soft_v = self.softmax(fusion_att_v)
		agg_image_embeds = fusion_soft_v * image_embeds

		fusion_soft_t = self.softmax(fusion_att_t)
		agg_text_embeds = fusion_soft_t * text_embeds

		image_prefer = self.gate_image_prefer(content_embeds)
		text_prefer = self.gate_text_prefer(content_embeds)
		fusion_prefer = self.gate_fusion_prefer(content_embeds)

		# popular_image_prefer = self.gate_image_prefer(popular_behavior_embeds)
		# popular_text_prefer = self.gate_text_prefer(popular_behavior_embeds) 
		# niche_image_prefer = self.gate_image_prefer(niche_behavior_embeds)
		# niche_text_prefer = self.gate_text_prefer(niche_behavior_embeds)
		popular_fusion_prefer = self.gate_fusion_prefer(popular_behavior_embeds)
		niche_fusion_prefer = self.gate_fusion_prefer(niche_behavior_embeds)

		# niche_fusion_prefer = self.gate_text_prefer(niche_behavior_embeds)


		# image_prefer, text_prefer, fusion_prefer = self.dropout(image_prefer), self.dropout(text_prefer), self.dropout(fusion_prefer)
		
		# fusion_prefer_UIPG = self.gate_fusion_prefer(content_embeds_UIPG)
		# fusion_prefer_UING = self.gate_fusion_prefer(content_embeds_UING)		
		#fusion_prefer_UIPG, fusion_prefer_UING = self.dropout(fusion_prefer_UIPG), self.dropout(fusion_prefer_UING)

		agg_image_embeds = torch.multiply(image_prefer, agg_image_embeds)
		agg_text_embeds = torch.multiply(text_prefer, agg_text_embeds)
		fusion_embeds = torch.multiply(fusion_prefer, fusion_embeds)

		popular_fusion_embeds = torch.multiply(popular_fusion_prefer, fusion_embeds)
		niche_fusion_embeds = torch.multiply(niche_fusion_prefer, fusion_embeds)
		# popular_image_embeds = torch.multiply(popular_image_prefer, agg_image_embeds)
		# popular_text_embeds = torch.multiply(popular_text_prefer, agg_text_embeds)
		# niche_image_embeds = torch.multiply(niche_image_prefer, agg_image_embeds)
		# niche_text_embeds = torch.multiply(niche_text_prefer, agg_text_embeds)
		# fusion_embeds_UIPG = torch.multiply(fusion_prefer_UIPG, content_embeds_UIPG)
		# fusion_embeds_UING = torch.multiply(fusion_prefer_UING, content_embeds_UING)

		# side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds, fusion_embeds]), dim=0) 
		USE_PNG = True
		if USE_PNG:
			side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds, fusion_embeds, popular_fusion_embeds, niche_fusion_embeds]), dim=0) 
		else:
			side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds, fusion_embeds]), dim=0) 
		# side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds, fusion_embeds, fusion_embeds_UIPG, fusion_embeds_UING]), dim=0) 

		all_embeds = content_embeds + side_embeds

		all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

		if train:
			return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds, popular_fusion_embeds, niche_fusion_embeds

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

		ua_embeddings, ia_embeddings, side_embeds, content_embeds, popular_fusion_embeds, niche_fusion_embeds = self.forward(
			self.norm_adj, train=True)

		u_g_embeddings = ua_embeddings[users]
		pos_i_g_embeddings = ia_embeddings[pos_items]
		neg_i_g_embeddings = ia_embeddings[neg_items]

		batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
																	  neg_i_g_embeddings)

		side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
		content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
		cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
			side_embeds_users[users], content_embeds_user[users], 0.2)


		# 2. 大众兴趣融合表征对比（popular_fusion_embeds）
		popular_users, popular_items = torch.split(
			popular_fusion_embeds, [self.n_users, self.n_items], dim=0
		)
		# 大众兴趣：结构融合表征 vs 内容模态表征（强化通用特征一致性）
		cl_popular_item = self.InfoNCE(popular_items[pos_items], content_embeds_items[pos_items], temperature=0.2)
		cl_popular_user = self.InfoNCE(popular_users[users], content_embeds_user[users], temperature=0.2)
		cl_popular_loss = cl_popular_item + cl_popular_user

		# 3. 小众兴趣融合表征对比（niche_fusion_embeds）
		niche_users, niche_items = torch.split(
			niche_fusion_embeds, [self.n_users, self.n_items], dim=0
		)
		# 小众兴趣：结构融合表征 vs 内容模态表征（强化独特特征一致性，用更低温度）
		cl_niche_item = self.InfoNCE(niche_items[pos_items], content_embeds_items[pos_items], temperature=0.1)
		cl_niche_user = self.InfoNCE(niche_users[users], content_embeds_user[users], temperature=0.1)
		cl_niche_loss = cl_niche_item + cl_niche_user

		# 4. 跨层次兴趣对比（大众 vs 小众，强化差异）
		# 同一物品的大众表征与小众表征应保持差异
		cl_cross_item = self.InfoNCE(popular_items[pos_items], niche_items[pos_items], temperature=0.2)
		# 同一用户的大众表征与小众表征应保持差异
		cl_cross_user = self.InfoNCE(popular_users[users], niche_users[users], temperature=0.2)
		cl_cross_loss = cl_cross_item + cl_cross_user

		# 总损失：基础损失 + 各类对比损失（权重可调整）
		total_loss = (
			batch_mf_loss + batch_emb_loss + batch_reg_loss +
			self.cl_loss * cl_loss +  # 原始对比损失
			0.001 * cl_popular_loss +  # 大众兴趣对比
			0.0005 * cl_niche_loss +  # 小众兴趣对比
			0.0001 * cl_cross_loss  # 跨层次差异对比
		)

		USE_CL = True
		if USE_CL:
			return total_loss
		else:
			return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss 
	
 
	def full_sort_predict(self, interaction):
		user = interaction[0]

		restore_user_e, restore_item_e = self.forward(self.norm_adj)
		u_embeddings = restore_user_e[user]

		# dot with all item embedding to accelerate
		scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
		return scores

class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)  # 以文本为查询
        self.key = nn.Linear(dim, dim)    # 图像作为键
        self.value = nn.Linear(dim, dim)  # 图像作为值
    
    def forward(self, text_feat, image_feat):
        q = self.query(text_feat)  # [N, D]
        k = self.key(image_feat).transpose(0, 1)  # [D, N]
        attn = F.softmax(torch.matmul(q, k) / math.sqrt(q.shape[1]), dim=1)  # [N, N]
        # 用文本查询聚焦图像特征，再与文本融合
        image_focused = torch.matmul(attn, self.value(image_feat))  # [N, D]
        return (text_feat + image_focused) / 2  # 融合结果

