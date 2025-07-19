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

from pytorch_wavelets import DWT1DForward, DWT1DInverse 
from models.MMWI import MultiModalWaveletInterestAttention
from models.MMWA import MMWaveletAlignModule

class WaveRec(GeneralRecommender):
	'''

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


		# self.visual_modal_expert = VisualModalExpertNetwork(in_features=4096)
		# self.text_modal_expert = TextModalExpertNetwork(in_features=384)
		# self.multi_modal_attenion = MultiModalAttention(embed_dim=64, num_heads=self.attention_heads)
		
		#多模态多尺度对齐
		self.multimodal_multiscale_wavelet_align = MMWaveletAlignModule(wavelet='db4', level=3)
		# 多模态小波兴趣感知
		# self.mm_wavelet_interest_aware = MultiModalWaveletInterestAttention(embed_dim=self.embedding_dim)
		self.mm_wavelet_interest_aware = MultiModalWaveletInterestAttention(embed_dim=self.embedding_dim, wavelet_name='db1',decomp_level=1)

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
			# print("image_feats.shape:", image_feats.shape) # [7050, 64]
			# print("text_feats.shape:",text_feats.shape )
		if self.t_feat is not None:
			text_feats = self.text_trs(self.text_embedding.weight)

		# if self.v_feat is not None:
		# 	#image_feats = denoise_norm(self.image_embedding.weight, weight=0.6)
		# 	image_feats = self.visual_modal_expert(self.image_embedding.weight)
		# if self.t_feat is not None:
		# 	#text_feats = denoise_norm(self.text_embedding.weight, weight=0.4)
		# 	text_feats = self.text_modal_expert(self.text_embedding.weight)
		
		# (Multimodal Multi-Sacle Wavelet align, MMWA)
		if self.MMWA:
			image_embedding, text_embedding, fusion_embedding = self.multimodal_multiscale_wavelet_align(image_feats, text_feats)
		else:
			image_embedding = image_feats
			text_embedding = text_feats

		image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_embedding))
		text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_embedding))
		fusion_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_f(fusion_embedding))
		# interest_fusion, low_freq_interest, high_freq_interest = self.mm_wavelet_interest_aware(image_item_embeds, text_item_embeds, fusion_item_embeds)

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
		# multimodal interest aware(mmia)
		mmia, low_freq_interest, high_freq_interest = self.mm_wavelet_interest_aware(image_item_embeds, text_item_embeds, fusion_item_embeds)

		fusion_embeds = torch.cat([fusion_user_embeds, fusion_item_embeds + mmia], dim=0)

		########################################################Modality-aware Preference Module########################################################################
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
		'''
		content_embeds.shape: torch.Size([26495, 64])
		image_prefer.shape: torch.Size([26495, 64]) text_prefer.shape: torch.Size([26495, 64]) fusion_prefer.shape: torch.Size([26495, 64])
		agg_image_embeds.shape: torch.Size([26495, 64]) agg_text_embeds.shape: torch.Size([26495, 64]) fusion_embeds.shape: torch.Size([26495, 64])
		'''
		agg_image_embeds = torch.multiply(image_prefer, agg_image_embeds) # 图像模态偏好感知
		agg_text_embeds = torch.multiply(text_prefer, agg_text_embeds) # 文本模态偏好感知
		fusion_embeds = torch.multiply(fusion_prefer, fusion_embeds) # 兴趣偏好感知
		# 公共兴趣和个性化兴趣偏好感知

		side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds, fusion_embeds]), dim=0) 
		#side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds]), dim=0) 
		# side_embeds = torch.mean(torch.stack([fusion_embeds]), dim=0) 

		all_embeds = content_embeds + side_embeds
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
		# if self.MMWI:
		# 	cl_loss3 =self.InfoNCE(low_freq_interest, high_freq_interest, 0.2)
			# low_freq_embeds_users, low_freq_embeds_items = torch.split(low_freq_interest, [self.n_users, self.n_items], dim=0)
			# high_freq_embeds_user, high_freq_embeds_items = torch.split(high_freq_interest, [self.n_users, self.n_items], dim=0)
			# cl_loss3 = self.InfoNCE(low_freq_embeds_items[pos_items], high_freq_embeds_items[pos_items], 0.2) + self.InfoNCE(low_freq_embeds_users[users], high_freq_embeds_user[users], 0.2)
		# else:
		# 	cl_loss3 = 0.0

		# return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss1 + self.cl_loss * 0.1 * cl_loss2 + self.cl_loss * 0.1 * cl_loss3
		#return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss1 * cl_loss1 + self.cl_loss2  * cl_loss2 + self.cl_loss3 * cl_loss3
		return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss1 * cl_loss1  

	def full_sort_predict(self, interaction):
		user = interaction[0]

		restore_user_e, restore_item_e = self.forward(self.norm_adj)
		u_embeddings = restore_user_e[user]

		# dot with all item embedding to accelerate
		scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
		return scores


