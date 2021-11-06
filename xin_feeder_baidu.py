import os
import sys
import numpy as np
import random
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from layers.graph import Graph

import time

# def data_loader(pra_path, pra_batch_size=128, pra_shuffle=False, pra_drop_last=False, train_val_test='train'):
	# feeder = Feeder(data_path=pra_path, graph_args=graph_args, train_val_test=train_val_test)
#定义一个数据集
class Feeder(torch.utils.data.Dataset):
	""" Feeder for skeleton-based action recognition
	Arguments:
		data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
	"""

	def __init__(self, data_path, graph_args={}, train_val_test='train'):
		'''
		train_val_test: (train, val, test)
		'''
		self.data_path = data_path
		self.load_data()

		total_num = len(self.all_feature)
		# print(total_num)
		# print(np.linspace(0, total_num-1, int(total_num*0.8)))
		# equally choose validation set
		# np.linspace 等差数列,总数为int(total_num*0.8)
		train_id_list = list(np.linspace(0, total_num-1, int(total_num*0.8)).astype(int))
		# print(train_id_list)
		# set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等
		# range()创建一个整数列表
		val_id_list = list(set(list(range(total_num))) - set(train_id_list))
		# print(val_id_list)
		# # last 20% data as validation set
		self.train_val_test = train_val_test
		# print(self.all_feature.shape)
		# print(self.all_adjacency.shape)
		# print(self.all_mean_xy.shape)
		# Python lower() 方法转换字符串中所有大写字符为小写。
		if train_val_test.lower() == 'train':

			self.all_feature = self.all_feature[train_id_list]
			self.all_adjacency = self.all_adjacency[train_id_list]
			self.all_mean_xy = self.all_mean_xy[train_id_list]


		elif train_val_test.lower() == 'val':
			self.all_feature = self.all_feature[val_id_list]
			self.all_adjacency = self.all_adjacency[val_id_list]
			self.all_mean_xy = self.all_mean_xy[val_id_list]

		# 加**表示接受一个dict（字典）,加*表示接受一个tuple（元组）
		# 在这里**的作用是将传入的字典进行unpack，然后将字典中的值作为关键词参数传入函数中
		self.graph = Graph(**graph_args) #num_node = 120,max_hop = 1

	def load_data(self):
		# rb 以二进制格式打开一个文件用于只读。
		with open(self.data_path, 'rb') as reader:
			# pickle.load(file) 反序列化对象。将文件中的数据解析为一个Python对象
			# Training (N, C, T, V)=(5010, 11, 12, 120), (5010, 120, 120), (5010, 2)
			[self.all_feature, self.all_adjacency, self.all_mean_xy]= pickle.load(reader)
			
	# 返回长度
	def __len__(self):
		return len(self.all_feature)
	# 根据 idx 返回一行数据,调用enumerate(pra_data_loader)时的返回值
	def __getitem__(self, idx):
		# C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z,
		# object_length, pbject_width, pbject_height, heading] + [mask]
		now_feature = self.all_feature[idx].copy() # (C, T, V) = (11, 12, 120)
		now_mean_xy = self.all_mean_xy[idx].copy() # (2,) = (x, y) 
		# print(np.random.random())
		# trick:对于随机数大于0.5的,顺时针旋转一个角度
		if self.train_val_test.lower() == 'train' and np.random.random()>0.5:
			angle = 2 * np.pi * np.random.random()
			sin_angle = np.sin(angle)
			cos_angle = np.cos(angle)

			angle_mat = np.array(
				[[cos_angle, -sin_angle],
				[sin_angle, cos_angle]])

			xy = now_feature[3:5, :, :]
			num_xy = np.sum(xy.sum(axis=0).sum(axis=0) != 0) # get the number of valid data

			# angle_mat: (2, 2), xy: (2, 12, 120)
			out_xy = np.einsum('ab,btv->atv', angle_mat, xy)
			# 两个numpy数组的矩阵相乘
			now_mean_xy = np.matmul(angle_mat, now_mean_xy)
			xy[:,:,:num_xy] = out_xy[:,:,:num_xy]

			now_feature[3:5, :, :] = xy

		now_adjacency = self.graph.get_adjacency(self.all_adjacency[idx])
		now_A = self.graph.normalize_adjacency(now_adjacency)
		# print(now_feature.shape,now_A.shape, now_mean_xy.shape)

		return now_feature, now_A, now_mean_xy

