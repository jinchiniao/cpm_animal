# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import glob
import os
import torch
import torch.utils.data
import torchvision.transforms.functional
import cv2
import copy



def read_dataset(path):
	"""
	Read training dataset or validation dataset.
	:param path: The path of dataset.
	:return: The list of filenames.
	"""
	image_list = glob.glob(os.path.join(path, 'images/*.jpg'))

	#image_list = glob.glob(os.path.join(path, 'images/*.png'))

	return image_list


def read_mat(mode, path, image_list):
	"""
	Read joints.mat file.
	joints.mat in lspet is (14, 3, 10000); joints.mat in lsp is (3, 14, 2000)
	:param mode: 'lspet' or 'lsp'
	:param path: The path of joints.mat.
	:param image_list: The array of image filenames.
	:return:
	"""
	mat_arr = sio.loadmat(os.path.join(path, 'joints.mat'))['joints']
	# (x,y,z)
	# LSPET: z = 1 means the key points is not blocked.
	# LSP: z = 0 means the key points is not blocked.
	key_point_list = []
	limits = []

	if mode == 'lspet':
		key_point_list = np.transpose(mat_arr, (2, 0, 1)).tolist()

		# Calculate the limits to find center points
		limits = np.transpose(mat_arr, (2, 1, 0))

	if mode == 'lsp':
		# Guarantee z = 1 means the key points is not blocked
		mat_arr[2] = np.logical_not(mat_arr[2])
		key_point_list = np.transpose(mat_arr, (2, 1, 0)).tolist()
		# Calculate the limits to find center points
		limits = np.transpose(mat_arr, (2, 0, 1))

	center_point_list = []
	scale_list = []
	illegal_list=[]

	for i in range(limits.shape[0]):
		image = cv2.imread(image_list[i])
		h = image.shape[0]
		w = image.shape[1]

		legal_x=[]
		legal_y=[]
		for j in range(len(limits[i][0])):
			if limits[i][2][j] != 0 and limits[i][0][j] >0 and limits[i][0][j] <w and limits[i][1][j] >0 and limits[i][1][j] <h:
				legal_x.append(limits[i][0][j])
				legal_y.append(limits[i][1][j])

		# pass illegal pic
		if len(legal_x)<3 :
			print(image_list[i])
			illegal_list.append(i)
			continue
		# Calculate the center points of each image
		center_x = (min(legal_x) + max(legal_x)) / 2
		center_y = (min(legal_y) + max(legal_y)) / 2

		center_point_list.append([center_x, center_y])

		# Calculate the scale of each image
		scale = (max(legal_y) - min(legal_y) + 4) / 368
		scale_list.append(scale)
	for k,j in enumerate(illegal_list):
		key_point_list.pop(j-k)
		image_list.pop(j-k)
	print(len(key_point_list))
	print(len(image_list))
	return key_point_list, center_point_list, scale_list, image_list


def gaussian_kernel(size_w, size_h, center_x, center_y, sigma):
	grid_y, grid_x = np.mgrid[0:size_h, 0:size_w]
	D2 = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2

	return np.exp(-D2 / 2.0 / sigma / sigma)


class LSP_DATA(torch.utils.data.Dataset):
	def __init__(self, mode, path, stride, transformer=None):
		self.image_list = read_dataset(path)
		self.key_point_list, self.center_point_list, self.scale_list, self.img_list = read_mat(mode, path, self.image_list)
		self.stride = stride
		self.transformer = transformer
		self.sigma = 3.0

	def __getitem__(self, item):
		image_path = self.image_list[item]
		image = np.array(cv2.imread(image_path), dtype=np.float32)

		key_points = self.key_point_list[item]
		center_points = self.center_point_list[item]
		scale = self.scale_list[item]

		# Expand dataset
		if self.transformer:
			image, key_points, center_points = self.transformer(image, key_points, center_points, scale)
		h, w, _ = image.shape
		org_img=copy.deepcopy(image)

		# Generate heatmap
		size_h = int(h / self.stride)
		size_w = int(w / self.stride)
		heatmap = np.zeros((size_h, size_w, len(key_points) + 1), dtype=np.float32)

		# Generate the heatmap of all key points
		for i in range(len(key_points)):
			# pass block point 
			if key_points[i][2]==0:
				kernel=np.zeros((size_h,size_w))
				heatmap[:, :, i + 1] = kernel
				continue
			# Resize image from 368 to 46
			x = int(key_points[i][0]) * 1.0 / self.stride
			y = int(key_points[i][1]) * 1.0 / self.stride

			kernel = gaussian_kernel(size_h=size_h, size_w=size_w, center_x=x, center_y=y, sigma=self.sigma)
			kernel[kernel > 1] = 1
			kernel[kernel < 0.01] = 0
			heatmap[:, :, i + 1] = kernel

		# Generate the heatmap of background
		heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)

		# Generate centermap
		centermap = np.zeros((h, w, 1), dtype=np.float32)
		kernel = gaussian_kernel(size_h=h, size_w=w, center_x=center_points[0], center_y=center_points[1],
		                       sigma=self.sigma)
		kernel[kernel > 1] = 1
		kernel[kernel < 0.01] = 0
		centermap[:, :, 0] = kernel

		image -= image.mean()

		image = torchvision.transforms.functional.to_tensor(image)
		heatmap = torch.from_numpy(np.transpose(heatmap, (2, 0, 1)))
		centermap = torch.from_numpy(np.transpose(centermap, (2, 0, 1)))

		return image.float(), heatmap.float(), centermap.float(), org_img

	def __len__(self):
		return len(self.image_list)
