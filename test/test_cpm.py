import os
import sys
sys.path.append(os.getcwd())
if 'test' not in os.getcwd():
    os.chdir(os.getcwd()+"/test/")

import numpy as np
import cv2
import torch
import torchvision.transforms.functional as F
from preprocess.gen_data import gaussian_kernel
import time
import torch.nn as nn


print(os.getcwd())


def get_key_points(heatmap6, height, width, threshold=0.15):
    """
    Get all key points from heatmap6.
    :param heatmap6: The heatmap6 of CPM cpm.
    :param height: The height of original image.
    :param width: The width of original image.
    :return: All key points of the person in the original image.
    """
    # Get final heatmap
    heatmap = np.asarray(heatmap6.cpu().data)[0]

    key_points = []
    # Get k key points from heatmap6
    for i in heatmap[1:]:
        # Get the coordinate of key point in the heatmap (46, 46)
        y, x = np.unravel_index(np.argmax(i), i.shape)

        # Calculate the scale to fit original image
        scale_x = width / i.shape[1]
        scale_y = height / i.shape[0]
        x = int(x * scale_x)
        y = int(y * scale_y)
        # determine whether there is keypoint in the picture
        if i.max() < threshold:
            key_points.append([x, y, 0])
            continue
        key_points.append([x, y, 1])

    return key_points


def draw_image(image, key_points):
    """
    Draw limbs in the image.
    :param image: The test image.
    :param key_points: The key points of the person in the test image.
    :return: The painted image.
    """
    '''ALl limbs of person:
	head top->neck
	neck->left shoulder
	left shoulder->left elbow
	left elbow->left wrist
	neck->right shoulder
	right shoulder->right elbow
	right elbow->right wrist
	neck->left hip
	left hip->left knee
	left knee->left ankle
	neck->right hip
	right hip->right knee
	right knee->right ankle
	'''
    # human
    # limbs = [[13, 12], [12, 9], [9, 10], [10, 11], [12, 8], [8, 7], [7, 6], [12, 3], [3, 4], [4, 5], [12, 2], [2, 1],
    #        [1, 0]]
    # tigers
    limbs = [[12, 11], [10, 11], [13, 10], [8, 9], [7, 8], [13, 7], [
        14, 13], [5, 6], [4, 3], [3, 14], [14, 5], [14, 2], [2, 1], [2, 0]]
    colors = [[255, 182, 193], [220, 20, 60], [255, 20, 147], [75, 0, 130], [0, 0, 255], [	135, 206, 250], [127, 255, 170], [
        255, 250, 205], [255, 165, 0], [255, 0, 0], [105, 105, 105], [0, 128, 0], [173, 255, 47], [64, 224, 208]]
    # horse
    #limbs = [[1,2],[2,13],[13,10],[10,9],[9,11],[9,12],[11,3],[3,4],[4,5],[11,6],[6,7],[7,8],[13,17],[17,22],[17,18],[18,14],[14,15],[15,16],[18,19],[19,20],[20,21]]
    # draw key points
    for key_point in key_points:
        if key_point[2] == 0:
            continue
        x = key_point[0]
        y = key_point[1]
        cv2.circle(image, (x, y), radius=1, thickness=-1, color=(0, 0, 255))

    # draw limbs
    for k, limb in enumerate(limbs):
        if key_points[limb[0]][2] == 0 or key_points[limb[1]][2] == 0:
            continue
        start = key_points[limb[0]][:-1]
        end = key_points[limb[1]][:-1]
        color = colors[k]  # BGR
        cv2.line(image, tuple(start), tuple(end),
                 color, thickness=5, lineType=4)

    return image


if __name__ == "__main__":
    model = torch.load('../model_test/model/best_cpm_atrw_dpc_ts.pth').cuda()
    # to solve the incompatibility problem
    model.pool_center = nn.AvgPool2d(kernel_size=9, stride=8, padding=1)

    image_path = '../test_data/test21.jpg'
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    image = np.asarray(image, dtype=np.float32)
    image = cv2.resize(image, (368, 368), interpolation=cv2.INTER_CUBIC)

    # Normalize
    image -= image.mean()
    image = F.to_tensor(image)

    # Generate center map
    centermap = np.zeros((368, 368, 1), dtype=np.float32)
    kernel = gaussian_kernel(size_h=368, size_w=368,
                             center_x=184, center_y=184, sigma=3)
    kernel[kernel > 1] = 1
    kernel[kernel < 0.01] = 0
    centermap[:, :, 0] = kernel
    centermap = torch.from_numpy(np.transpose(centermap, (2, 0, 1)))

    image = torch.unsqueeze(image, 0).cuda()
    centermap = torch.unsqueeze(centermap, 0).cuda()

    model.eval()
    start = time.time()
    input_var = torch.autograd.Variable(image)
    center_var = torch.autograd.Variable(centermap)

    heat1, heat2, heat3, heat4, heat5, heat6 = model(input_var, center_var)
    key_points = get_key_points(heat6, height=height, width=width,threshold=0.00)
    print("time", time.time()-start, "s")
    image = draw_image(cv2.imread(image_path), key_points)

    cv2.imshow('test image', image)
    cv2.waitKey(0)

    cv2.imwrite(image_path.rsplit('.', 1)[0] + '_ans.jpg', image)
