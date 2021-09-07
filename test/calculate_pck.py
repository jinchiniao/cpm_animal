import os
import sys

sys.path.append(os.getcwd())
print(sys.path)
os.chdir(os.getcwd()+"/test/")
from test_cpm import get_key_points, draw_image
from cpm import cpm, cpm_condense
import copy
from preprocess.gen_data import LSP_DATA
from preprocess.Transformers import Compose, RandomCrop, RandomResized, TestResized
from train_cpm.utils import AverageMeter
import matplotlib.pyplot as plt
from pandas.plotting import table
import pandas as pd
import time
from preprocess.gen_data import gaussian_kernel
import torchvision.transforms.functional as F
import torch
import cv2
import numpy as np




print(os.getcwd())


def compute_pck_pckh(dt_kpts, gt_kpts, T):
    """
    pck指标计算
    :param dt_kpts:算法检测输出的估计结果,shape=[n,h,w]=[图片数数，3，关键点个数]
    :param gt_kpts: groundtruth人工标记结果,shape=[n,h,w]
    :return: 相关指标
    """
    dt = np.array(dt_kpts)
    gt = np.array(gt_kpts)
    assert(dt.shape[0] == gt.shape[0])
    ranges = np.array(T)
    kpts_num = gt.shape[2]
    ped_num = gt.shape[0]
    # compute dist
    legal_x = []
    legal_y = []
    scale = np.ones(dt.shape[0])
    dist = np.zeros((dt.shape[0], dt.shape[2]))
    legal_total = np.ones(dt.shape[0])
    for i in range(dt.shape[0]):
        for j in range(dt.shape[2]):
            if gt[i][2][j] != 0:
                legal_x.append(gt[i][0][j])
                legal_y.append(gt[i][1][j])
            dist[i][j] = np.sqrt(
                np.power(gt[i][0][j]-dt[i][0][j], 2)+np.power(gt[i][1][j]-dt[i][1][j], 2))
        # calculate as body length
        scale[i] = max(legal_x) - min(legal_x)
    dist /= scale[:, None]
    legal_total = np.sum(gt[:, 2, :], axis=0)
    # compute pck
    pck = np.zeros([ranges.shape[0], gt.shape[2]+1])
    pck_recall = np.zeros([ranges.shape[0], gt.shape[2]+1])
    pck_posi_right = np.zeros([ranges.shape[0], gt.shape[2]+1])
    pck_exsit = np.zeros([ranges.shape[0], gt.shape[2]+1])
    for idh, trh in enumerate(list(ranges)):
        for kpt_idx in range(kpts_num):
            pck[idh, kpt_idx] = 100*np.mean((np.logical_or(np.logical_and(np.logical_and(dist[:, kpt_idx] <= trh,
                                            gt[:, 2, kpt_idx] > 0),dt[:, 2, kpt_idx] > 0), np.logical_and(dt[:, 2, kpt_idx] == 0, gt[:, 2, kpt_idx] == 0))))
            pck_recall[idh, kpt_idx] = 100*np.mean(np.logical_or(np.logical_and(np.logical_and(
                dist[:, kpt_idx] <= trh, gt[:, 2, kpt_idx] > 0), dt[:, 2, kpt_idx] == gt[:, 2, kpt_idx]), gt[:, 2, kpt_idx] == 0))
            pck_posi_right[idh, kpt_idx] = 100*np.mean((np.logical_or(np.logical_and(
                dist[:, kpt_idx] <= trh, gt[:, 2, kpt_idx] > 0), gt[:, 2, kpt_idx] == 0)))
            pck_exsit[idh, kpt_idx] =100*np.mean((np.logical_or(np.logical_and(
                dt[:, 2, kpt_idx] > 0, gt[:, 2, kpt_idx] > 0), np.logical_and(
                dt[:, 2, kpt_idx] ==0, gt[:, 2, kpt_idx] == 0))))
        # compute average pck
        pck[idh, -1] = 100*np.mean((np.logical_or(np.logical_and(np.logical_and(
            dist[:, ] <= trh, gt[:, 2] > 0),dt[:, 2] > 0), np.logical_and(dt[:, 2] == 0, gt[:, 2] == 0))))
        pck_recall[idh, -1] = 100*np.mean(np.logical_or(np.logical_and(np.logical_and(
            dist[:, ] <= trh, gt[:, 2] > 0), dt[:, 2] == gt[:, 2]), gt[:, 2] == 0))
        pck_posi_right[idh, -1] = 100*np.mean(np.logical_or(
            (np.logical_and(dist[:, ] <= trh, gt[:, 2] > 0)), gt[:, 2] == 0))
        pck_exsit[idh, -1]=100*np.mean((np.logical_or(np.logical_and(
                dt[:, 2] > 0, gt[:, 2] > 0), np.logical_and(
                dt[:, 2] ==0, gt[:, 2] == 0))))
    return pck, pck_recall, pck_posi_right, pck_exsit


def output_accuracy(dataset_path, model, device=None, save_name='default', batch_size=1):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = LSP_DATA('lsp', dataset_path, 8, Compose([TestResized(368)]))
    test_loader = torch.utils.data.dataloader.DataLoader(
        data, batch_size=batch_size)
    #model = model.to(device)
    T = np.arange(0, 0.31, 0.01)
    key_points_all_test = list()
    key_points_all_true = list()
    time_cost_total = 0
    for j, data in enumerate(test_loader):
        with torch.no_grad():
            inputs, heatmap, centermap, image = data
            image = np.squeeze(image)

            height, width, _ = image.shape
            image = np.asarray(image, dtype=np.float32)
            image = cv2.resize(image, (368, 368), interpolation=cv2.INTER_CUBIC)
            org_img = copy.deepcopy(image)

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

            image = torch.unsqueeze(image, 0).to(device)
            centermap = torch.unsqueeze(centermap, 0).to(device)

    #        model.eval()
            start = time.time()
            input_var = torch.autograd.Variable(image)
            center_var = torch.autograd.Variable(centermap)

            heat1, heat2, heat3, heat4, heat5, heat6 = model(input_var, center_var)
        key_points = get_key_points(heat6, height=height, width=width,threshold=0.05)
        key_points_true = get_key_points(heatmap, height=height, width=width)
        time_cost = time.time()-start
        print("time", time_cost, "s")
        time_cost_total += time_cost
        key_points_all_test.append(np.array(key_points))
        key_points_all_true.append(np.array(key_points_true))
        image = draw_image(org_img/255, key_points)
    print("average time:", time_cost_total/len(test_loader))

    kpate = np.transpose(np.array(key_points_all_test), (0, 2, 1))
    kpatr = np.transpose(np.array(key_points_all_true), (0, 2, 1))
    pck, pck_recall, pck_posi_right,pck_exist= compute_pck_pckh(kpate, kpatr, T)
    df = pd.DataFrame(pck, index=["pck@"+str(i) for i in T], columns=['left ear', 'right ear', 'nose', 'right shoulder', 'right front paw', 'left shoulder', 'left front paw',
                      'right hip', 'right knee', 'right back paw', 'left hip', 'left knee', 'left back paw', 'root of tail', 'center of 3 and 14', 'total'], dtype=float)
    df_recall = pd.DataFrame(pck_recall, index=["pck@"+str(i) for i in T], columns=['left ear', 'right ear', 'nose', 'right shoulder', 'right front paw', 'left shoulder',
                             'left front paw', 'right hip', 'right knee', 'right back paw', 'left hip', 'left knee', 'left back paw', 'root of tail', 'center of 3 and 14', 'total'], dtype=float)
    df_posi_right = pd.DataFrame(pck_posi_right, index=["pck@"+str(i) for i in T], columns=['left ear', 'right ear', 'nose', 'right shoulder', 'right front paw', 'left shoulder',
                                 'left front paw', 'right hip', 'right knee', 'right back paw', 'left hip', 'left knee', 'left back paw', 'root of tail', 'center of 3 and 14', 'total'], dtype=float)
    df_exsit=pd.DataFrame(pck_exist, index=["pck@"+str(i) for i in T], columns=['left ear', 'right ear', 'nose', 'right shoulder', 'right front paw', 'left shoulder',
                                 'left front paw', 'right hip', 'right knee', 'right back paw', 'left hip', 'left knee', 'left back paw', 'root of tail', 'center of 3 and 14', 'total'], dtype=float)
    save_path = os.path.join(os.getcwd(), 'result_'+save_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    df.to_excel(save_path+'/pck_of_'+save_name+'.xls')
    df_recall.to_excel(save_path+'/pck_of_'+save_name+'_recall.xls')
    df_posi_right.to_excel(save_path+'/pck_of_'+save_name+'_posi_right.xls')
    df_exsit.to_excel(save_path+'/pck_of_'+save_name+'_exsit.xls')

if __name__ == "__main__":

    dataset_path = '../atrw_split/trainset'

    #model_name_list = ['atrw','atrw_transfer','atrw_dpc','atrw_dpc_ts']
    model_name_list = ['atrw_dpc_ts_quant']

    for i in model_name_list:
        model = torch.load('../model/best_cpm_'+i+'.pth')
        #model.pool_center = torch.nn.AvgPool2d(kernel_size=9, stride=8, padding=1)
        print(i+" result:")
        output_accuracy(dataset_path=dataset_path, model=model,device=torch.device('cpu'),
                        save_name='cpm_'+i, batch_size=1)
