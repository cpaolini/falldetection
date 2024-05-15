# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 18:24:11 2022

@author: nrshr
"""

import cv2
import torch
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import scipy.io as scio
import os
from PIL import Image
from torch.autograd import Variable
import model as model
# import model_depthreg_noncomp as model
import anchor as anchor
# import anchor_depthreg_noncomp  as anchor
from tqdm import tqdm
import random_erasing
import logging
import time
import datetime
import random
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn as nn
from matplotlib.patches import Circle, ConnectionPatch
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
np.set_printoptions(suppress=True)
fx = 288
fy = -288
u0 = 160
v0 = 120

#DataHyperParms
TrainImgFrames = 17991
TestImgFrames = 4863
keypointsNumber = 15
cropWidth = 288
cropHeight = 288
batch_size = 256
learning_rate = 0.00035
Weight_Decay = 1e-4
# Weight_Decay = 0.1
nepoch = 50
RegLossFactor = 5#3
spatialFactor = 1#0.5
RandCropShift = 15#5
RandshiftDepth = 5#1
RandRotate = 45#180
RandScale = (0.4, 0.8)#(1.0, 0.5)
# xy_thres = 120
depth_thres = 0.5#0.4
imgWidth = 320
imgHeight = 240
depthFactor = 50
Use_Normal = False
# randomseed = 12345
# random.seed(randomseed)
# np.random.seed(randomseed)
# torch.manual_seed(randomseed)

save_dir = './result/ITOP_up_HP_anchor_model/'

try:
    os.makedirs(save_dir)
except OSError:
    pass

trainingImageDir = '/mnt/beegfs/home/ramesh/Datasets/ITOP_side/PreProcessed/train/' # mat images
testingImageDir = '/mnt/beegfs/home/ramesh/Datasets/ITOP_side/PreProcessed/test/' # mat images
keypointsfileTest = '/mnt/beegfs/home/ramesh/A2J/data/itop_side/itop_side_keypoints3D_test.mat' # in world coordinates
keypointsfileTrain = '/mnt/beegfs/home/ramesh/A2J/data/itop_side/itop_side_keypoints3D_train.mat' # in world coordinates

bndbox_test = scio.loadmat('../data/itop_side/itop_side_bndbox_test_npy.mat' )['FRbndbox_test']
bndbox_train = scio.loadmat('../data/itop_side/itop_side_bndbox_train_npy.mat' )['FRbndbox']
Img_mean = np.load('../data/itop_side/itop_side_mean.npy')[3] #3.44405131082671
Img_std = np.load('../data/itop_side/itop_side_std.npy')[3] #0.5403981602753222

model_dir = '/mnt/beegfs/home/ramesh/A2J/model/ITOP_side.pth'
result_file = 'result_test.txt'

# def pixel2world(x):
    # x[:, :, 0] = (x[:, :, 0] - u0) * x[:, :, 2] * 0.0035
    # x[:, :, 1] = (v0 - x[:, :, 1]) * x[:, :, 2] * 0.0035
    # return x
def pixel2world(x,y,z):
    worldX = (x - 160.0)*z*0.0035
    worldY = (120.0 - y)*z*0.0035
    return worldX,worldY

def world2pixel(x,y,z):
    pixelX = 160.0 + x / (0.0035 * z)
    pixelY = 120.0 - y / (0.0035 * z)
    return pixelX,pixelY

# def world2pixel(x):
#     x[:, :, 0] = u0 + x[:, :, 0] / (x[:, :, 2] * 0.0035)
#     x[:, :, 1] = v0 - x[:, :, 1] / (x[:, :, 2] * 0.0035)
#     return x

joint_id_to_name = {
    0: 'Head',
    1: 'Neck',
    2: 'RShoulder',
    3: 'LShoulder',
    4: 'RElbow',
    5: 'LElbow',
    6: 'RHand',
    7: 'LHand',
    8: 'Torso',
    9: 'RHip',
    10: 'LHip',
    11: 'RKnee',
    12: 'LKnee',
    13: 'RFoot',
    14: 'LFoot',
}

# use GPU if available   
if (torch.cuda.device_count() > 0):
    print('You have',torch.cuda.device_count(),'CUDA devices available')
    for i in range(torch.cuda.device_count()):
        print(' Device',str(i),': ',torch.cuda.get_device_name(i))
    print('Selecting all available devices')
    device = torch.device('cuda')
else:
    print('No CUDA devices available..selecting CPU')
    device = torch.device('cpu')
#
def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('torso'), keypoints.index('right_hip')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('torso'), keypoints.index('left_hip')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('torso'), keypoints.index('neck')],
        [keypoints.index('neck'), keypoints.index('right_shoulder')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('neck'), keypoints.index('left_shoulder')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('neck'), keypoints.index('head')]
    ]
    return kp_lines

jointColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
               [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
               [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [85, 255, 85]]

def get_keypoints():
    """Get the itop keypoints"""
    keypoints = [
        'head',
        'neck',
        'right_shoulder',
        'left_shoulder',
        'right_elbow',
        'left_elbow',
        'right_wrist',
        'left_wrist',
        'torso',
        'right_hip',
        'left_hip',
        'right_knee',
        'left_knee',
        'right_ankle',
        'left_ankle']
    return keypoints

def draw_humans_visibility(img, humans, limbs, jointColors, visibilities=None):
    visibilities = visibilities or None
    for i, human in enumerate(humans):
        human_vis = np.array(human)
        for k, limb in enumerate(limbs):
            if visibilities is not None and visibilities[i][limb[0]] < 0.5:
                color = [0, 0, 0]
            else:
                color = [0, 0, 255]
            center1 = human_vis[limb[0], :2].astype(int)
            img = cv2.circle(img, tuple(center1), 3, color, thickness=2, lineType=8, shift=0)

            if visibilities is not None and visibilities[i][limb[1]] < 0.5:
                color = [0, 0, 0]
            else:
                color = [0, 0, 255]
            center2 = human_vis[limb[1], :2].astype(int)
            img = cv2.line(img, tuple(center1), tuple(center2), jointColors[k], 2)
            img = cv2.circle(img, tuple(center2), 3, color, thickness=2, lineType=8, shift=0)

    return img

def evaluation2D_perJoint(source, target, dist_th_2d):
    assert np.shape(source) == np.shape(target), "source has different shape with target"
    count = 0
    acc_vec = []

    for j in range(keypointsNumber):
        for i in range(len(source)):
            if np.square(source[i, j, 0] - target[i, j, 0]) + np.square(
                    source[i, j, 1] - target[i, j, 1]) < np.square(dist_th_2d):
                count = count + 1

        accuracy = count / (len(source))
        print('joint_', j, joint_id_to_name[j], ', accuracy: ', accuracy)
        acc_vec.append(accuracy)
        accuracy = 0
        count = 0

def evaluation2D(source, target, dist_th_2d):
    assert np.shape(source) == np.shape(target), "source has different shape with target"
    count = 0
    for i in range(len(source)):
        for j in range(keypointsNumber):
            if np.square(source[i, j, 0] - target[i, j, 0]) + np.square(
                    source[i, j, 1] - target[i, j, 1]) < np.square(dist_th_2d):
                count = count + 1
    accuracy = count / (len(source) * keypointsNumber)
    return accuracy

#loading GT keypoints and center points
keypointsWorldtest = scio.loadmat(keypointsfileTest)['keypoints3D'].astype(np.float32) # this is in world coordinates
keypointsPixeltest = np.ones((len(keypointsWorldtest),15,2),dtype='float32')
# keypointsPixeltest = world2pixel(keypointsWorldtest.copy()) # this is in pixel coordinates, 320 x 240
keypointsPixeltest_tuple = world2pixel(keypointsWorldtest[:,:,0],keypointsWorldtest[:,:,1],keypointsWorldtest[:,:,2])
assert np.shape(keypointsPixeltest_tuple[0])==(len(keypointsWorldtest),15), "data transform error!"
keypointsPixeltest[:,:,0] = keypointsPixeltest_tuple[0]
keypointsPixeltest[:,:,1] = keypointsPixeltest_tuple[1]

keypointsWorldtrain = scio.loadmat(keypointsfileTrain)['keypoints3D'].astype(np.float32)
keypointsPixeltrain = np.ones((len(keypointsWorldtrain),15,2),dtype='float32')
# keypointsPixeltrain = world2pixel(keypointsWorldtrain.copy())
keypointsPixeltrain_tuple = world2pixel(keypointsWorldtrain[:,:,0],keypointsWorldtrain[:,:,1],keypointsWorldtrain[:,:,2])
assert np.shape(keypointsPixeltrain_tuple[0])==(len(keypointsWorldtrain),15), "data transform error!"
keypointsPixeltrain[:,:,0] = keypointsPixeltrain_tuple[0]
keypointsPixeltrain[:,:,1] = keypointsPixeltrain_tuple[1]


def transform(img, label, matrix):
    '''
    img: [H, W] label, [N,2]
    '''
    img_out = cv2.warpAffine(img,matrix,(cropWidth,cropHeight))
    label_out = np.ones((keypointsNumber, 3))
    label_out[:,:2] = label[:,:2].copy()
    label_out = np.matmul(matrix, label_out.transpose())
    label_out = label_out.transpose()

    return img_out, label_out

def dataPreprocess(index, img, keypointsWorld, keypointsPixel, bndbox, augment=True):
    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32') 
    labelOutputs = np.ones((keypointsNumber, 3), dtype = 'float32') 

    if augment:
        RandomOffset_1 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_2 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_3 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_4 = np.random.randint(-1*RandCropShift,RandCropShift)
        # RandomOffsetDepth = np.random.normal(0, RandshiftDepth, cropHeight*cropWidth).reshape(cropHeight,cropWidth) 
        # RandomOffsetDepth[np.where(RandomOffsetDepth < RandshiftDepth)] = 0
        RandomRotate = np.random.randint(-1*RandRotate,RandRotate)
        RandomScale = np.random.rand()*RandScale[0]+RandScale[1]
        matrix = cv2.getRotationMatrix2D((cropWidth/2,cropHeight/2),RandomRotate,RandomScale)
    else:
        RandomOffset_1, RandomOffset_2, RandomOffset_3, RandomOffset_4 = 0, 0, 0, 0
        RandomRotate = 0
        RandomScale = 1
        # RandomOffsetDepth = 0
        matrix = cv2.getRotationMatrix2D((cropWidth/2,cropHeight/2),RandomRotate,RandomScale)

    #imCrop = img[:,:].copy()
    # imCrop = img.copy()
    new_Xmin = max(bndbox[index][0] + RandomOffset_1, 0)
    new_Ymin = max(bndbox[index][1] + RandomOffset_2, 0)
    new_Xmax = min(bndbox[index][2] + RandomOffset_3, img.shape[1] - 1)
    new_Ymax = min(bndbox[index][3] + RandomOffset_4, img.shape[0] - 1)

    imCrop = img.copy()[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)] 

    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)
    
    imgResize = np.asarray(imgResize,dtype = 'float32')  # H*W*C
   
    imgResize = (imgResize - Img_mean) / Img_std

    ## label
    label_xy = np.ones((keypointsNumber, 2), dtype='float32')
    label_xy[:, 0] = (keypointsPixel[index, :, 0] - new_Xmin) * cropWidth / (new_Xmax - new_Xmin)  # x
    label_xy[:, 1] = (keypointsPixel[index, :, 1] - new_Ymin) * cropHeight / (new_Ymax - new_Ymin)  # y

    ## commented temporarily for visulization purpose##
    if augment:
        imgResize, label_xy = transform(imgResize, label_xy, matrix)  ## rotation, scale

    imageOutputs[:,:,0] = imgResize
    
    labelOutputs[:,1] = label_xy[:,0]
    labelOutputs[:,0] = label_xy[:,1]
    # labelOutputs[:,2] = (keypointsUVD[index,:,2])#*RandomScale   # Z  
    labelOutputs[:,2] = (keypointsWorld.copy()[index,:,2])*depthFactor   # Z  
    # if index == 0:
        # print(f"new_Xmin = {new_Xmin}")
        # print(f"new_Ymin = {new_Ymin}")
        # print(f"new_Xmax = {new_Xmax}")
        # print(f"new_Ymax = {new_Ymax}")

        # print(f"keypointsUVD[0,:,:] = {keypointsUVD[0,:,:]}") # this is in the 320x240

        # print(f"labelOutputs = {labelOutputs}") # this is in the 288x288
    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)
    labelOutputs = np.asarray(labelOutputs)

    data, label = torch.from_numpy(imageNCHWOut), torch.from_numpy(labelOutputs)
 
    return data, label

###################### Pytorch dataloader #################
class my_dataloader(torch.utils.data.Dataset):
    def __init__(self, ImgDir, keypointsWorld, keypointsPixel, num, bndbox, augment=True):
    
        self.ImgDir = ImgDir
        self.keypointsWorld = keypointsWorld
        self.keypointsPixel = keypointsPixel
        self.num = num
        self.bndbox = bndbox
        self.augment = augment
        # self.randomErase = random_erasing.RandomErasing(probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0])
        self.randomErase = random_erasing.RandomErasing(probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0], scale=0.2)

    def __getitem__(self, index):
    
        data4D = scio.loadmat(self.ImgDir + str(index+1) + '.mat')['DepthNormal']
        depth = data4D[:,:,3]

        data, label = dataPreprocess(index, depth, self.keypointsWorld, self.keypointsPixel, self.bndbox, self.augment)
        if self.augment:
            data = self.randomErase(data)
        return data, label

    def __len__(self):
        return self.num
    
train_image_datasets = my_dataloader(trainingImageDir, keypointsWorldtrain, keypointsPixeltrain, TrainImgFrames, bndbox_train, augment=True)
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size = batch_size,
shuffle = True, num_workers = 8)

test_image_datasets = my_dataloader(testingImageDir, keypointsWorldtest, keypointsPixeltest, TestImgFrames, bndbox_test, augment=False)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size = batch_size,
shuffle = False, num_workers = 8)

writer = SummaryWriter('./result/ITOP_up_Hyperparameters_anchor/runs/A2J_ITOP_experiment')

def train():
    net = model.A2J_model(num_classes = keypointsNumber)
    # net = model.Cls_Reg_Dual_Path_Net(num_classes = keypointsNumber, useNormal=Use_Normal)
    # net = net.cuda()
    net = torch.nn.DataParallel(net).to(device)
    
    post_precess = anchor.post_process(shape=[cropHeight//16,cropWidth//16],stride=16,P_h=None, P_w=None)
    criterion = anchor.A2J_loss(shape=[cropHeight//16,cropWidth//16],thres = [16.0,32.0],stride=16,\
        spatialFactor=spatialFactor,img_shape=[cropHeight, cropWidth],P_h=None, P_w=None)
    # criterion = anchor.FocalLoss(shape=[cropHeight//16,cropWidth//16],thres = [16.0,32.0],stride=16,\
    #     spatialFactor=spatialFactor,img_shape=[cropHeight, cropWidth],P_h=None, P_w=None)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=Weight_Decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                    filename=os.path.join(save_dir, 'train.log'), force=True, level=logging.INFO)
    logging.info('======================================================')
    
    for epoch in range(nepoch):
        net = net.train()
        scheduler.step()
        train_loss_add = 0.0
        val_loss_add = 0.0
        Cls_loss_add = 0.0
        Cls_loss_val_add = 0.0
        Reg_loss_add = 0.0
        Reg_loss_val_add = 0.0
        timer = time.time()
    
        # Training loop
        for i, (img, label) in enumerate(train_dataloaders):
            
            torch.cuda.synchronize() 

            # img, label = img.cuda(), label.cuda()
            img, label = img.to(device), label.to(device)
            heads  = net(img)     
            optimizer.zero_grad()  

#           visualize input images to the network with GT points -- block taken form MP3DHP
            depth_max = 5#6
            depth_mean = 3#Img_mean
            depth_std = 2#Img_std
            img, label = img.cpu(), label.cpu()
            img_ = img.numpy().copy()
            img_ = img_[0, 0, :, :]
            label = label.numpy()
            label_new = np.copy(label)
            label_new[:, :, 0] = label[:, :, 1]
            label_new[:, :, 1] = label[:, :, 0]

            human_2d = [label_new[0]]

            single_img = np.copy(img_)
            single_img *= depth_std
            single_img += depth_mean
            single_img[single_img >= depth_max] = depth_max
            single_img /= depth_max
            single_img *= 255
            single_img = draw_humans_visibility(single_img,
                                                human_2d,
                                                kp_connections(get_keypoints()),
                                                jointColors)
            plt.imshow(single_img)
            single_img = single_img.astype(np.uint8)
            cv2.imwrite(save_dir + '/images/input/input_' + str(i) + '.png', single_img)
#
            # label = torch.from_numpy(label).cuda()
            label = torch.from_numpy(label).to(device)
            Cls_loss, Reg_loss = criterion(heads, label)
    
            loss = 1*Cls_loss + Reg_loss*RegLossFactor
            loss.backward()
            optimizer.step()
    
            torch.cuda.synchronize()
            
            train_loss_add = train_loss_add + (loss.item())*len(img)
            Cls_loss_add = Cls_loss_add + (Cls_loss.item())*len(img)
            Reg_loss_add = Reg_loss_add + (Reg_loss.item())*len(img)
            # printing loss info
            if i%10 == 0:
                print('epoch: ',epoch, ' step: ', i, 'Cls_loss ',Cls_loss.item(), 'Reg_loss ',Reg_loss.item(), ' total loss ',loss.item())
    
        # scheduler.step()
    
        # time taken
        torch.cuda.synchronize()
        timer = time.time() - timer
        timer = timer / TrainImgFrames
        print('==> time to learn 1 sample = %f (ms)' %(timer*1000))
    
        train_loss_add = train_loss_add / TrainImgFrames
        Cls_loss_add = Cls_loss_add / TrainImgFrames
        Reg_loss_add = Reg_loss_add / TrainImgFrames
        print('mean train_loss_add of 1 sample: %f, #train_indexes = %d' %(train_loss_add, TrainImgFrames))
        print('mean Cls_loss_add of 1 sample: %f, #train_indexes = %d' %(Cls_loss_add, TrainImgFrames))
        print('mean Reg_loss_add of 1 sample: %f, #train_indexes = %d' %(Reg_loss_add, TrainImgFrames))
    
        Error_test = 0
        Error_train = 0
        Error_test_wrist = 0
        Accuracy_test = 0

        if (epoch % 1 == 0):  
            net = net.eval()
            output = torch.FloatTensor()
            labels = torch.FloatTensor()
            outputTrain = torch.FloatTensor()
    
            for i, (img, label) in tqdm(enumerate(test_dataloaders)):
                with torch.no_grad():
                    # img, label = img.cuda(), label.cuda()   
                    img, label = img.to(device), label.to(device)     
                    heads = net(img)  
                    pred_keypoints = post_precess(heads, voting=False)
                    output = torch.cat([output,pred_keypoints.data.cpu()], 0)
                    labels = torch.cat([labels,label.data.cpu()], 0)
                    
                    Cls_loss_val, Reg_loss_val = criterion(heads, label)
                    loss_val = 1 * Cls_loss_val + Reg_loss_val * RegLossFactor
                    val_loss_add = val_loss_add + (loss_val.item()) * len(img)
                    Cls_loss_val_add = Cls_loss_val_add + (Cls_loss_val.item()) * len(img)
                    Reg_loss_val_add = Reg_loss_val_add + (Reg_loss_val.item()) * len(img)

#                   Visualize outputs in Validation - block taken from MP3DHP
                    depth_max = 5#6
                    depth_mean = 3#Img_mean
                    depth_std = 2#Img_std
                    img, label = img.cpu(), label.cpu()
                    img_ = img.numpy().copy()
                    img_ = img_[0, 0, :, :]
                    pred_keypoints = pred_keypoints.cpu().numpy()
                    pred = pred_keypoints[0]
                    pred_new = pred.copy()                    
                    pred_new[:, 0] = pred[:, 1]
                    pred_new[:, 1] = pred[:, 0]
                    pred_new = [pred_new]

                    single_img = np.copy(img_)
                    single_img *= depth_std
                    single_img += depth_mean
                    single_img[single_img >= depth_max] = depth_max
                    single_img /= depth_max
                    single_img *= 255
                    single_img = draw_humans_visibility(single_img,
                                                        pred_new,
                                                        kp_connections(get_keypoints()),
                                                        jointColors)
                    plt.imshow(single_img)
                    single_img = single_img.astype(np.uint8)
                    cv2.imwrite(save_dir + '/images/output_updated/epoch_' + str(epoch) + '_output_' + str(i) + '.png', single_img)
#    
            val_loss_add     = val_loss_add / TestImgFrames
            Cls_loss_val_add = Cls_loss_val_add / TestImgFrames
            Reg_loss_val_add = Reg_loss_val_add / TestImgFrames
            print('mean val_loss_add of 1 sample: %f, #test_indexes = %d' % (val_loss_add, TestImgFrames))
            print('mean Cls_loss_val_add of 1 sample: %f, #test_indexes = %d' % (Cls_loss_val_add, TestImgFrames))
            print('mean Reg_loss_val_add of 1 sample: %f, #test_indexes = %d' % (Reg_loss_val_add, TestImgFrames))

            result = output.cpu().data.numpy() # result is in 288x288
            targetLabels = labels.cpu().data.numpy() # this label is in 288x288

            # Accuracy_test = evaluation10CMRule(result, targetLabels, bndbox_test)
            Accuracy_test = evaluation10CMRule(result,keypointsWorldtest,bndbox_test)
            print('Accuracy:', Accuracy_test)
            # evaluation10CMRule_perJoint(result, targetLabels, bndbox_test)
            evaluation10CMRule_perJoint(result,keypointsWorldtest,bndbox_test)

            # Error_test = errorCompute(result, targetLabels, bndbox_test)
            Error_test = errorCompute(result,keypointsWorldtest,bndbox_test)
            print('epoch: ', epoch, 'Test error:', Error_test)

            saveNamePrefix = '%s/net_%d_wetD_' % (save_dir, epoch) + str(Weight_Decay) + '_depFact_' + str(spatialFactor) + '_RegFact_' + str(RegLossFactor) + '_rndShft_' + str(RandCropShift)
            torch.save(net.state_dict(), saveNamePrefix + '.pth')
    
        # tensorboard and log 
        # writer.add_scalar('Loss/Train', train_loss_add, epoch)
        # writer.add_scalar('Loss/Test', val_loss_add, epoch)
        writer.add_scalar('Accuracy/Test', Accuracy_test, epoch)
        writer.add_scalar('Error/Test', Error_test, epoch)
        writer.add_scalars('Loss', {'Training Loss':train_loss_add, 'Validation Loss':val_loss_add}, epoch)
        logging.info('Epoch#%d: total loss=%.4f, Cls_loss=%.4f, Reg_loss=%.4f, Err_test=%.4f, lr = %.6f'
        %(epoch, train_loss_add, Cls_loss_add, Reg_loss_add, Error_test, scheduler.get_last_lr()[0]))

def test():
    net = model.A2J_model(num_classes = keypointsNumber)
    net.load_state_dict(torch.load('/mnt/beegfs/home/ramesh/A2J/src_train/result/ITOP_batch_64_12345/net_25_wetD_0.0001_depFact_0.5_RegFact_3_rndShft_5.pth'))
    net = net.cuda()
    net.eval()
    
    post_precess = anchor.post_process(shape=[cropHeight//16,cropWidth//16],stride=16,P_h=None, P_w=None)
    
    output = torch.FloatTensor()
    torch.cuda.synchronize() 
    for i, (img, label) in tqdm(enumerate(test_dataloaders)):    
        with torch.no_grad():
    
            # img, label = img.cuda(), label.cuda()    
            img, label = img.to(device), label.to(device) 
            heads = net(img)  
            pred_keypoints = post_precess(heads,voting=False)
            output = torch.cat([output,pred_keypoints.data.cpu()], 0)
        
    torch.cuda.synchronize()       
    
    result = output.cpu().data.numpy()
    # writeTxt(result)
    error = errorCompute(result, keypointsWorldtest, bndbox_test)
    print('Error:', error)

def evaluation10CMRule(source, target, bndbox):
    assert np.shape(source) == np.shape(target), "source has different shape with target"
    Test1_ = np.zeros(source.shape)
    Test1_[:, :, 0] = source[:,:,1]
    Test1_[:, :, 1] = source[:,:,0]
    Test1_[:, :, 2] = source[:,:,2]
    Test1 = Test1_  # [x, y, z]

    for i in range(len(source)):
        Test1[i,:,0] = Test1_[i,:,0]*(bndbox[i,2]-bndbox[i,0])/cropWidth + bndbox[i,0]  # x
        Test1[i,:,1] = Test1_[i,:,1]*(bndbox[i,3]-bndbox[i,1])/cropHeight + bndbox[i,1]  # y
        Test1[i,:,2] = Test1_[i,:,2]/depthFactor

    # print(f"Test1[0,:,:] = {Test1[0,:,:]}")
    # print(f"Test2[0,:,:] = {Test2[0,:,:]}")
    outputs = np.ones((len(Test1),keypointsNumber,3))
    TestWorld_tuple = pixel2world(Test1[:,:,0],Test1[:,:,1],Test1[:,:,2])
    # outputs = pixel2world(Test1.copy())
    outputs[:,:,0] = TestWorld_tuple[0]
    outputs[:,:,1] = TestWorld_tuple[1]
    outputs[:,:,2] = Test1[:,:,2]
    count = 0
    accuracy = 0
    for i in range(len(source)):
        for j in range(keypointsNumber):
            answer = np.square(outputs[i, j, 0] - target[i, j, 0]) + np.square(outputs[i, j, 1] - target[i, j, 1]) + np.square(outputs[i, j, 2] - target[i, j, 2])
            if answer < np.square(0.1):  # 10cm # 0.1
                count = count + 1
            if i == 0:
                print(f"i = {i}\tj = {j}\tanswer = {answer}\tcount = {count}")

    accuracy = count / (len(source) * keypointsNumber)
    return accuracy

def evaluation10CMRule_perJoint(source, target, bndbox):
    assert np.shape(source) == np.shape(target), "source has different shape with target"
    Test1_ = np.zeros(source.shape)
    Test1_[:, :, 0] = source[:,:,1]
    Test1_[:, :, 1] = source[:,:,0]
    Test1_[:, :, 2] = source[:,:,2]
    Test1 = Test1_  # [x, y, z]

    for i in range(len(source)):
        Test1[i,:,0] = Test1_[i,:,0]*(bndbox[i,2]-bndbox[i,0])/cropWidth + bndbox[i,0]  # x
        Test1[i,:,1] = Test1_[i,:,1]*(bndbox[i,3]-bndbox[i,1])/cropHeight + bndbox[i,1]  # y
        Test1[i,:,2] = Test1_[i,:,2]/depthFactor

    outputs = np.ones((len(Test1),keypointsNumber,3))
    TestWorld_tuple = pixel2world(Test1[:,:,0],Test1[:,:,1],Test1[:,:,2])
    # outputs = pixel2world(Test1.copy())
    outputs[:,:,0] = TestWorld_tuple[0]
    outputs[:,:,1] = TestWorld_tuple[1]
    outputs[:,:,2] = Test1[:,:,2]

    count = 0
    accuracy = 0
    for j in range(keypointsNumber):
        for i in range(len(source)):
            answer = np.square(outputs[i, j, 0] - target[i, j, 0]) + np.square(outputs[i, j, 1] - target[i, j, 1]) + np.square(outputs[i, j, 2] - target[i, j, 2])
            if answer < np.square(0.1):  # 10cm # 0.1
                count = count + 1   

        accuracy = count / (len(source))
        print('joint_', j, joint_id_to_name[j], ', accuracy: ', accuracy)
        accuracy = 0
        count = 0


def errorCompute(source, target, bndbox):
    assert np.shape(source) == np.shape(target), "source has different shape with target"
    Test1_ = np.zeros(source.shape)
    Test1_[:, :, 0] = source[:,:,1]
    Test1_[:, :, 1] = source[:,:,0]
    Test1_[:, :, 2] = source[:,:,2]
    Test1 = Test1_  # [x, y, z]

    for i in range(len(source)):
        Test1[i,:,0] = Test1_[i,:,0]*(bndbox[i,2]-bndbox[i,0])/cropWidth + bndbox[i,0]  # x
        Test1[i,:,1] = Test1_[i,:,1]*(bndbox[i,3]-bndbox[i,1])/cropHeight + bndbox[i,1]  # y
        Test1[i,:,2] = Test1_[i,:,2]/depthFactor

    outputs = np.ones((len(Test1),keypointsNumber,3))
    TestWorld_tuple = pixel2world(Test1[:,:,0],Test1[:,:,1],Test1[:,:,2])
    # outputs = pixel2world(Test1.copy())
    outputs[:,:,0] = TestWorld_tuple[0]
    outputs[:,:,1] = TestWorld_tuple[1]
    outputs[:,:,2] = Test1[:,:,2]
    errors = np.sqrt(np.sum((target - outputs) ** 2, axis=2))
    
    return np.mean(errors)

# def writeTxt(result, center):
    # resultUVD_ = result.copy()
    # resultUVD_[:, :, 0] = result[:,:,1]
    # resultUVD_[:, :, 1] = result[:,:,0]
    # resultUVD = resultUVD_  # [x, y, z]
    
    # center_pixel = center.copy()
    # centre_world = pixel2world2(center.copy(), fx, fy, u0, v0)

    # centerlefttop = centre_world.copy()
    # centerlefttop[:, 0, 0] = centerlefttop[:, 0, 0] - xy_thres
    # centerlefttop[:, 0, 1] = centerlefttop[:, 0, 1] + xy_thres

    # centerrightbottom = centre_world.copy()
    # centerrightbottom[:, 0, 0] = centerrightbottom[:, 0, 0] + xy_thres
    # centerrightbottom[:, 0, 1] = centerrightbottom[:, 0, 1] - xy_thres

    # lefttop_pixel = world2pixel2(centerlefttop, fx, fy, u0, v0)
    # rightbottom_pixel = world2pixel2(centerrightbottom, fx, fy, u0, v0)

    
    # for i in range(len(result)):
    #     Xmin = max(lefttop_pixel[i, 0, 0], 0)
    #     Ymin = max(lefttop_pixel[i, 0, 1], 0)
    #     Xmax = min(rightbottom_pixel[i, 0, 0], 320 * 2 - 1)
    #     Ymax = min(rightbottom_pixel[i, 0, 1], 240 * 2 - 1)
    #     # resultUVD[i,:,0] = resultUVD_[i,:,0]*320/cropWidth  # x
    #     # resultUVD[i,:,1] = resultUVD_[i,:,1]*240/cropHeight  # y
    #     # resultUVD[i,:,2] = result[i,:,2]
    #     resultUVD[i, :, 0] = resultUVD_[i, :, 0] * (Xmax - Xmin) / cropWidth + Xmin  # x
    #     resultUVD[i, :, 1] = resultUVD_[i, :, 1] * (Ymax - Ymin) / cropHeight+ Ymin  # y
    #     resultUVD[i, :, 2] = result[i, :, 2] + center[i][0][2]
    
    # resultReshape = resultUVD.reshape(len(result), -1)
    
    # with open(os.path.join(save_dir, result_file), 'w') as f:     
    #     for i in range(len(resultReshape)):
    #         for j in range(keypointsNumber*3):
    #             f.write(str(resultReshape[i, j])+' ')
    #         f.write('\n') 
    
    # f.close()

    
if __name__ == '__main__':
    train()
    # test()
