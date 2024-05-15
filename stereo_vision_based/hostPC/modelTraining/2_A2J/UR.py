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
import anchor as anchor
from tqdm import tqdm
import random_erasing
import logging
import time
import datetime
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# DataHyperParms
TrainImgFrames = 1817#17991
TestImgFrames = 516#4863
keypointsNumber = 15
cropWidth = 288
cropHeight = 288
batch_size = 32
learning_rate = 0.00035
Weight_Decay = 1e-4
nepoch = 26
RegLossFactor = 3
spatialFactor = 0.5
RandCropShift = 5
RandshiftDepth = 1
RandRotate = 180
RandScale = (1.0, 0.5)

randomseed = 12345
random.seed(randomseed)
np.random.seed(randomseed)
torch.manual_seed(randomseed)

save_dir = './result/UR_FALL_batch_32_12345'

try:
    os.makedirs(save_dir)
except OSError:
    pass

trainingImageDir = '/mnt/beegfs/home/ramesh/Datasets/UR_Fall_Detection_Dataset/Fall_Dataset/Train/5. NormalizedDepthImages/'
testingImageDir = '/mnt/beegfs/home/ramesh/Datasets/UR_Fall_Detection_Dataset/Fall_Dataset/Test/5. NormalizedDepthImages/'  # mat images

# keypointsfileTest = '/mnt/beegfs/home/ramesh/A2J/data/itop_side/itop_side_keypoints3D_test.mat'
# keypointsfileTrain = '/mnt/beegfs/home/ramesh/A2J/data/itop_side/itop_side_keypoints3D_train.mat'

bndbox_train = scio.loadmat('../data/ur_fall/BoundaryBox_Train.mat')['bndbox']
bndbox_test =  scio.loadmat('../data/ur_fall/BoundaryBox_Test.mat')['bndbox']
# Img_mean = np.load('../data/itop_side/itop_side_mean.npy')[3]
# Img_std = np.load('../data/itop_side/itop_side_std.npy')[3]

Img_mean = 27477.211
Img_std = 3032.3

model_dir = '/mnt/beegfs/home/ramesh/A2J/model/ITOP_side.pth'
result_file = 'result_test.txt'
monitoringValue = 1


def pixel2world(x):
    # x[:, :, 0] = (x[:, :, 0] - 160.0) * x[:, :, 2] * 0.0035
    # x[:, :, 1] = (120.0 - x[:, :, 1]) * x[:, :, 2] * 0.0035
    x[:, :, 0] = (x[:, :, 0] - 640.0) * x[:, :, 2]
    x[:, :, 1] = (480.0 - x[:, :, 1]) * x[:, :, 2]
    return x


def world2pixel(x):
    x[:, :, 0] = 160.0 + x[:, :, 0] / (x[:, :, 2] * 0.0035)
    x[:, :, 1] = 120.0 - x[:, :, 1] / (x[:, :, 2] * 0.0035)
    return x


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

# loading GT keypoints and center points
# keypointsWorldtest = scio.loadmat(keypointsfileTest)['keypoints3D'].astype(np.float32)
# keypointsPixeltest = np.ones((len(keypointsWorldtest), 15, 2), dtype='float32')
# keypointsPixeltest = world2pixel(keypointsWorldtest)

# keypointsWorldtrain = scio.loadmat(keypointsfileTrain)['keypoints3D'].astype(np.float32)
# keypointsPixeltrain = np.ones((len(keypointsWorldtrain), 15, 2), dtype='float32')
# keypointsPixeltrain = world2pixel(keypointsWorldtrain)


def transform(img, matrix):
# def transform(img, label, matrix):
    '''
    img: [H, W] label, [N,2]
    '''
    img_out = cv2.warpAffine(img, matrix, (cropWidth, cropHeight))
    # label_out = np.ones((keypointsNumber, 3))
    # label_out[:, :2] = label[:, :2].copy()
    # label_out = np.matmul(matrix, label_out.transpose())
    # label_out = label_out.transpose()

    return img_out


def dataPreprocess(index, img, bndbox, depth_thres=0.4, augment=True):
# def dataPreprocess(index, img, keypointsUVD, depth_thres=0.4, augment=True):
    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32')
    # labelOutputs = np.ones((keypointsNumber, 3), dtype='float32')

    if augment:
        RandomOffset_1 = np.random.randint(-1 * RandCropShift, RandCropShift)
        RandomOffset_2 = np.random.randint(-1 * RandCropShift, RandCropShift)
        RandomOffset_3 = np.random.randint(-1 * RandCropShift, RandCropShift)
        RandomOffset_4 = np.random.randint(-1 * RandCropShift, RandCropShift)
        RandomOffsetDepth = np.random.normal(0, RandshiftDepth, cropHeight * cropWidth).reshape(cropHeight, cropWidth)
        RandomOffsetDepth[np.where(RandomOffsetDepth < RandshiftDepth)] = 0
        RandomRotate = np.random.randint(-1 * RandRotate, RandRotate)
        RandomScale = np.random.rand() * RandScale[0] + RandScale[1]
        matrix = cv2.getRotationMatrix2D((cropWidth / 2, cropHeight / 2), RandomRotate, RandomScale)
    else:
        RandomOffset_1, RandomOffset_2, RandomOffset_3, RandomOffset_4 = 0, 0, 0, 0
        RandomRotate = 0
        RandomScale = 1
        RandomOffsetDepth = 0
        matrix = cv2.getRotationMatrix2D((cropWidth / 2, cropHeight / 2), RandomRotate, RandomScale)

    new_Xmin = bndbox[index][1]
    new_Ymin = bndbox[index][2]
    new_Xmax = bndbox[index][3]
    new_Ymax = bndbox[index][4]

    imCrop = img.copy()[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)]
    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)

    imgResize = np.asarray(imgResize, dtype='float32')  # H*W*C
    imgResize = (imgResize - Img_mean) / Img_std
    if index == monitoringValue - 1:
    #     print('\n'+DIVIDER)
    #     print(f'new_Xmin = {new_Xmin}')
    #     print(f'new_Ymin = {new_Ymin}')
    #     print(f'new_Xmax = {new_Xmax}')
    #     print(f'new_Ymax = {new_Ymax}')
    #     print('\n'+DIVIDER)
    #     #cv2.imshow('img',img)
        cv2.imshow('imgCrop',imCrop)
    #     #cv2.imshow('imgResize',imgResize)
    #     cv2.imshow('imgResize post normalize',imgResize)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ## label
    # label_xy = np.ones((keypointsNumber, 2), dtype='float32')
    # label_xy[:, 0] = keypointsUVD[index, :, 0].copy() * cropWidth / 320  # x
    # label_xy[:, 1] = keypointsUVD[index, :, 1].copy() * cropHeight / 240  # y

    if augment:
        imgResize = transform(imgResize, matrix)  ## rotation, scale

    # print(f"Shape of imageoutputs = {imageOutputs.shape}")
    # print(f"Shape of imgResize = {imgResize.shape}")

    #    imageOutputs[:,:,0] = imgResize
    imageOutputs = imgResize
    # print("Done Copying")

    # labelOutputs[:, 1] = label_xy[:, 0]
    # labelOutputs[:, 0] = label_xy[:, 1]
    # labelOutputs[:, 2] = (keypointsUVD[index, :, 2]) * RandomScale  # Z

    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)
    # labelOutputs = np.asarray(labelOutputs)

    data = torch.from_numpy(imageNCHWOut)
    return data


###################### Pytorch dataloader #################
class my_dataloader(torch.utils.data.Dataset):
    def __init__(self, ImgDir, bndbox, num, augment=True):
    # def __init__(self, ImgDir, keypointsUVD, num, augment=True):
        self.ImgDir = ImgDir
        # self.keypointsUVD = keypointsUVD
        self.num = num
        self.bndbox = bndbox
        self.augment = augment
        self.randomErase = random_erasing.RandomErasing(probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0])

    def __getitem__(self, index):
        data4D = scio.loadmat(self.ImgDir + str(index + 1) + '.mat')['DepthNormal']
        depth = data4D[:, :]
        # print(f"index = {index}")
        data = dataPreprocess(index, depth, self.bndbox, self.augment)
        # data, label = dataPreprocess(index, depth, self.keypointsUVD, self.augment)

        if self.augment:
            data = self.randomErase(data)

        return data

    def __len__(self):
        return self.num


train_image_datasets = my_dataloader(trainingImageDir, bndbox_train, TrainImgFrames, augment=True)
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=batch_size,
                                                shuffle=True, num_workers=8)

test_image_datasets = my_dataloader(testingImageDir, bndbox_test, TestImgFrames, augment=False)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size,
                                               shuffle=False, num_workers=8)


def train():
    net = model.A2J_model(num_classes=keypointsNumber)
    net = net.cuda()

    post_precess = anchor.post_process(shape=[cropHeight // 16, cropWidth // 16], stride=16, P_h=None, P_w=None)
    criterion = anchor.A2J_loss(shape=[cropHeight // 16, cropWidth // 16], thres=[16.0, 32.0], stride=16, \
                                spatialFactor=spatialFactor, img_shape=[cropHeight, cropWidth], P_h=None, P_w=None)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=Weight_Decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                        filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
    logging.info('======================================================')

    for epoch in range(nepoch):
        net = net.train()
        train_loss_add = 0.0
        Cls_loss_add = 0.0
        Reg_loss_add = 0.0
        timer = time.time()

        # Training loop
        for i, (img) in enumerate(train_dataloaders):

            torch.cuda.synchronize()

            img = img.cuda()
            heads = net(img)
            # print(regression)
            optimizer.zero_grad()

            Cls_loss, Reg_loss = criterion(heads, label)

            loss = 1 * Cls_loss + Reg_loss * RegLossFactor
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()

            train_loss_add = train_loss_add + (loss.item()) * len(img)
            Cls_loss_add = Cls_loss_add + (Cls_loss.item()) * len(img)
            Reg_loss_add = Reg_loss_add + (Reg_loss.item()) * len(img)

            # printing loss info
            if i % 10 == 0:
                print('epoch: ', epoch, ' step: ', i, 'Cls_loss ', Cls_loss.item(), 'Reg_loss ', Reg_loss.item(),
                      ' total loss ', loss.item())

        scheduler.step(epoch)

        # time taken
        torch.cuda.synchronize()
        timer = time.time() - timer
        timer = timer / TrainImgFrames
        print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))

        train_loss_add = train_loss_add / TrainImgFrames
        Cls_loss_add = Cls_loss_add / TrainImgFrames
        Reg_loss_add = Reg_loss_add / TrainImgFrames
        print('mean train_loss_add of 1 sample: %f, #train_indexes = %d' % (train_loss_add, TrainImgFrames))
        print('mean Cls_loss_add of 1 sample: %f, #train_indexes = %d' % (Cls_loss_add, TrainImgFrames))
        print('mean Reg_loss_add of 1 sample: %f, #train_indexes = %d' % (Reg_loss_add, TrainImgFrames))

        Error_test = 0
        Error_train = 0
        Error_test_wrist = 0

        if (epoch % 1 == 0):
            net = net.eval()
            output = torch.FloatTensor()
            outputTrain = torch.FloatTensor()

            for i, (img) in tqdm(enumerate(test_dataloaders)):
                with torch.no_grad():
                    img = img.cuda()
                    heads = net(img)
                    pred_keypoints = post_precess(heads, voting=False)
                    output = torch.cat([output, pred_keypoints.data.cpu()], 0)

            result = output.cpu().data.numpy()
            # Error_test = errorCompute(result, keypointsWorldtest)
            print('epoch: ', epoch, 'Test error:', Error_test)
            saveNamePrefix = '%s/net_%d_wetD_' % (save_dir, epoch) + str(Weight_Decay) + '_depFact_' + str(
                spatialFactor) + '_RegFact_' + str(RegLossFactor) + '_rndShft_' + str(RandCropShift)
            # print(f"saveNamePrefix = {saveNamePrefix}")
            torch.save(net.state_dict(), saveNamePrefix + '.pth')

        # log
        logging.info('Epoch#%d: total loss=%.4f, Cls_loss=%.4f, Reg_loss=%.4f, Err_test=%.4f, lr = %.6f'
                     % (epoch, train_loss_add, Cls_loss_add, Reg_loss_add, Error_test, scheduler.get_lr()[0]))


def test():
    net = model.A2J_model(num_classes=keypointsNumber)
    net.load_state_dict(torch.load(
        '/mnt/beegfs/home/ramesh/A2J/src_train/result/UR_FALL_batch_32_12345/net_25_wetD_0.0001_depFact_0.5_RegFact_3_rndShft_5.pth'))
    net = net.cuda()
    net.eval()

    post_precess = anchor.post_process(shape=[cropHeight // 16, cropWidth // 16], stride=16, P_h=None, P_w=None)

    output = torch.FloatTensor()
    torch.cuda.synchronize()
    for i, (img) in tqdm(enumerate(test_dataloaders)):
        with torch.no_grad():
            img = img.cuda()
            heads = net(img)
            pred_keypoints = post_precess(heads, voting=False)
            output = torch.cat([output, pred_keypoints.data.cpu()], 0)

    torch.cuda.synchronize()

    result = output.cpu().data.numpy()
    writeTxt(result)
    # error = errorCompute(result, keypointsWorldtest)
    # print('Error:', error)


def errorCompute(source, target):
    assert np.shape(source) == np.shape(target), "source has different shape with target"

    Test1_ = source.copy()
    target_ = target.copy()
    Test1_[:, :, 0] = source[:, :, 1]
    Test1_[:, :, 1] = source[:, :, 0]
    Test1 = Test1_  # [x, y, z]

    for i in range(len(Test1_)):
        Test1[i, :, 0] = Test1_[i, :, 0] * 320 / cropWidth  # x
        Test1[i, :, 1] = Test1_[i, :, 1] * 240 / cropHeight  # y
        Test1[i, :, 2] = source[i, :, 2]

    labels = pixel2world(target_)
    outputs = pixel2world(Test1.copy())

    errors = np.sqrt(np.sum((labels - outputs) ** 2, axis=2))

    return np.mean(errors)


def writeTxt(result):
    resultUVD_ = result.copy()
    resultUVD_[:, :, 0] = result[:, :, 1]
    resultUVD_[:, :, 1] = result[:, :, 0]
    resultUVD = resultUVD_  # [x, y, z]

    for i in range(len(result)):
        resultUVD[i, :, 0] = resultUVD_[i, :, 0] * 640 / cropWidth  # x
        resultUVD[i, :, 1] = resultUVD_[i, :, 1] * 480 / cropHeight  # y
        resultUVD[i, :, 2] = result[i, :, 2]

    resultReshape = resultUVD.reshape(len(result), -1)

    with open(os.path.join(save_dir, result_file), 'w') as f:
        for i in range(len(resultReshape)):
            for j in range(keypointsNumber * 3):
                f.write(str(resultReshape[i, j]) + ' ')
            f.write('\n')

    f.close()


if __name__ == '__main__':
    train()
    test()
