import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
from torchvision import models
from torch.nn import init
import resnet
from torch.nn import Parameter
from torch.autograd import Variable
import sys,os
import time
import math
import senet
import densenet


class Pyramidconv(nn.Module):
   def __init__(self, num_features_in, num_features_out):
        super(Pyramidconv, self).__init__()
        self.conv1 = nn.Conv2d(num_features_in, num_features_out//2, dilation=1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features_in, num_features_out//4, dilation=2, kernel_size=3, padding=2)
        self.conv3 = nn.Conv2d(num_features_in, num_features_out//4, dilation=3, kernel_size=3, padding=3)
   def forward(self, inputs):
        out1 = self.conv1(inputs)
        out2 = self.conv2(inputs)
        out3 = self.conv3(inputs)
        out = torch.cat((out1,out2,out3),1)
        return out
# https://github.com/yhenon/pytorch-retinanet/blob/master/model.py
class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        
        # upsample C5 to get P5 from the FPN paper
        self.P5 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=1, padding=1)
        #self.bn5 = nn.BatchNorm2d(feature_size)
        #self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        #self.P5_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        # add P5 elementwise to C4
        self.P4 = nn.Conv2d(C4_size, feature_size, kernel_size=3, stride=1, padding=1)
        #self.bn4 = nn.BatchNorm2d(feature_size)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        #self.P4_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P3 = nn.Conv2d(C3_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        #self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        # add P4 elementwise to C3
        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, inputs):
        C3, C4, C5 = inputs
        P5_x = self.P5(C5)
        #P5_upsampled_x = self.P5_upsampled(P5_x)        
        P4_x =self.P4(C4)
        P4_x = P5_x + P4_x		
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P3_x = self.P3(C3)
        P3_x = P3_x + P4_upsampled_x
        #P3_x = self.P3_2(P3_x)
        P3_x = self.relu(self.bn3(P3_x))
        return P3_x
class DepthRegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=16, num_classes=15, feature_size=256):
        super(DepthRegressionModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        #self.conv1 = Pyramidconv(num_features_in, feature_size)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        #self.conv2 = Pyramidconv(feature_size, feature_size)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        #self.conv3 = Pyramidconv(feature_size, feature_size)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        #self.conv4 = Pyramidconv(feature_size, feature_size)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        #self.feat_bn = nn.BatchNorm2d(feature_size)
        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        #out = self.feat_bn(out)
        out = self.output(out)

        #out = out.permute(0, 1, 3, 2)  ## [N, C, H, W] --->>> [N, C, W, H]

        # out is B x C x W x H, with C = 3*num_anchors
        out1 = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out2.contiguous().view(out2.shape[0], -1, self.num_classes)

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=16, num_classes=15, feature_size=256):
        super(RegressionModel, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        #self.conv1 = Pyramidconv(num_features_in, feature_size)
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        #self.conv2 = Pyramidconv(feature_size, feature_size)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        #self.conv3 = Pyramidconv(feature_size, feature_size)
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        #self.conv4 = Pyramidconv(feature_size, feature_size)
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors*num_classes*2, kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        out = self.output(out)

        #out = out.permute(0, 1, 3, 2)  ## [N, C, H, W] --->>> [N, C, W, H]

        # out is B x C x W x H, with C = 3*num_anchors
        out1 = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes, 2)
        return out2.contiguous().view(out2.shape[0], -1, self.num_classes, 2)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=16, num_classes=15, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        #self.conv1 = Pyramidconv(num_features_in, feature_size)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        #self.conv2 = Pyramidconv(feature_size, feature_size)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        #self.conv3 = Pyramidconv(feature_size, feature_size)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        #self.conv4 = Pyramidconv(feature_size, feature_size)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        #self.feat_bn = nn.BatchNorm2d(feature_size)
        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        #self.output_act = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        #out = self.feat_bn(out)
        out = self.output(out)
        #out = self.output_act(out)

        #out = out.permute(0, 1, 3, 2)  ## [N, C, H, W] --->>> [N, C, W, H]

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)







class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )
       
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         init.xavier_normal_(m.weight)
                

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = y.view(b,c)
        y = self.fc(y).view(b, c, 1, 1)
        return y  



class Se_DualPath(nn.Module):
    def __init__(self):
        super(Se_DualPath, self).__init__()
        
        modelPreTrain50 = resnet.resnet34(pretrained=True)
        modelPreTrain50_2 = resnet.resnet18(pretrained=True)
        self.model = modelPreTrain50
        self.model2 = modelPreTrain50_2
        
        #if block == BasicBlock:
        #fpn_sizes1 = [self.model.layer2[layers[1]-1].conv2.out_channels, self.model.layer3[layers[2]-1].conv2.out_channels, self.model.layer4[layers[3]-1].conv2.out_channels]
        #elif block == Bottleneck:
        #    fpn_sizes = [self.layer2[layers[1]-1].conv3.out_channels, self.layer3[layers[2]-1].conv3.out_channels, self.layer4[layers[3]-1].conv3.out_channels]
        #self.fpn1 = PyramidFeatures(fpn_sizes1[0], fpn_sizes1[1], fpn_sizes1[2])

        #fpn_sizes2 = [self.model2.layer2[layers[1]-1].conv2.out_channels, self.model2.layer3[layers[2]-1].conv2.out_channels, self.model2.layer4[layers[3]-1].conv2.out_channels]
        #self.fpn2 = PyramidFeatures(fpn_sizes2[0], fpn_sizes2[1], fpn_sizes2[2])

        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.SE1_1 = SELayer(64)
        self.SE2_1 = SELayer(64)
        self.SE1_2 = SELayer(128)
        self.SE2_2 = SELayer(128)
        self.SE1_3 = SELayer(256)
        self.SE2_3 = SELayer(256)
        self.SE1_4 = SELayer(512)
        self.SE2_4 = SELayer(512)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        #self.linear1 = nn.Linear(512, num_classes)
        #self.linear2 = nn.Linear(512, num_classes)
        #init.constant_(self.conv1.bias, 0)
        #init.xavier_normal_(self.conv1.weight)
        #init.xavier_normal_(self.linear1.weight)
        #init.xavier_normal_(self.linear2.weight)
        #self.weight = Parameter(torch.Tensor(num_classes))
    
    def forward(self, x): 
        n, c, h, w = x.size()
        
        x1 = x[:,0:3,:,:]  # normal
        x2 = x[:,3:4,:,:]  # depth
        x2 = x2.expand(n,3,h,w)
         
        #x2 = self.conv1(x2) 
        #x = torch.cat([x1,x2],0)       
        out1 = self.model.conv1(x1)
        out1 = self.model.bn1(out1)
        out1 = self.model.relu(out1)
        out1 = self.model.maxpool(out1)

        out2 = self.model2.conv1(x2)
        out2 = self.model2.bn1(out2)
        out2 = self.model2.relu(out2)
        out2 = self.model2.maxpool(out2)


        out1_1 = self.model.layer1(out1)
        out2_1 = self.model2.layer1(out2)
        out1_1,out2_1 = self.SE2_1(out2_1)*out1_1, self.SE1_1(out1_1)*out2_1  

        out1_2 = self.model.layer2(out1_1)
        out2_2 = self.model2.layer2(out2_1)
        out1_2,out2_2 = self.SE2_2(out2_2)*out1_2, self.SE1_2(out1_2)*out2_2  

        out1_3 = self.model.layer3(out1_2)
        out2_3 = self.model2.layer3(out2_2)
        out1_3,out2_3 = self.SE2_3(out2_3)*out1_3, self.SE1_3(out1_3)*out2_3  

        out1_4 = self.model.layer4(out1_3)
        out2_4 = self.model2.layer4(out2_3)
        out1_4,out2_4 = self.SE2_4(out2_4)*out1_4, self.SE1_4(out1_4)*out2_4  

        #out1 = self.fpn([x2, x3, x4])

        # out1 = self.avg_pool(out1)
        # out1 = self.bn1(out1)   
        

        # out2 = self.avg_pool(out2)
        # out2 = self.bn2(out2)

        # #print('#############', out1.size(), out2.size())
        # out1 = out1.view(out1.size(0), -1)
        # out2 = out2.view(out2.size(0), -1)
        
        # out1 = self.linear1(out1)
        # out2 = self.linear2(out2)

        out = torch.cat([out1_4, out2_4], 1)

        return out





class ResNetBackBone(nn.Module):
    def __init__(self):
        super(ResNetBackBone, self).__init__()
        
        modelPreTrain50 = resnet.resnet50(pretrained=True)
        #pretrain_coco = torch.load('pose_resnet_50_256x192.pth.tar')
        #modelPreTrain50.load_state_dict(pretrain_coco,False)
        self.model = modelPreTrain50
        #fpn_sizes = [self.model.layer2[-1].conv2.out_channels, self.model.layer3[-1].conv2.out_channels, self.model.layer4[-1].conv2.out_channels]
        #self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
    def forward(self, x): 
        n, c, h, w = x.size()  # x: [B, 1, H ,W]
        
        #x1 = x[:,0:3,:,:]  # normal
        x = x[:,0:1,:,:]  # depth
        x = x.expand(n,3,h,w)
              
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)
        #out = self.fpn([x2, x3, x4])
        return x3,x4  
		
class seResNextBackBone(nn.Module):
    def __init__(self):
        super(seResNextBackBone, self).__init__()
        
        modelPreTrain50 = senet.se_resnext50_32x4d()
        self.model = modelPreTrain50
        
    
    def forward(self, x): 
        n, c, h, w = x.size()  # x: [B, 1, H ,W]
        
        #x1 = x[:,0:3,:,:]  # normal
        x = x[:,0:1,:,:]  # depth
        x = x.expand(n,3,h,w)
              
        out = self.model.features(x)

        return out  


class DenseNetBackBone(nn.Module):
    def __init__(self):
        super(DenseNetBackBone, self).__init__()
        
        
        self.model = densenet.densenet121(pretrained=True)
        
    
    def forward(self, x): 
        n, c, h, w = x.size()  # x: [B, 1, H ,W]
        
        #x1 = x[:,0:3,:,:]  # normal
        x = x[:,0:1,:,:]  # depth
        x = x.expand(n,3,h,w)
              
        out = self.model.features(x)

        return out  


class Cls_Reg_Dual_Path_Net(nn.Module):
    def __init__(self, num_classes, useNormal=False, is_3D=True):
        super(Cls_Reg_Dual_Path_Net, self).__init__()
        self.is_3D = is_3D
        if useNormal:
            self.Backbone = Se_DualPath()   # normal + depth
            self.regressionModel = RegressionModel(1024, num_classes=num_classes)
            self.classificationModel = ClassificationModel(1024, num_classes=num_classes)
            self.DepthRegressionModel = DepthRegressionModel(2048, num_classes=num_classes)
        else:
            #self.Backbone = ResNetBackBone() # 1 channel depth only, resnet50
            self.Backbone = ResNetBackBone() # 1 channel depth only, resnet50
            #self.regressionModel = RegressionModel(512)
            #self.classificationModel = ClassificationModel(512, num_classes=num_classes)
            #self.DepthRegressionModel = DepthRegressionModel(512, num_classes=num_classes)        
            #self.Backbone = DenseNetBackBone()
            self.regressionModel = RegressionModel(2048, num_classes=num_classes)
            self.classificationModel = ClassificationModel(1024, num_classes=num_classes)
            if is_3D:
                self.DepthRegressionModel = DepthRegressionModel(2048, num_classes=num_classes)        
    
    def forward(self, x): 
        
        x3,x4 = self.Backbone(x)
        classification  = self.classificationModel(x3)
        regression = self.regressionModel(x4)
        if self.is_3D:
            DepthRegressionModel  = self.DepthRegressionModel(x4)
        #print(classification)
            return (classification, regression, DepthRegressionModel)
        return (classification, regression)




if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    timer = time.time()
    #net = Se_DualPath()
    net = Cls_Reg_Dual_Path_Net()
    print(net)
    #timer = time.time()
    y1 = net(Variable(torch.randn(4,1,160,160)))
    print(y1.size())
    #print(y2.size())
    timer = time.time() - timer
    print('time comsuming(sec): ',timer)  # cuda:20s, w/o cuda:80s
		
		