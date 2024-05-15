import numpy as np
import torch
import torch.nn as nn
import sys,os
import math
import torch.nn.functional as F


    
class FocalLoss(nn.Module):
    def __init__(self,P_h=[2,6], P_w=[2,6], shape=[8,4], stride=8,thres = [10.0,20.0],spatialFactor=0.1,img_shape=[0,0],is_3D=True):
        super(FocalLoss, self).__init__()
        anchors = generate_anchors(P_h=P_h, P_w=P_w)
        self.all_anchors = torch.from_numpy(shift(shape,stride,anchors)).cuda().float()
        self.thres = torch.from_numpy(np.array(thres)).cuda().float()
        self.spatialFactor = spatialFactor
        self.img_shape = img_shape
        self.is_3D = is_3D
    def calc_distance(self, a, b):
        dis = torch.zeros(a.shape[0],b.shape[0]).cuda()
        for i in range(a.shape[1]):
            dis += torch.abs(torch.unsqueeze(a[:, i], dim=1) - b[:,i])  
            #dis += torch.sqrt(torch.pow(torch.unsqueeze(a[:, i], dim=1) - b[:,i],2))
        return dis

    def forward(self, heads, annotations):
        alpha = 0.25
        gamma = 2.0
        if self.is_3D:
            classifications, regressions, depthregressions = heads
        else:
            classifications, regressions = heads
        #classifications,scalar,mu = classifications_tuple
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = self.all_anchors # num_anchors(w*h*A) x 2
        anchor_regression_loss_tuple = []

        for j in range(batch_size):

            #classification = torch.sigmoid(classifications[j, :, :]) #N*(w*h*A)*P
            classification = classifications[j, :, :] #N*(w*h*A)*P
            regression = regressions[j, :, :, :] #N*(w*h*A)*P*2
            if self.is_3D:
                depthregression = depthregressions[j, :, :]#N*(w*h*A)*P
            bbox_annotation = annotations[j, :, :]#N*P*3=>P*3
            #bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            #bbox_h = torch.clamp(bbox_annotation[:,0:1],0,self.img_shape[0])
            #bbox_w = torch.clamp(bbox_annotation[:,1:2],0,self.img_shape[1])
            #bbox_4anchor = torch.unsqueeze(torch.cat((bbox_h,bbox_w),1),1)#P*1*2
            reg_weight = F.softmax(classification,dim=0) #(w*h*A)*P
            reg_weight_xy = torch.unsqueeze(reg_weight,2).expand(reg_weight.shape[0],reg_weight.shape[1],2)#(w*h*A)*P*2			
            gt_xy = bbox_annotation[:,:2]#P*2 
            #print(reg_weight_xy.shape)
            #print(bbox_4anchor.shape)
            #print(gt_xy.shape)
            anchor_diff = torch.abs(gt_xy-(reg_weight_xy*torch.unsqueeze(anchor,1)).sum(0)) #P*2
            anchor_loss = torch.where(
                torch.le(anchor_diff, 1),
                0.5 * 1 * torch.pow(anchor_diff, 2),
                anchor_diff - 0.5 / 1
            )
            anchor_regression_loss = anchor_loss.mean()
            anchor_regression_loss_tuple.append(anchor_regression_loss)
            #classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            #dis = self.calc_distance(anchor, bbox_4anchor) # num_anchors(w*h*A) x num_annotations(P)

            #dis_min, dis_argmin = torch.min(dis, dim=1) # num_anchors x 1

            
            #targets = torch.ones(dis.shape) * -1 #(w*h*A)*(P)
            #targets = targets.cuda()
            #targets = torch.where(torch.lt(dis,self.thres[0]),torch.ones(targets.shape).cuda(),targets)
            #tmp = torch.where(torch.lt(dis,self.thres[0]),torch.ones(targets.shape).cuda(),torch.zeros(targets.shape).cuda())
            #print(tmp.sum(0))
            #targets = torch.where(torch.ge(dis,self.thres[1]),torch.zeros(targets.shape).cuda(),targets)
            
            #targets[torch.ge(dis_min, self.thres[1]).cpu().numpy(), :] = 0

            #positive_indices = torch.lt(dis_min, self.thres[0])
            #dis_min, dis_argmin = torch.min(dis, dim=1) # num_anchors x 1
            #positive_indices = torch.lt(dis_min, self.thres[0])
            #num_positive_anchors = positive_indices.sum()


            #print("pos_num",num_positive_anchors)
            #print("neg_num",negative_indices.sum())
            #assigned_annotations = dis_argmin

#######################conpute focal loss########################           
            #alpha_factor = torch.ones(targets.shape).cuda() * alpha            
            #alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            #focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            #focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            #bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            #print(classification)
            # cls_loss = focal_weight * torch.pow(bce, gamma)
            #cls_loss = focal_weight * bce
            #cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            #classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))
#######################regression 4 spatial###################
            reg = torch.unsqueeze(anchor,1) + regression #(w*h*A)*P*2
            regression_diff = torch.abs(gt_xy-(reg_weight_xy*reg).sum(0)) #P*2
            regression_loss = torch.where(
                torch.le(regression_diff, 1),
                0.5 * 1 * torch.pow(regression_diff, 2),
                regression_diff - 0.5 / 1
                )
            regression_loss = regression_loss.mean()*self.spatialFactor
########################regression 4 depth###################
            if self.is_3D:
                gt_depth = bbox_annotation[:,2] #P
                regression_diff_depth = torch.abs(gt_depth - (reg_weight*depthregression).sum(0))#(w*h*A)*P       
                regression_loss_depth = torch.where(
                    torch.le(regression_diff_depth, 3),
                    0.5 * (1/3) * torch.pow(regression_diff_depth, 2),
                    regression_diff_depth - 0.5 / (1/3)
                    )
                regression_loss += regression_diff_depth.mean()           
############################################################
            regression_losses.append(regression_loss)
        return torch.stack(anchor_regression_loss_tuple).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)   

def generate_anchors(P_h=None, P_w=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if P_h is None:
        P_h = np.array([2,6,10,14])

    if P_w is None:
        P_w = np.array([2,6,10,14])

    num_anchors = len(P_h) * len(P_h)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 2))
    k = 0
    for i in range(len(P_w)):
        for j in range(len(P_h)):
            anchors[k,1] = P_w[j]
            anchors[k,0] = P_h[i]
            k += 1  
    return anchors          


def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.
    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None,
    shapes_callback=None,
):

    image_shapes = compute_shape(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


class post_process(nn.Module):
    def __init__(self, P_h=[2,6], P_w=[2,6], shape=[48,26], stride=8,thres = 8,is_3D=True):
        super(post_process, self).__init__()
        anchors = generate_anchors(P_h=P_h,P_w=P_w)
        self.all_anchors = torch.from_numpy(shift(shape,stride,anchors)).cuda().float()
        self.thres = torch.from_numpy(np.array(thres)).cuda().float()
        self.is_3D = is_3D
    def calc_distance(self, a, b):
        dis = torch.zeros(a.shape[0],b.shape[0]).cuda()
        for i in range(a.shape[1]):
            #dis += torch.abs(torch.unsqueeze(a[:, i], dim=1) - b[:,i])  
            dis += torch.pow(torch.unsqueeze(a[:, i], dim=1) - b[:,i],0.5)
        return dis

    def forward(self,heads,voting=False):
        if self.is_3D:
            classifications, regressions, depthregressions = heads
        else:
            classifications, regressions = heads
#        classifications,scalar,mu = classifications_tuple
        batch_size = classifications.shape[0]
        anchor = self.all_anchors #*(w*h*A)*2
        P_keys = []
        for j in range(batch_size):

            classification = classifications[j, :, :] #N*(w*h*A)*P
            regression = regressions[j, :, :, :] #N*(w*h*A)*P*2
            if self.is_3D:
                depthregression = depthregressions[j, :, :]#N*(w*h*A)*P
            #zero_pad = torch.zeros(regression.shape[0],1).cuda()
            #anchor_pad = torch.cat((anchor,zero_pad),1)
            #print(anchor.shape)
            #print(regression.shape)
            reg = torch.unsqueeze(anchor,1) + regression
            reg_weight = F.softmax(classifications[j, :, :],dim=0) #(w*h*A)*P
            reg_weight_xy = torch.unsqueeze(reg_weight,2).expand(reg_weight.shape[0],reg_weight.shape[1],2)#(w*h*A)*P*2
            P_xy = (reg_weight_xy*reg).sum(0)
            if self.is_3D:
                P_depth = (reg_weight*depthregression).sum(0)
                P_depth = torch.unsqueeze(P_depth,1)
                P_key = torch.cat((P_xy,P_depth),1)            
                P_keys.append(P_key)
            else:
                P_keys.append(P_xy)
        return torch.stack(P_keys)



def shift(shape, stride, anchors):
    shift_h = np.arange(0, shape[0]) * stride
    shift_w = np.arange(0, shape[1]) * stride

    shift_h, shift_w = np.meshgrid(shift_h, shift_w)
    shifts = np.vstack((shift_h.ravel(), shift_w.ravel())).transpose()


    # add A anchors (1, A, 2) to
    # cell K shifts (K, 1, 2) to get
    # shift anchors (K, A, 2)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 2))

    return all_anchors

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #classifications=torch.zeros(4,4*8*16,5) #N*(w*h*A)*(P)
    #regressions=torch.rand(4,4*8*16,3) #N*(w*h*A)*(3)s
    #classifications=torch.zeros(32,4*8*16,5) #N*(w*h*A)*(P)
    #regressions=torch.rand(32,4*8*16,3) #N*(w*h*A)*(3)s
    #label = 50*torch.rand(32,5,3)
    #focal = FocalLoss()
    #res = focal(classifications.cuda(),regressions.cuda(),label.cuda())
    anchors = generate_anchors()
    all_anchors = shift(shape=[8,4],stride=16,anchors=anchors)
    print(res)
