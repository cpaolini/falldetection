from torchsummary import summary
import torch
import os
import model as model
import anchor as anchor

keypointsNumber = 15
model_dir = '/mnt/beegfs/home/ramesh/A2J/src_train/result/MP3DHP_BgAugment_A2J/net_34_wetD_0.0001_depFact_1_RegFact_5_rndShft_15.pth'
    
float_model = './build/float_model'
if __name__ == "__main__":
      # load trained model
    net = model.A2J_model(num_classes = keypointsNumber)
    #net = torch.nn.DataParallel(net)

    #net.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth')))
    summary(net, (1, 288, 288))

