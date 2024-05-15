import os
import json
import tqdm
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from model import FallModel
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import numpy as np

save_dir = './result/SDSU_PSG_1Frame/'
try:
    os.makedirs(save_dir)
except OSError:
    pass

class loadDataset(Dataset):
    def __init__(self, json_file_path, window_size):
        self.data_list, self.label_list = self.read_data_from_json(json_file_path, window_size)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        label = self.label_list[index]

        return data, label

    def read_data_from_json(self, json_file_path, window_size):
        with open(json_file_path, 'r') as f:
            grouped_data_dict = json.load(f)

        data_list = []
        label_list = []

        for group_id, data in grouped_data_dict.items():
            temp = data['3d_joints']
            arr = np.array(temp)
            # print(f"length of array = {arr.shape}, bagfile = {group_id}")
            keypoints = torch.Tensor(data['3d_joints'])
            label = data['isFall']
            num_frames = keypoints.shape[0]
            if keypoints.shape == (15, 3):
            # if keypoints.shape == (200, 15, 3):
            # if keypoints.shape == (4, 15, 3):
                keypoints.resize_(1,15,3)
                data_list.append(keypoints)
                label_list.append(label)
        print(f'number of Falls = {label_list.count(1)}\tnumber of Non-Falls = {label_list.count(0)}')

        return data_list, label_list


# Define the training function
def train_model(train_dataloader, test_dataloader, num_epochs):
    # Instantiate the FallNetBase model
    # model = FallNetBase(num_joints_in=15, num_features_in=45, conv_channels=512, num_xyz=3, num_class=2)
    model = FallModel()

    # Set the initial learning rate and create the optimizer
    initial_lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    # Create the Cross Entropy Loss function
    criterion = nn.CrossEntropyLoss()

    # Create the learning rate scheduler for exponential decay
    scheduler = ExponentialLR(optimizer, gamma=0.98)

    # Set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # model.load_state_dict(torch.load("/mnt/beegfs/home/ramesh/Quantization/FallDetector/build/float_model/ur_fall/2_fall_detection_model.pth"), strict=True)

    # TensorBoard writer
    writer = SummaryWriter(os.path.join(save_dir, 'Tensorboard/runs/'))

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                    filename=os.path.join(save_dir, 'train.log'), force=True, level=logging.INFO)
    logging.info('======================================================')
    # Training loop
    global_step = 0
    prev_accuracy = float(0)
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        timer = time.time()        
        train_loss_add = 0.0
        val_loss_add = 0.0
        accuracy_val_add = 0.0
        targets_list = []
        preds_list = []
        targets_val_list = []
        preds_val_list = []
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            # torch.cuda.synchronize() 

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # torch.cuda.synchronize()

            # Update metrics
            preds = torch.argmax(outputs, dim=1)
            accuracy = torch.sum(preds == targets).item() / len(targets)
            targets_list.extend(targets.cpu().numpy())
            preds_list.extend(preds.cpu().numpy())

            # Write to TensorBoard
            # writer.add_scalar("Loss/Train", loss.item(), global_step)
            # writer.add_scalar("Accuracy/Train", accuracy, global_step)

            train_loss_add = train_loss_add + (loss.item())*len(inputs)
            # Print the loss for every few iterations
            # if batch_idx % 50 == 0:
            #     print(f"Epoch [{epoch}/{num_epochs}], Iteration [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item()}, Accuracy: {accuracy}")

            global_step += 1

        # Adjust the learning rate
        scheduler.step()
        train_loss_add = train_loss_add / len(train_dataloader.dataset)
        train_cm = confusion_matrix(targets_list, preds_list)
        train_f1 = f1_score(targets_list, preds_list)
        train_precision = precision_score(targets_list, preds_list)
        train_recall = recall_score(targets_list, preds_list)
        train_accuracy = accuracy_score(targets_list, preds_list)
        # time taken
        # torch.cuda.synchronize()
        # timer = time.time() - timer
        # timer = timer / len(train_dataloader.dataset)
        # print('==> time to learn 1 sample = %f (ms)' %(timer*1000))

        if epoch % 1 == 0:
            model.eval() # Set the model to evaluation mode
            print(75*"*")
            count = 0
            for idx, (keypoint, label) in enumerate(test_dataloader):
                with torch.no_grad():
                    keypoint = keypoint.to(device)
                    label = label.to(device)
                    # Forward pass
                    output = model(keypoint)
                    loss_val = criterion(output, label)
                    # Update metrics
                    pred = torch.argmax(output, dim=1)
                    targets_val_list.extend(label.cpu().numpy())
                    preds_val_list.extend(pred.cpu().numpy())
                    accuracy_val = torch.sum(pred == label).item() / len(label)

                    val_loss_add = val_loss_add + (loss_val.item()) * len(keypoint)
                    accuracy_val_add = accuracy_val_add + accuracy_val

                    # Print the loss for every few iterations
                    if idx % 469 == 1:
                        print(f"Epoch [{epoch}/{num_epochs}], Iteration [{idx}/{len(test_dataloader)}], Loss: {val_loss_add/idx:.4f}, Accuracy: {accuracy_val_add/idx:.4f}")
                    count += 1

            val_loss_add = val_loss_add / len(test_dataloader.dataset)
            accuracy_val_add = accuracy_val_add / len(test_dataloader.dataset)
            val_cm = confusion_matrix(targets_val_list, preds_val_list)
            tn, fp, fn, tp = confusion_matrix(targets_val_list, preds_val_list).ravel()
            val_f1 = f1_score(targets_val_list, preds_val_list)
            val_precision = precision_score(targets_val_list, preds_val_list)
            val_recall = recall_score(targets_val_list, preds_val_list)
            val_accuracy = accuracy_score(targets_val_list, preds_val_list)
            
        # Write to TensorBoard
        writer.add_scalar("Accuracy/Test", accuracy_val_add, epoch)
        writer.add_scalars('Loss', {'Training Loss':train_loss_add, 'Validation Loss':val_loss_add}, epoch)
        logging.info('Epoch#%d: lr = %.6f, Training_loss = %.4f, Validation_loss = %.4f, Accuracy_test = %.4f'
                %(epoch, scheduler.get_last_lr()[0], train_loss_add, val_loss_add, accuracy_val_add))
        logging.info('targets_val_list: %s' % targets_val_list)
        logging.info('preds_val_list: %s' % preds_val_list)
        logging.info('Train Confusion Matrix:\n%s' % train_cm)
        logging.info('Train F1 Score:\t%.4f' % train_f1)
        logging.info('Train Precision:\t%.4f' % train_precision)
        logging.info('Train Recall:\t%.4f' % train_recall)
        logging.info('Train Accuracy:\t%.4f' % train_accuracy)
        logging.info('Validation Confusion Matrix:\n%s' % val_cm)
        logging.info('tn = %s, fp = %s, fn = %s, tp = %s' %( tn, fp, fn, tp ))
        logging.info('Validation F1 Score:\t%.4f' % val_f1)
        logging.info('Validation Precision:\t%.4f' % val_precision)
        logging.info('Validation Recall:\t%.4f' % val_recall)
        logging.info('Validation Accuracy:\t%.4f\n\n' % val_accuracy)
        print(f"accuracy = {accuracy_val_add}\tprev_accuracy = {prev_accuracy}")

        # Save the trained model
        if accuracy_val_add > prev_accuracy:
            torch.save(model.state_dict(), "/mnt/beegfs/home/ramesh/Quantization/FallDetector/build/1frame/float_model/"+str(epoch)+"_fall_detection_model.pth")
            prev_accuracy = accuracy_val_add
    # Close the TensorBoard writer
    writer.close()

# Define the main function
def main():

    # Create the training dataset
    # 
    # train_dataset = loadDataset('/mnt/beegfs/home/ramesh/Quantization/FallDetector/grouped_data_train.json', window_size = 5)
    # train_dataset = loadDataset('/mnt/beegfs/home/ramesh/SDSU_PSG_RGBD/python_scripts/SDSU_PSG_fall_NoFall_C1_C2.json', window_size = 5)
    # train_dataset = loadDataset('/mnt/beegfs/home/ramesh/SDSU_PSG_RGBD/python_scripts/train_data.json', window_size = 5)
    # train_dataset = loadDataset('/mnt/beegfs/home/ramesh/SDSU_PSG_RGBD/python_scripts/train_data_4fps.json', window_size = 5)
    train_dataset = loadDataset('/mnt/beegfs/home/ramesh/SDSU_PSG_RGBD/python_scripts/train_data_1frame.json', window_size = 5)

    # Define your training dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    # Create the testing dataset
    test_dataset = loadDataset('/mnt/beegfs/home/ramesh/SDSU_PSG_RGBD/python_scripts/validation_data_1fram.json', window_size = 5)
    # test_dataset = loadDataset('/mnt/beegfs/home/ramesh/SDSU_PSG_RGBD/python_scripts/validation_data_4fps.json', window_size = 5)

    # Define your training dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Set the number of epochs
    num_epochs = 100

    # Train the model
    train_model(train_dataloader, test_dataloader, num_epochs)

if __name__ == "__main__":
    main()
