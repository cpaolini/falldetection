import os
import time
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from model import FallModel

DIVIDER = '-----------------------------------------'

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
            keypoints = torch.Tensor(data['3d_joints'])
            label = data['isFall']
            num_frames = keypoints.shape[0]
            if keypoints.shape == (15, 3):
                keypoints.resize_(1,15,3)
                data_list.append(keypoints)
                label_list.append(label)
        print(f'number of Falls = {label_list.count(1)}\tnumber of Non-Falls = {label_list.count(0)}')
        return data_list, label_list

def test(model, device, test_loader):
  model.eval() # Set the model to evaluation mode
  accuracy_val = 0
  with torch.no_grad():
    for idx, (keypoint, label) in tqdm(enumerate(test_loader)): 
      keypoint, label = keypoint.to(device), label.to(device)
      # Forward pass
      output = model(keypoint)
      # Update metrics
      pred = torch.argmax(output, dim=1)
      accuracy_val += torch.sum(pred == label).item()

    acc = 100. * accuracy_val / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(accuracy_val, len(test_loader.dataset), acc))
  return


def main():
    float_model = './float_model' 
    test_dataset = loadDataset('./validation_data_1fram.json', window_size = 5)
    # test_dataset = loadDataset('./train_data_1frame.json', window_size = 5)
    device = torch.device('cpu')
    # Define your test dataloader
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = FallModel().to(device)
    model.load_state_dict(torch.load(os.path.join(float_model,'61_fall_detection_model.pth'), map_location=torch.device('cpu')))

    time1 = time.time()
    
    test(model, device, test_dataloader)
    
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(len(test_dataloader.dataset) / timetotal)
    print (DIVIDER)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" %(fps, len(test_dataloader.dataset), timetotal))

if __name__ == '__main__':
    main()