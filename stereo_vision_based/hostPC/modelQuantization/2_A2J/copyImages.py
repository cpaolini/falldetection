
from tqdm import tqdm
import json
import shutil
import os
from PIL import Image
import numpy as np
from PIL import ImageDraw

with open('/mnt/beegfs/home/ramesh/Quantization/A2J/SDSU_PSG_a2j_filepath.json') as f:
    train_test_dict = json.load(f)   

with open('/mnt/beegfs/home/ramesh/Quantization/A2J/SDSU_PSG_a2j_test.json') as f:
    dict = json.load(f)
    annotations_test = dict

test_image_ids_list = list(annotations_test.keys())

destinationPath="/mnt/beegfs/home/ramesh/Quantization/A2J/sdsu_psg_depth_validationSet/"
destPath="/mnt/beegfs/home/ramesh/Quantization/A2J/sdsu_psg_depth_valPNG/"

# for index in tqdm(range(len(test_image_ids_list))):
#     shutil.copy(train_test_dict[test_image_ids_list[index]], destinationPath)

for index in tqdm(range(len(test_image_ids_list))):
    source_path = train_test_dict[test_image_ids_list[index]]
    destination_file_path = os.path.join(destinationPath, f"{index}.npy")
    arr = np.load(destination_file_path)
    im = Image.fromarray(np.uint8(arr * 255), 'L')
    I1 = ImageDraw.Draw(im)
    I1.text((28, 36), str(test_image_ids_list[index]))

    path = os.path.join(destPath, f"{index}.png")
    im.save(path)
    shutil.copy(source_path, destination_file_path)


print(f"Done Copying check your files in {destinationPath}")