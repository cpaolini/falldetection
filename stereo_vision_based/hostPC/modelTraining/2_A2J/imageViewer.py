import numpy as np
from matplotlib import pyplot as plt

img_array = np.load('/mnt/beegfs/home/ramesh/Datasets/MP-3DHP/ardata/dataset/test_bgaug/depth_maps/00000000.npy')

plt.imshow(img_array, cmap='gray')
plt.show()

