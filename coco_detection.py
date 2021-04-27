from config import *
import os
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms

class simpleDataset(data.Dataset) :
    def __init__(self, root, filenames, labels):
        self.root = root
        self.filenames = filenames
        self.labels = labels

    def __getitem__(self, index):
        img_filename = self.filenames[index]

        image = Image.open(os.path.join(self.root, img_filename))
        label = self.labels[index]

        image = transforms.ToTensor()(image)
        label = torch.as_tensor(label, dtype=torch.int64)

        return image, label

    def __len__(self):
        return len(self.filenames)


root = 'data/train2014'
filenames = os.listdir('data/train2014')[:5]
labels = [0,1,1]

my_dataset = simpleDataset(root=root,
                          filenames = filenames,
                          labels = labels)



batch_size = 1
num_workers = 0

data_loader = torch.utils.data.DataLoader(my_dataset,
                                          batch_size = batch_size,
                                          shuffle=False,
                                          num_workers=num_workers
                                          )

import numpy as np
import matplotlib.pyplot as plt

for imgs, labels in data_loader :
    img = transforms.ToPILImage()(imgs[0])
    plt.imshow(img)
    plt.show()
    print(labels)
