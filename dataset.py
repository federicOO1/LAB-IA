import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
import torchvision
import os
import matplotlib.pyplot as plt
from torchvision import transforms, utils

import numpy as np
import PIL

import warnings
from tqdm.auto import tqdm
import torch.utils.checkpoint as checkpoint
import warnings
warnings.filterwarnings('ignore')
import torch.optim as optim
import shutil
import random
from sklearn.metrics import accuracy_score
import albumentations
import rasterio
from rasterio.plot import reshape_as_image

class PotsdamDataset(Dataset):
    def __init__(self, images_folder, labels_folder, train_val_test, transform=None, mean=None, std=None, size=None):
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.train_val_test = train_val_test
        self.image_paths = []
        self.world_file_paths = []
        self.mask_paths = []
        self.transform = transform
        self.mean = mean
        self.std = std
        self.size = size
        self.normalize = transforms.Normalize(mean=mean, std=std)


        img_folder = os.path.join(self.images_folder, train_val_test)
        lbl_folder = os.path.join(self.labels_folder, train_val_test)

        for file_name in os.listdir(img_folder):
          if file_name.endswith('.tif'):
              image_path = os.path.join(img_folder, file_name)
              self.image_paths.append(image_path)
              world_file_path = os.path.join(img_folder, file_name.replace('.tif', '.tfw'))
              if os.path.exists(world_file_path):
                            self.world_file_paths.append(world_file_path)

        for label_name in os.listdir(lbl_folder):
                    mask_path = os.path.join(lbl_folder, label_name)
                    if os.path.exists(mask_path):
                      self.mask_paths.append(mask_path)

        self.image_paths.sort()
        self.world_file_paths.sort()
        self.mask_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def get_image_paths(self, indices):
        return [self.image_paths[idx] for idx in indices]

    def get_mask_paths(self, indices):
        return [self.mask_paths[idx] for idx in indices]

    def load_world_file(self, world_file_path):
          lines = open(world_file_path).readlines()
          try:
              parameters = [float(line.strip()) for line in lines if line.strip()]
              if len(parameters) == 6:
                  return parameters
              else:
                  raise ValueError("Il file .tfw non contiene 6 parametri.")
          except Exception as e:
              print(f"Errore durante la lettura dei parametri di georeferenziazione: {str(e)}")
              return None

    def RGB_to_class(self, rgb_label):
        # Mappa colori con classe
        colors_to_labels = {
            (255, 255, 255): 0,
            (0, 0, 255): 1,
            (0, 255, 255): 2,
            (0, 255, 0): 3,
            (255, 255, 0): 4,
            (255, 0, 0): 5
        }

        # Trasponi l'array per avere le dimensioni (6000, 6000, 3)
        transposed_label = np.transpose(rgb_label, (1, 2, 0))

        class_label = np.zeros((6000, 6000), dtype=np.int64)

        for color, label in colors_to_labels.items():
            #print("np.array(color).reshape(1, 1, 3)",np.array(color).reshape(1, 1, 3))
            #print("\n\n\n trasp_label:",transposed_label)
            mask = np.all(transposed_label == np.array(color).reshape(1, 1, 3), axis=-1)
            #print("mask:",mask)
            class_label[mask] = label
            #print("\n\n\n class_label:",class_label)

        #class_label_tensor = torch.tensor(class_label, dtype=torch.long)

        return class_label

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        world_file_path = self.world_file_paths[idx]
        mask_path = self.mask_paths[idx]

        image = rasterio.open(image_path).read()

        world_params = self.load_world_file(world_file_path)

        mask = rasterio.open(mask_path).read()

        label = self.RGB_to_class(mask)

        if self.transform is not None:
            # Transponi l'immagine per avere le dimensioni (C, H, W)
            #print("1",image.shape)
            image = image.transpose(1, 2, 0)
            #print("image:",image.shape,"label:",label.shape)
            augmented = self.transform(image=image,mask=label)
            image = augmented['image'].numpy()
            label = augmented['mask'].numpy()
            #print("image:",image.shape,"label:",label.shape)

        return image, label
