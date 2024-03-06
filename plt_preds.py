from model import UNet
from dataset import PotsdamDataset
from utils import (calculate_iou, calculate_miou, plot_metrics, calculate_overall_accuracy)
import torch
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_overall_accuracy
from rasterio.plot import show

# Load del modello addestrato
total_mean = torch.tensor([86.7113, 92.8464, 86.2080, 98.1235]) # per non far girare ogni volta il loop per il calcolo della media e della std
total_std = torch.tensor([35.2340, 35.0720, 36.5135, 35.6512])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=4,out_channels=6)
model.load_state_dict(torch.load('models/model_checkpoint.pth')['model_state_dict'])
model.to(device)
test_transform = A.Compose([
    A.Resize(height=768, width=768),
    A.Normalize(
              mean=(0, 0, 0, 0),
              std=(1, 1, 1, 1),
              max_pixel_value=255.0,
            ),
   ToTensorV2(always_apply=True),
        ]
    )

model.eval()

def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 255]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 255, 255]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [255, 0, 0]
    return mask_rgb


RGBIR_folder = 'data/4_Ortho_RGBIR'
LABELS_folder = 'data/5_Labels_all'
BATCH_SIZE = 4
IMAGE_HEIGHT = 512

dataset_test = PotsdamDataset(RGBIR_folder, LABELS_folder, 'test', transform=test_transform, size=IMAGE_HEIGHT)
test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

for inputs, targets in tqdm(test_loader):
    inputs = inputs.to(device)
    targets = targets.to(device, dtype=torch.long)

    with torch.no_grad():
      outputs = model(inputs)
      probabilities = F.softmax(outputs, dim=1)  
      _, predicted = torch.max(probabilities, 1)
      


    for i in tqdm(range(len(inputs))):
        input_image = inputs[i].permute(1, 2, 0).cpu().numpy()
        true_label = targets[i].cpu().numpy()
        predicted_label = predicted[i].cpu().numpy()
        acc = calculate_overall_accuracy(targets=targets[i], predictions=predicted[i])
        print(" Accuracy: ", acc)

        plt.figure(figsize=(10, 5))

        # Plot dell'immagine di input
        plt.subplot(1, 3, 1)
        plt.imshow(input_image)
        plt.title('Input Image')

        # Plot dell'etichetta reale
        plt.subplot(1, 3, 2)
        plt.imshow(label2rgb(true_label))
        plt.title('True Label')

        # Plot dell'etichetta predetta
        plt.subplot(1, 3, 3)
        plt.imshow(label2rgb(predicted_label))
        plt.title('My prediction')

        plt.show()
