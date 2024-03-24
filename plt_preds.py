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
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.lines import Line2D


# Load del modello addestrato
total_mean = torch.tensor([86.7113, 92.8464, 86.2080, 98.1235]) # per non far girare ogni volta il loop per il calcolo della media e della std
total_std = torch.tensor([35.2340, 35.0720, 36.5135, 35.6512])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=4,out_channels=6)
model.load_state_dict(torch.load('models/model_checkpoint.pth')['model_state_dict'])
model.to(device)


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
IMAGE_HEIGHT = IMAGE_WIDTH = 512
test_transform = A.Compose([
                    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                    A.Normalize(
                        mean=(0,0,0,0),
                        std=(1,1,1,1),
                        max_pixel_value = 255.0,
                    ),
                    ToTensorV2(),
                ]
    )
dataset_test = PotsdamDataset(RGBIR_folder, LABELS_folder, 'test', transform=test_transform, size=IMAGE_HEIGHT)
test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

legend = [
    ("Superfici impermeabili", [255, 255, 255]),
    ("Edifici", [0, 0, 255]),
    ("Vegetazione bassa", [0, 255, 0]),
    ("Vegetazione alta", [0, 255, 255]),
    ("Auto", [255, 255, 0]),
    ("Sfondo", [255, 0, 0])
]

y_pred = []
y_true = []
for inputs, targets in tqdm(test_loader):
    inputs = inputs.to(device)
    targets = targets.to(device, dtype=torch.long)

    with torch.no_grad():
      outputs = model(inputs) 
      predicted = outputs.argmax(dim=1)

    y_pred.extend(predicted.cpu().numpy())
    print(len(y_pred), y_pred)
    y_true.extend(targets.cpu().numpy())
    print(len(y_true), y_true)
    

    for i in tqdm(range(len(inputs))):
        input_image = inputs[i].permute(1, 2, 0).cpu().numpy()
        true_label = targets[i].cpu().numpy()
        predicted_label = predicted[i].cpu().numpy()
        acc = round(calculate_overall_accuracy(targets=targets[i], predictions=predicted[i]),2)
        print("Accuracy: ", acc*100,'%')

        fig, axes = plt.subplots(1, 3, figsize=(15, 7.5))

        # Plot dell'immagine di input
        axes[0].imshow(input_image)
        axes[0].set_title('Input Image')

        # Plot dell'etichetta reale
        axes[1].imshow(label2rgb(true_label))
        axes[1].set_title('True Label')

        # Plot dell'etichetta predetta
        axes[2].imshow(label2rgb(predicted_label))
        axes[2].set_title('My prediction')

        # Aggiungi la legenda
        legend_elements = []
        for class_name, color in legend:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=np.array(color) / 255, markersize=10, label=class_name, markeredgewidth=1))

        fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend))
        fig.tight_layout()  # Ottimizza la disposizione dei subplot
        plt.figtext(0.5, 0.95, f'Accuracy: {acc*100}%', ha='center')
        plt.show()


        
classes = ['Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car', 'Clutter/background']
y_true_array = np.hstack(y_true)
y_pred_array = np.hstack(y_pred)
conf_mat = confusion_matrix(y_true_array.flatten(), y_pred_array.flatten())
plt.figure(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d')
plt.xticks(ticks=np.arange(len(classes)) + 0.5, labels=classes, rotation=45)
plt.yticks(ticks=np.arange(len(classes)) + 0.5, labels=classes, rotation=0)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('confusion_matrix')
plt.show()
