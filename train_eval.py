import warnings
warnings.filterwarnings('ignore')
from model import UNet
from dataset import PotsdamDataset
from utils import (calculate_iou, calculate_miou, plot_metrics, calculate_overall_accuracy, create_csv_file)
import torch
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import timedelta

LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RGBIR_folder = 'data/4_Ortho_RGBIR'
LABELS_folder = 'data/5_Labels_all'
BATCH_SIZE = 4
LOAD_MODEL = False
NUM_EPOCHS = 5
NUM_CLASSES = 6
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
best_mIoU = 0.0
save_best_model = True
train_losses_global = []
val_losses_global = []
val_iou_global = []



def train(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(loader):
        inputs, targets = inputs.to(device), targets.to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_iou = [0.0] * num_classes
    batch_miou = []
    total_accuracy = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            inputs, targets = inputs.to(device), targets.to(device, dtype=torch.long)
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1)
            ious_per_batch = calculate_iou(predictions, targets, num_classes)
            miou = calculate_miou(predictions, targets, num_classes)
            total_accuracy = calculate_overall_accuracy(predictions, targets)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            batch_miou.append(miou)
            for class_id in range(num_classes):
                total_iou[class_id] += ious_per_batch[class_id] * inputs.size(0)
    # Calculate average metrics
    avg_loss = total_loss / len(loader.dataset)
    avg_iou = [iou / len(loader.dataset) for iou in total_iou]
    avg_miou = torch.tensor(batch_miou).mean().numpy()
    avg_accuracy = total_accuracy / len(loader.dataset)

    # Return the calculated metrics
    return avg_loss, avg_iou, avg_miou, avg_accuracy


def main():

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Working on ",device)
        baseline_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                    mean=(0, 0, 0, 0),
                    std=(1, 1, 1, 1),
                    max_pixel_value=255.0,
                    ),
        ToTensorV2(always_apply=True),
                ]
            )
        train_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.RandomRotate90(),
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.Normalize(
                    mean=(0, 0, 0, 0),
                    std=(1, 1, 1, 1),
                    max_pixel_value=255.0,
                    ),
        ToTensorV2(always_apply=True),
                ]
            )
        val_transform = A.Compose(
                [
                    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                    A.Normalize(
                        mean=(0, 0, 0, 0),
                        std=(1, 1, 1, 1),
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2(),
                ]
        )

        model = UNet(in_channels=4, out_channels=6).to(device)

        params_to_update = model.parameters()

        optimizer = optim.Adam(params_to_update, lr=LEARNING_RATE)
        #scheduler = ExponentialLR(optimizer, gamma=0.95)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()

        if LOAD_MODEL:
            loaded_data = torch.load('models/model_and_metrics.pth')

            model.load_state_dict(loaded_data['model_state_dict'])
            optimizer.load_state_dict(loaded_data['optimizer_state_dict'])

        train_losses = []
        val_losses = []
        val_iou = []
        val_miou = []
        val_accuracy = []

        best_miou = 0.0
        best_epoch = 0


        # Definisci l'ottimizzatore

        transformed_dataset_train = PotsdamDataset(RGBIR_folder, LABELS_folder, 'train', transform=baseline_transform, size=IMAGE_HEIGHT)
        transformed_dataset_val = PotsdamDataset(RGBIR_folder, LABELS_folder, 'val', transform=val_transform, size=IMAGE_HEIGHT)

        train_loader = DataLoader(transformed_dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(transformed_dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


        for epoch in range(NUM_EPOCHS):
            # Fase di addestramento
            train_loss = train(model, train_loader, criterion, optimizer, device, scaler)
            train_losses.append(train_loss)

            # Fase idi validazione
            val_loss, val_iou_value, val_miou_value, val_accuracy_value = evaluate(model, val_loader, device, NUM_CLASSES)
            val_losses.append(val_loss)
            val_iou.append(val_iou_value)
            val_miou.append(val_miou_value)
            val_accuracy.append(val_accuracy_value)

            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation IoU per class: {val_iou_value}, Validation mIoU: {val_miou_value:.4f}, Validation Accuracy: {val_accuracy_value:.4f}')

            # Salvataggio del modello con la migliore mIoU
            if save_best_model and val_miou_value > best_miou:
                best_miou = val_miou_value
                best_epoch = epoch
                saved_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }
                save_path = 'models/baseline/model_3_2.pth'
                torch.save(saved_data, save_path)

            #scheduler.step()

        # Riepilogo
        print(f'Miglior mIoU: {best_miou:.4f} all\'epoca {best_epoch}')
        plot_metrics(train_losses, val_losses, val_iou, val_miou, val_accuracy)
        create_csv_file('models/baseline/metrics_checkpoint_3_2.csv', train_losses, val_losses, val_iou, val_miou, val_accuracy)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    exec_time = end_time - start_time

    print(f"Tempo totale di esecuzione: {str(timedelta(seconds=exec_time))}")