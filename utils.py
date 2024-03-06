import torch
import torchvision
import dataset
from torch.utils.data import DataLoader
import rasterio
import matplotlib.pyplot as plt
import csv

def calculate_iou(predictions, targets, num_classes):

    ious_per_class = []
    for cls in range(num_classes):
        intersection = torch.logical_and(predictions == cls, targets == cls).sum().float()
        union = torch.logical_or(predictions == cls, targets == cls).sum().float()
        iou = (intersection + 1e-6) / (union + 1e-6)  # Aggiungi epsilon per evitare divisione per zero
        ious_per_class.append(iou.item())
    return ious_per_class



def calculate_miou(predictions, targets, num_classes):

    ious_per_batch = calculate_iou(predictions, targets, num_classes)
    miou = sum(ious_per_batch) / num_classes
    return miou


def create_csv_file(filename, train_losses, val_losses, val_iou, val_miou, val_accuracy):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Scrivi l'intestazione con i nomi delle colonne
        writer.writerow(['train_loss', 'val_loss', 'iou_class_1', 'iou_class_2', 'iou_class_3', 'iou_class_4', 'iou_class_5', 'iou_class_6', 'miou', 'overall_accuracy'])

        # Scrivi i valori delle metriche per ogni epoca
        for epoch in range(len(train_losses)):
            # Estrai i valori per l'epoca corrente
            train_loss = train_losses[epoch]
            val_loss = val_losses[epoch]
            iou_class_values = val_iou[epoch]
            val_miou_value = val_miou[epoch]
            overall_accuracy_value = val_accuracy[epoch]

            # Scrivi i valori nella riga del file CSV
            writer.writerow([train_loss, val_loss] + iou_class_values + [val_miou_value, overall_accuracy_value])




def plot_metrics(train_losses, val_losses, val_iou, val_miou, val_accuracy, figsize=(12, 6)):

    epochs = range(len(train_losses))  # Calculate epochs based on train losses

    # Plot train and validation losses
    plt.figure(figsize=figsize)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Validation Losses')
    plt.grid(True)
    
    # Plot IoU per class
    colors = ['gray', 'blue', 'cyan', 'lime', 'yellow', 'red']
    classes = ['Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car', 'Clutter/background']
    plt.figure(figsize=figsize)
    for idx in range(len(val_iou[0])):
        class_iou_values = [iou[idx] for iou in val_iou]
        plt.plot(epochs, class_iou_values, label=f'Classe: {classes[idx]}', color=colors[idx])
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.title('Validation IoU per class')
    plt.grid(True)
    plt.show()

    # Plot mIoU
    plt.figure(figsize=figsize)
    plt.plot(epochs, val_miou, label='Validation mIoU', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.title('Validation mIoU')
    plt.grid(True)
    plt.show()

    # Plot validation accuracy
    plt.figure(figsize=figsize)
    plt.plot(epochs, val_accuracy, color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Overall Accuracy per Epoch')
    plt.grid(axis='y')
    for i, acc in enumerate(val_accuracy):
        plt.text(i + 1, acc + 0.01, f'{acc:.4f}', ha='center', va='bottom')

    # Show all plots
    plt.show()


def calculate_overall_accuracy(predictions, targets):

    correct = (predictions == targets).sum().item()
    total = targets.numel()
    accuracy = correct / total
    return accuracy