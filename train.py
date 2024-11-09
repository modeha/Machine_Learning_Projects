from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from fire import Fire
from torch import device as Device, device
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from scipy.special import softmax
from torch.utils.data.datapipes.utils.decoder import Decoder

from classes.abstract_sampler import AbstractSampler
# from idp.decoders.decoder import Decoder
from classes.metrics import mean_average_precision, ocr_accuracy
from classes.process_data import train, validate
# from idp.utils.process_data import train, validate
from classes.scheduler import SchedulerWrapper
from models.detection_cnn import DetectionCNN
from models.my_cnn import MY_CNN
import os



import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from fire import Fire
from classes.custom_dataset import CustomDataset

def training_loop(
        device: Device,
        model: Module,
        optimizer: Optimizer,
        scheduler: SchedulerWrapper,
        dataloader_train: DataLoader,
        dataloader_val: DataLoader,
        decoder: Optional[Decoder],
        *,
        num_epochs: int = 100,
        early_stopping: Optional[int] = 12,
        gradient_accumulation_steps: int = 1,
        checkpoint_folder: str = "../checkpoint",
        save_every: Optional[int] = None,
        verbose: bool = True,
):
    train_losses = []
    val_losses = []
    char_accuracies = []
    word_accuracies = []
    mAPs = [[], [], []]
    best_val_loss = np.inf
    not_improved_counter = 0

    for epoch in range(1, num_epochs + 1):
        if verbose:
            print(f"Starting epoch {epoch} ...")
            print(f"Learning rate for epoch: {epoch} is: {scheduler.get_last_lr()}")

        # Losses
        train_outputs = train(device, model, dataloader_train, optimizer, scheduler, gradient_accumulation_steps)
        train_losses.append(train_outputs["loss"])

        val_outputs = validate(device, model, dataloader_val)
        val_losses.append(val_outputs["loss"])
        new_best_loss = val_losses[-1] < best_val_loss

        # Generate and save loss plot
        fig = plt.figure()
        plt.plot(range(1, epoch + 1), train_losses, label="train loss")
        plt.plot(range(1, epoch + 1), val_losses, label="val loss")
        plt.title("Training/validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(checkpoint_folder, "loss.png"))
        plt.close(fig)

        # Metrics
        if decoder:
            val_gt_words = [
                dataloader_val.dataset.vector_to_word(dataloader_val.dataset[index]["targets"])
                for index in (
                    dataloader_val.sampler.indices
                    if isinstance(dataloader_val.sampler, AbstractSampler)
                    else range(len(dataloader_val.dataset))
                )
            ]
            val_predicted_words = [
                decoder.decode(softmax(val_prediction, axis=1))[0] for val_prediction in val_outputs["predictions"]
            ]

            char_acc, word_acc = ocr_accuracy(val_predicted_words, val_gt_words)
            char_accuracies.append(char_acc)
            word_accuracies.append(word_acc)

            # Generate and save metric plot
            fig = plt.figure()
            plt.plot(range(1, epoch + 1), char_accuracies, label="char accuracy")
            plt.plot(range(1, epoch + 1), word_accuracies, label="word accuracy")
            plt.title("Validation character and word accuracies")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.ylim([0, 1])
            plt.legend()
            plt.savefig(os.path.join(checkpoint_folder, "accuracy.png"))
            plt.close(fig)
        else:
            # Ensure ground truth objects are extracted correctly
            ground_truth_objects = [datum[1] for datum in dataloader_val.dataset if len(datum) > 1]

            # Filter out invalid entries (integers) from detections and ground_truth_objects
            filtered_detections = [d for d in val_outputs["detections"] if isinstance(d, (np.ndarray, list, dict))]
            filtered_ground_truth_objects = [g for g in ground_truth_objects if isinstance(g, (np.ndarray, list, dict))]

            # Ensure they have the same length
            if len(filtered_detections) == len(filtered_ground_truth_objects):
                # Calculate mean average precision
                mAP, mAP_50, mAP_75 = mean_average_precision(filtered_detections, filtered_ground_truth_objects)
                mAPs[0].append(mAP)
                mAPs[1].append(mAP_50)
                mAPs[2].append(mAP_75)

                # Generate and save metric plot
                fig = plt.figure()
                plt.plot(range(1, epoch + 1), mAPs[0], label="mAP")
                plt.plot(range(1, epoch + 1), mAPs[1], label="mAP_50")
                plt.plot(range(1, epoch + 1), mAPs[2], label="mAP_75")
                plt.title("Validation mean average precision")
                plt.xlabel("Epoch")
                plt.ylabel("Mean average precision")
                plt.ylim([0, 1])
                plt.legend()
                plt.savefig(os.path.join(checkpoint_folder, "mean_average_precision.png"))
                plt.close(fig)

        # Checkpoints and early stopping
        if save_every and epoch % save_every == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_folder, f"CP{epoch:03}.pkl"))

        if new_best_loss:
            not_improved_counter = 0
            best_val_loss = val_losses[-1]
            torch.save(model.state_dict(), os.path.join(checkpoint_folder, f"CPBest.pkl"))
        else:
            not_improved_counter += 1

        if early_stopping and not_improved_counter > early_stopping:
            if verbose:
                print("Stopped training early ...")
            break


# Assume MY_CNN and CustomDataset are already defined elsewhere

import pandas as pd

def initialize_dataset(config, dataset_type, transform):
    """
    Initializes a dataset (train or validation) based on the configuration.

    Args:
        config (dict): The configuration dictionary loaded from YAML.
        dataset_type (str): 'train' or 'validation' to specify the dataset.
        transform (callable): Transformations to apply to the data.

    Returns:
        dataset: An instance of CustomDataset initialized with paths from config.
    """
    dataset_config = config['data']['datasets'][dataset_type]

    # Load the features and labels CSV files
    features_df = pd.read_csv(dataset_config['root'])
    labels_df = pd.read_csv(dataset_config['labels_path'])

    # Print out the columns to debug the issue
    print("Labels DataFrame columns:", labels_df.columns)

    # Initialize the CustomDataset with the loaded data
    return CustomDataset(features_df, labels_df, transform=transform)

#
# import os
# import yaml
# import torch
# from torch.utils.data import DataLoader
#
# def main(config_path):
#     # Set the device (CUDA or CPU)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # Load the YAML configuration file
#     with open(config_path, 'r') as file:
#         config = yaml.safe_load(file)
#
#     # Extract dataset paths from the configuration
#     train_features_path = config['data']['datasets']['train']['root']
#     train_labels_path = config['data']['datasets']['train']['labels_path']
#     val_features_path = config['data']['datasets']['validation']['root']
#     val_labels_path = config['data']['datasets']['validation']['labels_path']
#
#     # Initialize the train and validation datasets
#     train_dataset = CustomDataset(
#         features_path=train_features_path,
#         labels_path=train_labels_path,
#         transforms=None  # Add any required transforms here
#     )
#
#     val_dataset = CustomDataset(
#         features_path=val_features_path,
#         labels_path=val_labels_path,
#         transforms=None  # Add any required transforms here
#     )
#
#     # Create DataLoaders
#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=config['data']['dataloaders']['train']['batch_size'],
#         shuffle=config['data']['dataloaders']['train']['shuffle'],
#         num_workers=config['data']['dataloaders']['train']['num_workers']
#     )
#
#     val_loader = DataLoader(
#         dataset=val_dataset,
#         batch_size=config['data']['dataloaders']['validation']['batch_size'],
#         shuffle=config['data']['dataloaders']['validation']['shuffle'],
#         num_workers=config['data']['dataloaders']['validation']['num_workers']
#     )
#
#     # Load model configuration from the YAML file
#     model_config = config['model']
#     model = MY_CNN().to(device)
#
#     print(f"Model Type: {model_config['type']}")
#
#     # Define the optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['optimizer']['lr'])
#
#     # Define the learning rate scheduler
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                 step_size=config['training']['scheduler']['step_size'],
#                                                 gamma=config['training']['scheduler']['gamma'])
#
#     # Training loop setup
#     num_epochs = config['training_loop']['num_epochs']
#     print(f"Starting training for {num_epochs} epochs...")
#
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         for batch_idx, (data, target) in enumerate(train_loader):
#             data, target = data.to(device), target.to(device)
#
#             optimizer.zero_grad()
#             output = model(data)
#             loss = torch.nn.functional.cross_entropy(output, target)
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#
#         print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")
#
#         # Validation step
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for data, target in val_loader:
#                 data, target = data.to(device), target.to(device)
#                 output = model(data)
#                 val_loss += torch.nn.functional.cross_entropy(output, target).item()
#
#         print(f"Validation Loss: {val_loss / len(val_loader)}")
#
#         # Step the scheduler
#         scheduler.step()
#
#         # Save checkpoint every few epochs
#         if (epoch + 1) % config['training_loop']['save_every'] == 0:
#             checkpoint_folder = config['training_loop']['checkpoint_folder']
#             os.makedirs(checkpoint_folder, exist_ok=True)  # Ensure directory exists
#
#             checkpoint_path = os.path.join(checkpoint_folder, f"model_epoch_{epoch + 1}.pth")
#             torch.save(model.state_dict(), checkpoint_path)
#             print(f"Checkpoint saved: {checkpoint_path}")
#
#
#
# if __name__ == "__main__":
#     Fire(main)
import os
import yaml
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def compute_accuracy(output, target):
    """Compute the accuracy of the model."""
    _, predicted = torch.max(output, 1)  # Get the index of the max log-probability
    correct = (predicted == target).sum().item()
    return correct / target.size(0)

def plot_loss_and_accuracy(epoch, train_losses, val_losses, train_accuracies, val_accuracies, checkpoint_folder):
    """Plot and save the loss and accuracy for each epoch."""
    os.makedirs(checkpoint_folder, exist_ok=True)

    # Plot Loss
    plt.figure()
    plt.plot(range(1, epoch + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epoch + 1), val_losses, label="Validation Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(checkpoint_folder, "loss_plot.png"))
    plt.close()

    # Plot Accuracy
    plt.figure()
    plt.plot(range(1, epoch + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, epoch + 1), val_accuracies, label="Validation Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(checkpoint_folder, "accuracy_plot.png"))
    plt.close()

def main(config_path):
    # Set the device (CUDA or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # config_path = "C:/Users/mohse/cdo-idp-dev/configs/my_cnn/my_cnn_training_config.yaml"

    # Load the YAML configuration file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract dataset paths from the configuration
    train_features_path = config['data']['datasets']['train']['root']
    train_labels_path = config['data']['datasets']['train']['labels_path']
    val_features_path = config['data']['datasets']['validation']['root']
    val_labels_path = config['data']['datasets']['validation']['labels_path']

    # Initialize the train and validation datasets
    train_dataset = CustomDataset(
        features_path=train_features_path,
        labels_path=train_labels_path,
        transforms=None  # Add any required transforms here
    )

    val_dataset = CustomDataset(
        features_path=val_features_path,
        labels_path=val_labels_path,
        transforms=None  # Add any required transforms here
    )

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['data']['dataloaders']['train']['batch_size'],
        shuffle=config['data']['dataloaders']['train']['shuffle'],
        num_workers=config['data']['dataloaders']['train']['num_workers']
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config['data']['dataloaders']['validation']['batch_size'],
        shuffle=config['data']['dataloaders']['validation']['shuffle'],
        num_workers=config['data']['dataloaders']['validation']['num_workers']
    )

    # Load model configuration from the YAML file
    model_config = config['model']
    model = MY_CNN().to(device)

    print(f"Model Type: {model_config['type']}")

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['optimizer']['lr'])

    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=config['training']['scheduler']['step_size'],
                                                gamma=config['training']['scheduler']['gamma'])

    # Training loop setup
    num_epochs = config['training_loop']['num_epochs']
    print(f"Starting training for {num_epochs} epochs...")

    # Lists to store losses and accuracies for each epoch
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    checkpoint_folder = config['training_loop']['checkpoint_folder']

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train_samples = 0

        # Training phase
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            correct_train += (output.argmax(1) == target).sum().item()
            total_train_samples += target.size(0)

        train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train_samples
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")

        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val_samples = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss = torch.nn.functional.cross_entropy(output, target)
                total_val_loss += val_loss.item()
                correct_val += (output.argmax(1) == target).sum().item()
                total_val_samples += target.size(0)

        val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val_samples
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

        # Step the scheduler
        scheduler.step()

        # Save checkpoint every few epochs
        if (epoch + 1) % config['training_loop']['save_every'] == 0:
            os.makedirs(checkpoint_folder, exist_ok=True)  # Ensure directory exists
            checkpoint_path = os.path.join(checkpoint_folder, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Plot and save loss and accuracy graphs after each epoch
        plot_loss_and_accuracy(epoch + 1, train_losses, val_losses, train_accuracies, val_accuracies, checkpoint_folder)

if __name__ == "__main__":
    Fire(main)
