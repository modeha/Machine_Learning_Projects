import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import yaml


# Define the config loading function
def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print("Loaded configuration:")
    print(config)
    return config


# IAM dataset loader class
class IAMCustomDataset(Dataset):
    def __init__(self, forms_path, xml_path, transform=None):
        self.forms_path = forms_path
        self.xml_path = xml_path
        self.transform = transform
        self.image_files = self._load_image_files()

        # Ensure there are valid image-label pairs
        if len(self.image_files) == 0:
            raise ValueError("No valid image-label pairs found in the dataset.")

    def _load_image_files(self):
        """Loads the image files and parses the corresponding XML metadata."""
        image_files = []
        for xml_file in os.listdir(self.xml_path):
            if xml_file.endswith(".xml"):
                # Parse XML to extract the label (modify based on your XML structure)
                tree = ET.parse(os.path.join(self.xml_path, xml_file))
                root = tree.getroot()

                # Find TextLine, handle the case if it's not found
                text_line = root.find(".//TextLine")
                if text_line is None:
                    print(f"Warning: No TextLine found in {xml_file}")
                    continue  # Skip this file if no TextLine is found

                # Find ImageFilename, handle the case if it's not found
                image_filename_element = root.find(".//ImageFilename")
                if image_filename_element is None:
                    print(f"Warning: No ImageFilename found in {xml_file}")
                    continue  # Skip this file if no ImageFilename is found

                label = text_line.attrib.get('text', 'Unknown')  # Use a default value if 'text' attribute is missing
                image_filename = image_filename_element.attrib.get('name', None)

                if image_filename is not None:
                    image_path = os.path.join(self.forms_path, image_filename + ".png")
                    if os.path.exists(image_path):
                        image_files.append((image_path, label))
                    else:
                        print(f"Warning: Image file {image_path} not found for {xml_file}")
                else:
                    print(f"Warning: Image filename is None in {xml_file}")

        print(f"Loaded {len(image_files)} valid image-label pairs.")
        return image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path, label = self.image_files[idx]
        image = Image.open(image_path).convert('L')  # Load grayscale image

        if self.transform:
            image = self.transform(image)

        # Return the image and the label (you may need to convert the label to tensor)
        return image, label


def load_datasets(config):
    """
    Load the IAM dataset using paths from the YAML config and return DataLoaders.
    Args:
    - config (dict): Configuration dictionary with dataset details.

    Returns:
    - dataloader_train (DataLoader): DataLoader for the training set.
    - dataloader_val (DataLoader): DataLoader for the validation set.
    """
    # Check if transforms exist in config, else apply default normalization
    if 'transforms' in config['data']:
        mean = config['data']['transforms'][1]['Normalize']['mean']
        std = config['data']['transforms'][1]['Normalize']['std']
    else:
        mean = [0.5]
        std = [0.5]

    # Define the transformation based on the config or default
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Load IAM dataset
    forms_path = config['data']['forms_path']
    xml_path = config['data']['xml_path']

    train_dataset = IAMCustomDataset(forms_path=forms_path, xml_path=xml_path, transform=transform)
    val_dataset = IAMCustomDataset(forms_path=forms_path, xml_path=xml_path, transform=transform)

    # Ensure that datasets are not empty
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Training or validation dataset is empty.")

    # Use only a portion of the dataset (e.g., 20%)
    portion = config['data'].get('portion', 1.0)  # Default to 100% if not specified

    # Get subset of indices for the training and validation sets
    num_train_samples = int(len(train_dataset) * portion)
    num_val_samples = int(len(val_dataset) * portion)

    if num_train_samples == 0 or num_val_samples == 0:
        raise ValueError("No samples found after applying dataset portion.")

    train_indices = np.random.choice(len(train_dataset), num_train_samples, replace=False)
    val_indices = np.random.choice(len(val_dataset), num_val_samples, replace=False)

    # Create subset datasets using the indices
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # Initialize dataloaders using the subsets
    dataloader_train = DataLoader(
        train_subset,
        batch_size=config['data']['dataloaders']['train']['batch_size'],
        shuffle=config['data']['dataloaders']['train']['shuffle'],
        num_workers=config['data']['dataloaders']['train']['num_workers']
    )

    dataloader_val = DataLoader(
        val_subset,
        batch_size=config['data']['dataloaders']['validation']['batch_size'],
        shuffle=config['data']['dataloaders']['validation']['shuffle'],
        num_workers=config['data']['dataloaders']['validation']['num_workers']
    )

    return dataloader_train, dataloader_val


def main():
    # Load the configuration
    config_path = 'C:\\Users\\mohse\\cdo-idp-dev\\configs\\htr_resnet\\htr_resnet_training_config.yaml'  # Path to your YAML file
    config = load_config(config_path)

    # Load datasets and dataloaders
    dataloader_train, dataloader_val = load_datasets(config)

    # Test the first batch from the training dataloader
    print("Testing training data loader...")
    for images, labels in dataloader_train:
        print("Batch of images shape:", images.shape)
        print("Batch of labels:", labels)
        break  # Just to test one batch

    # Test the first batch from the validation dataloader
    print("\nTesting validation data loader...")
    for images, labels in dataloader_val:
        print("Batch of images shape:", images.shape)
        print("Batch of labels:", labels)
        break  # Just to test one batch


if __name__ == "__main__":
    main()
