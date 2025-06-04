"""
FER2013 Dataset Handler
Loads and preprocesses the Facial Expression Recognition dataset from Kaggle
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torchvision import transforms
from typing import Optional, Tuple


class FER2013Dataset(Dataset):
    """
    FER2013 Dataset class for loading facial expression images
    
    The dataset contains 48x48 grayscale images of faces with 7 emotion classes:
    0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral
    """
    
    emotion_map = {
        0: 'Angry',
        1: 'Disgust', 
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }
    
    def __init__(self, csv_path: str, split: str = 'train', transform: Optional[transforms.Compose] = None):
        """
        Args:
            csv_path: Path to fer2013.csv file
            split: One of 'train', 'val', or 'test'
            transform: Optional transform to be applied on images
        """
        # Load data
        self.data = pd.read_csv(csv_path)
        
        # Filter by split
        if split == 'train':
            self.data = self.data[self.data['Usage'] == 'Training'].reset_index(drop=True)
        elif split == 'val':
            self.data = self.data[self.data['Usage'] == 'PublicTest'].reset_index(drop=True)
        elif split == 'test':
            self.data = self.data[self.data['Usage'] == 'PrivateTest'].reset_index(drop=True)
        else:
            raise ValueError(f"Split must be one of ['train', 'val', 'test'], got {split}")
            
        self.transform = transform
        self.split = split
        
        print(f"Loaded {split} set with {len(self.data)} samples")
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: Tensor of shape (C, H, W)
            label: Integer emotion label (0-6)
        """
        # Get pixel values and reshape
        pixels = self.data.iloc[idx]['pixels']
        pixels = np.array([int(p) for p in pixels.split()], dtype=np.uint8)
        image = pixels.reshape(48, 48)
        
        # Convert to 3-channel image (RGB) for compatibility with pretrained models
        image = np.stack([image] * 3, axis=-1)
        
        # Get label
        label = int(self.data.iloc[idx]['emotion'])
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: just convert to tensor
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            
        return image, label
    
    def get_class_counts(self) -> dict:
        """Returns the number of samples per emotion class"""
        counts = self.data['emotion'].value_counts().to_dict()
        return {self.emotion_map[k]: v for k, v in counts.items()}


# Transform definitions
def get_transforms(augmentation: bool = False) -> dict:
    """
    Returns transform pipelines for train and val/test
    
    Args:
        augmentation: If True, includes data augmentation for training
    """
    # Normalization values (approximate for grayscale images)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    
    if augmentation:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    # Val/Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return {
        'train': train_transform,
        'val': test_transform,
        'test': test_transform
    }


def get_data_loaders(csv_path: str, 
                     batch_size: int = 64,
                     augmentation: bool = False,
                     num_workers: int = 2) -> dict:
    """
    Creates data loaders for train, validation, and test sets
    
    Args:
        csv_path: Path to fer2013.csv
        batch_size: Batch size for loading
        augmentation: Whether to use data augmentation
        num_workers: Number of parallel workers for data loading
        
    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders
    """
    transforms = get_transforms(augmentation=augmentation)
    
    # Create datasets
    train_dataset = FER2013Dataset(csv_path, split='train', transform=transforms['train'])
    val_dataset = FER2013Dataset(csv_path, split='val', transform=transforms['val'])
    test_dataset = FER2013Dataset(csv_path, split='test', transform=transforms['test'])
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Created data loaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def setup_kaggle_colab():
    """
    Setup Kaggle API in Google Colab
    This function handles the kaggle.json upload and API setup
    """
    import os
    from google.colab import files
    
    # Create kaggle directory
    os.makedirs('/root/.kaggle', exist_ok=True)
    
    # Check if kaggle.json already exists
    if not os.path.exists('/root/.kaggle/kaggle.json'):
        print("Please upload your kaggle.json file...")
        uploaded = files.upload()
        
        # Move the file to the correct location
        for filename in uploaded.keys():
            os.rename(filename, '/root/.kaggle/kaggle.json')
    
    # Set permissions
    os.chmod('/root/.kaggle/kaggle.json', 600)
    
    print("Kaggle API setup complete!")


def download_fer2013_dataset():
    """
    Download FER2013 dataset from Kaggle competition
    Returns the path to the extracted CSV file
    """
    import os
    
    # Check if dataset already exists
    if os.path.exists('fer2013.csv'):
        print("Dataset already exists!")
        return 'fer2013.csv'
    
    print("Downloading FER2013 dataset from Kaggle...")
    
    # Download the dataset
    os.system("kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge")
    
    # Extract the dataset
    print("Extracting dataset...")
    os.system("unzip -q challenges-in-representation-learning-facial-expression-recognition-challenge.zip")
    
    # Clean up zip file
    os.system("rm challenges-in-representation-learning-facial-expression-recognition-challenge.zip")
    
    # List files to confirm
    files = os.listdir('.')
    print(f"Files in directory: {files}")
    
    if 'fer2013.csv' in files:
        print("Dataset downloaded and extracted successfully!")
        return 'fer2013.csv'
    else:
        raise FileNotFoundError("fer2013.csv not found after extraction!")


def get_kaggle_dataset():
    """
    Complete setup: Install Kaggle API, setup credentials, and download dataset
    Call this function at the beginning of your notebook
    """
    import os
    
    # Install kaggle if not already installed
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle API...")
        os.system("pip install -q kaggle")
        print("Kaggle API installed!")
    
    # Setup Kaggle credentials
    setup_kaggle_colab()
    
    # Download dataset
    csv_path = download_fer2013_dataset()
    
    return csv_path
