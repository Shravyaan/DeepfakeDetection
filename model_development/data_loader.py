# model_development/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

class DeepFakeDataset(Dataset):
    def __init__(self, manifest_file, transform=None):
        self.transform = transform
        self.samples = []
        
        # Read manifest file
        with open(manifest_file, 'r') as f:
            for line in f:
                path, label = line.strip().split(',')
                self.samples.append((path, int(label)))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        try:
            # For video paths, we need to handle face images
            if path.endswith('.mp4'):
                # Convert video path to extracted face path
                video_name = os.path.splitext(os.path.basename(path))[0]
                label_folder = 'real' if label == 0 else 'fake'
                
                # Look for corresponding face images
                face_dir = f"../extracted_faces/{label_folder}"
                if os.path.exists(face_dir):
                    # Find face images for this video
                    face_files = [f for f in os.listdir(face_dir) if video_name in f]
                    if face_files:
                        # Use the first face image found
                        image_path = os.path.join(face_dir, face_files[0])
                        image = Image.open(image_path).convert('RGB')
                    else:
                        # Fallback: create a blank image
                        image = Image.new('RGB', (224, 224), color='gray')
                else:
                    image = Image.new('RGB', (224, 224), color='gray')
            else:
                # Direct image path
                image = Image.open(path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a dummy image
            image = Image.new('RGB', (224, 224), color='gray')
            if self.transform:
                image = self.transform(image)
            return image, label

def get_data_loaders(batch_size=32):
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DeepFakeDataset('../manifests/train_manifest.csv', transform=train_transform)
    test_dataset = DeepFakeDataset('../manifests/test_manifest.csv', transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"📊 Dataset stats:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader

# Test the data loader
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders(batch_size=4)
    
    for images, labels in train_loader:
        print(f"Batch image shape: {images.shape}")
        print(f"Batch labels: {labels}")
        break