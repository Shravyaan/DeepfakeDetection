import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import DeepfakeDataset

# These are standard augmentations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Testing 'train' split loader...")
# Use one of the splits you processed (e.g., 'train')
try:
    train_dataset = DeepfakeDataset(data_dir='processed_data', 
                                  split='train', 
                                  transform=data_transforms)

    if len(train_dataset) == 0:
        print("No 'train' images found. Did you run extract_faces.py?")
    else:
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        # Get one batch
        images, labels = next(iter(train_loader))

        print(f"\nSuccess! Pulled one batch.")
        print(f"Batch of images shape: {images.shape}") # Should be [4, 3, 224, 224]
        print(f"Batch of labels: {labels}")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Did you run 'extract_faces.py' first?")
    print("Does your 'processed_data/train' folder have 'real' and 'fake' subfolders?")