import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.image_files = []

        # Load real images (label 0)
        real_dir = os.path.join(self.data_dir, 'real')
        if os.path.exists(real_dir):
            for f in os.listdir(real_dir):
                if f.endswith('.jpg'):
                    self.image_files.append((os.path.join(real_dir, f), 0))

        # Load fake images (label 1)
        fake_dir = os.path.join(self.data_dir, 'fake')
        if os.path.exists(fake_dir):
            for f in os.listdir(fake_dir):
                if f.endswith('.jpg'):
                    self.image_files.append((os.path.join(fake_dir, f), 1))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path, label = self.image_files[idx]
        image = read_image(img_path) / 255.0 # Read and scale to [0, 1]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)