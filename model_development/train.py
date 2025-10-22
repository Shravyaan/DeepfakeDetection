# model_development/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from data_loader import get_data_loaders
from model import create_model

class Trainer:
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.writer = SummaryWriter('logs')
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=0.001
        )
        
        self.best_accuracy = 0.0
        os.makedirs('checkpoints', exist_ok=True)
    
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(self.train_loader)} | Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        self.writer.add_scalar('Training Loss', epoch_loss, epoch)
        self.writer.add_scalar('Training Accuracy', epoch_acc, epoch)
        
        return epoch_loss, epoch_acc
    
    def evaluate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(self.test_loader)
        epoch_acc = 100. * correct / total
        
        self.writer.add_scalar('Validation Loss', epoch_loss, epoch)
        self.writer.add_scalar('Validation Accuracy', epoch_acc, epoch)
        
        # Save best model
        if epoch_acc > self.best_accuracy:
            self.best_accuracy = epoch_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'accuracy': epoch_acc,
            }, 'checkpoints/best_model.pth')
            print(f"💾 New best model saved with accuracy: {epoch_acc:.2f}%")
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs):
        print(f"Starting training on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.evaluate(epoch)
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            print('-' * 50)
        
        self.writer.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(batch_size=16)  # Smaller batch for testing
    
    # Create model
    model = create_model(device)
    
    # Create trainer and start training
    trainer = Trainer(model, train_loader, test_loader, device)
    trainer.train(epochs=5)  # Start with 5 epochs for testing

if __name__ == "__main__":
    main()