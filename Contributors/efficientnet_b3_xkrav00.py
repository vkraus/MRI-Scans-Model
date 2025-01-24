import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_transforms = {
   'train': A.Compose([
       A.RandomResizedCrop(size=(224, 224)),
       A.HorizontalFlip(p=0.5),
       A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15),
       A.RandomBrightnessContrast(p=0.2),
       A.RandomGamma(p=0.2),
       A.GaussNoise(p=0.2),
       A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.2),
       A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
       ToTensorV2(),
   ]),
   'test': A.Compose([
       A.Resize(height=224, width=224),
       A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
       ToTensorV2(),
   ])
}

class AlzheimerDataset(Dataset):
   def __init__(self, parquet_path, transform=None):
       self.df = pd.read_parquet(parquet_path)
       self.transform = transform
       
   def __len__(self):
       return len(self.df)
   
   def __getitem__(self, idx):
       row = self.df.iloc[idx]
       image_bytes = row['image']['bytes']
       image = Image.open(BytesIO(image_bytes)).convert('RGB')
       image = np.array(image)
       
       if self.transform:
           augmented = self.transform(image=image)
           image = augmented['image']
           
       return image, row['label']

def compute_class_weights(labels):
   return compute_class_weight('balanced', classes=np.unique(labels), y=labels)

def get_model():
   model = models.efficientnet_b3(pretrained=True)  # Using b3 instead of b0
   model.classifier[1] = nn.Sequential(
       nn.Dropout(p=0.4),
       nn.Linear(1536, 512),
       nn.ReLU(),
       nn.Dropout(p=0.4),
       nn.Linear(512, 4)
   )
   return model.to(device)

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50):
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
   best_acc = 0.0
   
   for epoch in range(num_epochs):
       model.train()
       train_loss = 0
       train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
       
       for images, labels in train_bar:
           images, labels = images.to(device), labels.to(device)
           
           optimizer.zero_grad()
           outputs = model(images)
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()
           
           train_loss += loss.item()
           train_bar.set_postfix({'loss': train_loss/len(train_loader)})
           
       model.eval()
       test_loss = 0
       correct = 0
       total = 0
       
       test_bar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
       with torch.no_grad():
           for images, labels in test_bar:
               images, labels = images.to(device), labels.to(device)
               outputs = model(images)
               loss = criterion(outputs, labels)
               test_loss += loss.item()
               
               _, predicted = outputs.max(1)
               total += labels.size(0)
               correct += predicted.eq(labels).sum().item()
               
               accuracy = 100.*correct/total
               test_bar.set_postfix({
                   'loss': test_loss/len(test_loader),
                   'accuracy': accuracy
               })
       
       scheduler.step(accuracy)
       
       if accuracy > best_acc:
           best_acc = accuracy
           torch.save(model.state_dict(), 'best_model.pth')
           print(f'Best model saved with accuracy: {best_acc:.2f}%')

def main():
   train_path = 'Data/train-00000-of-00001-c08a401c53fe5312.parquet'
   test_path = 'Data/test-00000-of-00001-44110b9df98c5585.parquet'
   batch_size = 16
   learning_rate = 0.0001  # Lower learning rate
   
   train_dataset = AlzheimerDataset(train_path, data_transforms['train'])
   test_dataset = AlzheimerDataset(test_path, data_transforms['test'])
   
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   test_loader = DataLoader(test_dataset, batch_size=batch_size)
   
   labels = train_dataset.df['label'].values
   class_weights = torch.FloatTensor(compute_class_weights(labels)).to(device)
   
   model = get_model()
   criterion = nn.CrossEntropyLoss(weight=class_weights)
   optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.02)
   
   train_model(model, train_loader, test_loader, criterion, optimizer)

if __name__ == "__main__":
   main()
