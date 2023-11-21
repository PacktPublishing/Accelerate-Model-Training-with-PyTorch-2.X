import os
os.environ['OMP_NUM_THREADS'] = "16"
os.environ['KMP_AFFINITY'] = "granularity=fine,compact,1,0"

import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision import models

def build_data_loader(data_dir, batch_size, random_seed=42, valid_size=0.1, shuffle=True, test=False):
    transform = transforms.Compose([transforms.ToTensor()])
    transform.crop_size=224
    transform.resize_size=224
    
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
 
    return (train_loader, test_loader)

def train(model, train_loader, num_epochs, criterion, optimizer, device):
    total_steps = len(train_loader)
    print("Steps: {}".format(total_steps))
    for epoch in range(num_epochs):
        start = time.time()
        for step, (images, labels) in enumerate(train_loader):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
        
            # Forward pass
            outputs = model(images)
            outputs.to(device)
            loss = criterion(outputs, labels)
        
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print ('Step [{}/{}], Loss: {:.4f}'.format(step+1, total_steps, loss.item()))
        end = time.time()
               
       
        print('Epoch [{}/{}], Loss: {:.4f}, time: {} seconds'.format(epoch+1, num_epochs, loss.item(), int(end-start)))

def test(model, test_loader, device):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

    print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.fc1 = nn.Linear(64*8*8, 512)
        
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def main():
    # General parameters
    data_dir = '/tmp'
    device = "cpu"

    # Hyperparameters
    lr = 0.00001
    weight_decay = 0.005
    batch_size = 32
    num_epochs = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam

    # Dataset 
    train_loader, test_loader = build_data_loader(data_dir=data_dir, batch_size=batch_size)

    # Model
    model = CNN()
    
    # Optimizer
    optimizer = optimizer(model.parameters(), lr, weight_decay=weight_decay)
    
    # Training 
    start = time.time()
    train(model, train_loader, num_epochs, criterion, optimizer, device)
    end = time.time()
    print('Training time: {} seconds'.format(int(end-start)))

    # Testing
    test(model, test_loader, device)

if __name__ == "__main__":
    main()



# Using Intel OpenMP image with Gloo




