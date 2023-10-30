import os
os.environ['OMP_NUM_THREADS'] = "32"
os.environ['MKL_NUM_THREADS'] = "32"

import time
import logging
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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return (train_loader, test_loader)

def train(model, data_loader, num_epochs, criterion, optimizer, device):
    total_steps = len(train_loader)
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
               
        print ('Epoch [{}/{}], Loss: {:.4f}, time: {} seconds'.format(epoch+1, num_epochs, loss.item(), int(end-start)))
            
# General parameters
data_dir = '/tmp'
device = "cuda"

# Hyperparameters
lr = 0.00001
weight_decay = 0.005
batch_size = 512
num_epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam

# Dataset 
train_loader, test_loader = build_data_loader(data_dir=data_dir, batch_size=batch_size)

# Model
net = models.densenet121()
net.train()
model = net.cuda()

# Optimizer
optimizer = optimizer(model.parameters(), lr, weight_decay=weight_decay)

# Training 
train(model, train_loader, num_epochs, criterion, optimizer, device)











