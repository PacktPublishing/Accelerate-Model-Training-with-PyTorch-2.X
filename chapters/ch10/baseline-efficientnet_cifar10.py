import os
os.environ['OMP_NUM_THREADS'] = "16"
os.environ['KMP_AFFINITY'] = "granularity=fine,compact,1,0"

import time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torch.utils.data.distributed import DistributedSampler

def build_data_loader(data_dir, batch_size, random_seed=42, valid_size=0.1, shuffle=True, test=False):
    transform = transforms.Compose([transforms.ToTensor()])
    transform.crop_size=512
    transform.resize_size=512
    
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=2)
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

def main():
    # General parameters
    data_dir = '/tmp'
    device = "cuda"

    # Hyperparameters
    lr = 0.0001
    weight_decay = 0.005
    batch_size = 256
    num_epochs = 25 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam

    # Dataset 
    train_loader, test_loader = build_data_loader(data_dir=data_dir, batch_size=batch_size)

    # Model
    weights = models.EfficientNet_B7_Weights.DEFAULT
    net = models.efficientnet_b7(weights=weights)
    net.train()
    model = net.to(device)
         
    # Optimizer
    optimizer = optimizer(model.parameters(), lr, weight_decay=weight_decay)

    # Training 
    start = time.time()
    train(model, train_loader, num_epochs, criterion, optimizer, device)
    end = time.time()
    print('Training time: {} seconds'.format(int(end-start)))

    test(model, test_loader, device)

  

if __name__ == "__main__":
    main()








