import os
os.environ['OMP_NUM_THREADS'] = "32"
os.environ['MKL_NUM_THREADS'] = "32"
os.environ['NVIDIA_TF32_OVERRIDE'] = "1"

import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torch.utils.data.sampler import SubsetRandomSampler
#from torch.profiler import profile, record_function, ProfilerActivity

torch.backends.cudnn.benchmark = True
torch.set_num_threads(32)
torch.set_num_interop_threads(1)

def build_data_loader(data_dir, batch_size, random_seed=42, valid_size=0.1, shuffle=True, test=False):
    transform = transforms.Compose([transforms.ToTensor()])
    transform.crop_size=224
    transform.resize_size=224
    
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)

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
            

def train_amp(model, data_loader, num_epochs, criterion, optimizer, device, prof):
    scaler = torch.cuda.amp.GradScaler()
    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        start = time.time()
        for step, (images, labels) in enumerate(train_loader): 
            # Move tensors to the configured device
            images = images.cuda()
            labels = labels.cuda()

            with torch.autocast(device_type=device, dtype=torch.float16, enabled=False, cache_enabled=False):
                output = model(images)
                loss = criterion(output, labels)

            # Backward and optimize
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            prof.step()
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
num_epochs = 1
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam

# Dataset 
train_loader, test_loader = build_data_loader(data_dir=data_dir, batch_size=batch_size)

# Model
#net = models.efficientnet_b7()
net = models.densenet121()
net.train()
model = net.cuda()

# Optimizer
optimizer = optimizer(model.parameters(), lr, weight_decay=weight_decay)

# Training
'''
start = time.time()

import nvidia_dlprof_pytorch_nvtx as nvtx
nvtx.init()
with torch.autograd.profiler.emit_nvtx():
    train_amp(model, train_loader, num_epochs, criterion, optimizer, device)

end = time.time()
print('Training time: {} seconds'.format(int(end-start)))
'''
# Profiling
prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('/tmp/densenet121/'),
        record_shapes=True,
        with_stack=True)


#activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
#prof = kineto_profile(activities=activities, with_stack=True)
        
input_sample, _ = next(iter(train_loader))
input_sample.to(device)

prof.start()
train_amp(model, train_loader, num_epochs, criterion, optimizer, device, prof)
prof.stop()
'''
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
prof.export_stacks("/tmp/pytorch_log.stack", metric='self_cpu_time_total')
prof.export_chrome_trace("/tmp/chrome_stack_trace.txt")
print(prof.events())
'''









