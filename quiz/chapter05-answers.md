# Chapter 5 - Building an Efficient Data Pipeline

## Quiz answers

### 1. What three main tasks are executed during the data loading process?

A. Loading, scaling, and resizing

B. Scaling, resizing, and loading

C. Resizing, loading, and filtering

**D. Loading, preparation, and augmentation**

### 2. Data loading feeds which phase of the training process?

**A. Forward**

B. Backward

C. Optimization

D. Loss calculation

### 3. Which components provided by the `torch.utils.data` API can be used to implement a data pipeline?

A. Datapipe and DataLoader

B. Dataset and DataLoading

**C. Dataset and DataLoader**

D. Datapipe and DataLoading

### 4. Besides increasing the number of workers in the data pipeline, what can we do to improve the performance of the data loading process?

A. Reduce the size of the dataset

B. Do not use a GPU

C. Avoid the usage of high-dimensional images

**D. Optimize data transfer between the CPU and GPU**

### 5. How can we accelerate the data transfer between the CPU and GPU?

A. Use smaller datasets

B. Use the fastest GPUs

**C. Allocate and use pinned memory instead of pageable memory**

D. Increase the amount of main memory

### 6. What should we do to enable the usage of pinned memory on DataLoader?

A. Nothing. It is already enabled by default.

**B. Set the pin_memory parameter to True.**

C. Set the experimental_copy parameter to True.

D. Update PyTorch to version 2.0.

### 7. Why can using more than one worker on the pipeline accelerate data loading on PyTorch?

A. PyTorch reduces the amount of allocated memory

B. PyTorch enables the usage of special hardware capabilities

C. PyTorch uses the fastest links to communicate with GPUs

**D. PyTorch processes simultaneously more than one dataset sample**

### 8. Which of the following is true when making a request to allocate pinned memory?

A. It is always satisfied

**B. It can fail**

C. It always fails

D. It cannot be done through PyTorch
