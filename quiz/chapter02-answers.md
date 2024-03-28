# Chapter 2 - Quiz answers

### 1. After running the training process using two GPUs in a single machine, we decided to add two extra GPUs to accelerate the training process. In this case, we tried to improve the performance of the training process by applying which of the following?

A. Horizontal scaling

**B. Vertical scaling**

C. Transversal scaling

D. Distributed scaling

### 2. The training process of a simple model is taking a long time to finish. After adjusting the batch size and cutting of one of the convolutional layers, we could train the model faster while achieving the same accuracy. In this case, we improve the performance of the training process by changing which of the following layers of the software stack?

**A. Application layer**

B. Hardware layer

C. Environment layer

D. Execution layer

### 3. Which of the following changes is applied to the environment layer?

A. Modify the hyperparameters

B. Adopt another network architecture

**C. Update the framework’s version**

D. Set a parameter in the operating system

### 4. Which one of the following components lies in the execution layer?
   
A. OpenMP

B. PyTorch

**C. Apptainer**

D. NCCL

### 5. As users of a given environment, we usually do not modify anything at the execution layer. What is the reason for that?

**A. We usually do not have administrative rights to change anything at the execution layer.**

B. There is no change at the execution layer that could accelerate the training process.

C. The execution and application layers are almost the same thing. So, there is no difference between changing one or another layer.

D. As we usually execute the training process on containers, there is no change on the execution layer that could improve the training process.

### 6. We have accelerated the training process of a given model by using two additional machines and applying a given capability provided by the machine learning framework. In this case, which of the following actions have we taken to improve the training process?

A. We have performed horizontal and vertical scaling

B. We have performed horizontal scaling and increased the number of resources

**C. We have performed horizontal scaling and applied changes to the environment layer**

D. We have performed horizontal scaling and applied changes to the execution layer

### 7. Controlling the behavior of a library through environment variables is a change that is applied in which of the following layers?

A. Application layer

**B. Environment layer**

C. Execution layer

D. Hardware layer

### 8. Increasing the batch size can improve the performance of the training process. However, it can also present which of the following side effects?

A. Reduce the number of samples

B. Reduce the number of operations

C. Reduce the number of training steps

**D. Reduce model accuracy**
