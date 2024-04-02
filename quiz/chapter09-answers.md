# Chapter 9 - Training with Multiple CPUs

## Quiz answers

### 1. In multicore systems, we can improve the performance of the training process by increasing the number of threads used by PyTorch. Concerning this topic, we can affirm which of the following?

**A. After crossing a certain number of threads, the performance improvement can deteriorate or stay the same.**

B. The performance improvement always keeps rising, no matter the number of threads.

C. There is no performance improvement when increasing the number of threads.

D. Performance improvement is only achieved when using 16 threads.

### 2. Which is the most basic communication backend supported by Pytorch?

A. NNI.

**B. Gloo**

C. MPI.

D. TorchInductor

### 3. Which is the default program launcher provided by PyTorch?

A. PyTorchrun.

B. Gloorun.

C. MPIRun.

**D. Torchrun.**

### 4. In the context of PyTorch, what is Intel OneCCL?

**A. Communication backend.**

B. Program launcher.

C. Checkpointing automation tool.

D. Profiling tool.

### 5. When considering a non-Intel environment, what would be the most reasonable choice for the communication backend?

A. Gloorun.

B. Torchrun.

C. OneCCL.

**D. Gloo.**

### 6. Concerning the performance of the training process when using Gloo or OneCCL as a communication backend, we can say which of the following?

A. There is no difference at all.

B. Gloo is always better than OneCCL.

**C. OneCCL can overcome Gloo in Intel platforms.**

D. OneCCL is always better than Gloo.

### 7. When distributing the training process among multiple CPUs and cores, we need to define the allocation of threads in order to do which of the following?

**A. Guarantee all threads have exclusive usage of computing resources.**

B. Guarantee secure execution.

C. Guarantee protected execution.

D. Guarantee that data are shared among all threads.

### 8. What are the two main tasks of torchrun?

A. Create a pool of shared memory and instantiate the processes in the operating system.

**B. Define the environment variables related to the distributed environment and instantiate the processes on the operating system.**

C. Define the environment variables related to the distributed environment and create a pool of shared memory.

D. Identify the best number of threads to run with PyTorch.
