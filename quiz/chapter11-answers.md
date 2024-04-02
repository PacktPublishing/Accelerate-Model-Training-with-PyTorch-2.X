# Chapter 11 - Training with Multiple Machines

## Quiz answers

### 1. What is a task submitted to a computing cluster called?

A. Thread.

B. Process.

**C. Job.**

D. Work

### 2. What are the main tasks executed by a workload manager?

**A. Resource management and job scheduling.**

B. Memory allocation and thread scheduling.

C. GPU management and node scheduling.

D. Resource management and node scheduling.

### 3. Which of the following is an open source, fault-tolerant, and highly scalable workload manager for large and small Linux clusters?

A. MPI.

**B. SLURM.**

C. NCCL.

D. Gloo.

### 4. A computing cluster is usually equipped with a high-performance network such as NVIDIA Infiniband. Besides providing a high bandwidth, a high-performance interconnection provides which of the following?

A. A high latency.

B. A high number of connections.

C. A low number of connections.

**D. A very low latency.**

### 5. RDMA reduces drastically the communication latency between two remote GPUs because it enables which of the following?

A. Allocation of higher memory space on GPUs.

B. Special hardware capabilities on GPUs.

**C. Data transfer without involving the CPU and main memory.**

D. Data transfer without involving network adapters and switches.

### 6. Which of the following is the best definition of Open MPI?

A. Open MPI is a compiler to create distributed applications.

**B. Open MPI is a toolset comprised of compilers, debuggers, and a complete runtime mechanism to create, debug, and run distributed applications.**

C. Open MPI is a standard that specifies a set of communication routines, data types, events, and operations used to implement distributed applications.

D. Open MPI is a communication backend exclusively created to run the distributed training under PyTorch.

### 7. Consider the scenario in which a distributed training is running four processes under two machines (each machine is executing two processes). In this case, what are the ranks assigned by Open MPI for the two processes executing on the second machine?

A. 0 and 1.

B. 0 and 2.

**C. 2 and 3.**

D. 0 and 3

### 8. Concerning the decision to distribute the training process among multiple machines or keep it in a single host, it is reasonable to ponder which of the following?

A. The power consumption of using network adapters.

B. The leak of memory space available on the network adapters.

C. Nothing; it is always recommended to use multiple machines to run the distributed training.

**D. The impact the interconnection network may have on the communication between the processes participating in the distributed training.**
