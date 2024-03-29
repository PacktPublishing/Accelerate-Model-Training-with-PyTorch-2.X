# Chapter 4 - Quiz Answers

### 1. A multicore system can have the following two types of computing cores:

A. Physical and active.

B. Physical and digital.

**C. Physical and logical.**

D. Physical and vectorial.

### 2. A set of threads created by the same process...

A. May share the same memory address space.

B. Do not share the same memory address space.

C. Is impossible in modern systems.

**D. Do share the same memory address space.**

### 3. Which of the following environment variables can be used to set the number of threads used by OpenMP?

A. OMP_NUM_PROCS

**B. OMP_NUM_THREADS**

C. OMP_NUMBER_OF_THREADS

D. OMP_N_THREADS

### 4. In a multicore system, the usage of OpenMP is able to improve the performance of the training process because it can...

A. Allocate the process to the main memory.

B. Bind threads to logical cores.

**C. Bind threads to physical cores.**

D. Avoid the usage of cache memory.

### 5. Concerning the implementation of OpenMP through Intel and GNU, we can assert that...

A. There is no difference between the performance obtained by both versions.

**B. The Intel version can outperform GNU’s implementation when running on Intel platforms.**

C. The Intel version never outperforms GNU’s implementation when running on Intel platforms.

D. The GNU version is always faster than Intel OpenMP, regardless of the hardware platform.

### 6. IPEX stands for Intel extension for PyTorch and is defined as...

A. A set of low-level hardware instructions.

B. A set of code examples.

**C. A set of libraries and tools.**

D. A set of documents.

### 7. What is the strategy adopted by IPEX to accelerate the training process?

A. IPEX enables the usage of special hardware instructions.

B. IPEX replaces all the training process operations with an optimized version.

C. IPEX fuses all operations of the training process into a monolithic piece of code.

**D. IPEX replaces some of the default PyTorch operations of the training process with its own optimized implementations.**

### 8. What is necessary to change in our original PyTorch code to use IPEX?

A. Nothing at all.

B. We just need to import the IPEX module.

**C. We need to import the IPEX module and wrap the model with the ipex.optimize() method.**

D. We just need to use the newest PyTorch version.
