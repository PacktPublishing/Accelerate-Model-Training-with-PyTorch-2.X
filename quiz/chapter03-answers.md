# Chapter 3 - Quiz answers

### 1. Which are the two execution modes of PyTorch?

A. Horizontal and vertical modes

**B. Eager and graph modes**

C. Eager and distributed modes

D. Eager and auto modes

### 2. In which execution mode does PyTorch execute operations as soon as they appear in the code?

A. Graph mode

**B. Eager mode**

C. Distributed mode

D. Auto mode

### 3. In which execution mode does PyTorch evaluate the complete set of operations seeking optimization opportunities?

**A. Graph mode**

B. Eager mode

C. Distributed mode

D. Auto mode

### 4. Compiling a model with PyTorch means changing from eager to graph mode when executing in which of the following phases of the training process?

A. Forward and optimization

B. Forward and loss calculation

**C. Forward and backward**

D. Forward and training

### 5. Concerning the time to execute the first training epoch in both eager and graph modes, what can we assert?

A. The time to execute the first training epoch is always the same in both eager and graph modes

B. The time to execute the first training epoch in graph mode is always smaller than executing 
in the eager mode

**C. The time to execute the first training epoch in graph mode is likely to be higher than 
executing in the eager mode**

D. The time to execute the first training epoch in eager mode is likely to be higher than 
executing in the eager mode

### 6. Which phases comprise the compiling workflow that’s executed by the Compile API?

A. Graph forward, graph backward, and graph compilation

B. Graph acquisition, graph backward, and graph compilation

C. Graph acquisition, graph lowering, and graph optimization

**D. Graph acquisition, graph lowering, and graph compilation**

### 7. TorchDynamo is a component of the Compile API that executes which phase?

A. Graph backward

**B. Graph acquisition**

C. Graph lowering

D. Graph optimization

### 8. TorchInductor is the default compiler backend of PyTorch’s Compile API. Which are the other compiler backends?

A. OpenMP and NCCL

B. OpenMP and Triton

**C. Cudagraphs and IPEX**

D. TorchDynamo and Cudagraphs
