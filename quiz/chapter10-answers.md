# Chapter 10 - Training with Multiple GPUs

## Quiz answers

### 1. Which are the three main types of GPU interconnection technologies?

A. PCI Express, NCCL, and GPU-Link.

B. PCI Express, NVLink, and NVSwitch.

C. PCI Express, NCCL, and GPU-Switch.

D. PCI Express, NVML, and NVLink.

### 2. NVLink is a proprietary interconnection technology that allows you to do which of the following?

A. Connect the GPU to the CPU.

B. Connect the GPU to the main memory.

C. Connect pairs of GPUs directly to each other.

D. Connect the GPU to the network adapter.

### 3. Which environment variable is used to define GPU affinity?

A. CUDA_VISIBLE_DEVICES.

B. GPU_VISIBLE_DEVICES.

C. GPU_ACTIVE_DEVICES.

D. CUDA_AFFINITY_DEVICES.

### 4. What is NCCL?

A. NCCL is an interconnection technology that’s used to link NVIDIA GPUs.

B. NCCL is a library that’s used to profile programs running on NVIDIA GPUs.

C. NCCL is a compiler toolkit that’s used to generate optimized code for NVIDIA GPUs.

D. NCCL is a library that provides optimized collective operations for NVIDIA GPUs.

### 5. Which program launcher can be used to run distributed training on multiple GPUs?

A. GPUrun.

B. Torchrun.

C. NCCLrun.

D. OneCCL.

### 6. If we set the CUDA_VISIBLE_DEVICES environment variable to a value of “2,3”, how many device numbers will be passed to the training script?

A. 2 and 3.

B. 3 and 2.

C. 0 and 1.

D. 0 and 7.

### 7. How can we obtain more information about the interconnection topology that’s adopted in a given multi-GPU environment?

A. Running the nvidia-topo-ls command with the -interconnection option.

B. Running the nvidia-topo-ls command with the -gpus option.

C. Running the nvidia-smi command with the -interconnect option.

D. Running the nvidia-smi command with the -topo option.

### 8. Which component is used by the PCI Express technology to interconnect PCI Express devices in a computing system?

A. PCIe switch.

B. PCIe nvswitch

C. PCIe link.

D. PCIe network
