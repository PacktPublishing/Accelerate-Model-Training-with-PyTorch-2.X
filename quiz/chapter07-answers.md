# Chapter 7 - Adopting Mixed Precision

## Quiz Answers

### 1. Which of the following numeric formats represents integers by using only 8 bits?

A. FP8.

B. INT32.

**C. INT8.**

D. INTFB8.

### 2. FP16 is a numeric representation that uses 16 bits to represent floating-point numbers. What is this numeric format also known as?

**A. Half-precision floating-point representation.**

B. Single-precision floating-point representation.

C. Double-precision floating-point representation.

D. OneQuarter-precision floating-point representation.

### 3. Which of the following is a numeric representation for floating-point numbers created by Google to attend to machine learning and artificial intelligence workloads?

A. GP16.

B. GFP16.

C. FP16.

**D. BFP16.**

### 4. NVIDIA created the TF32 data representation. Which of the following number of bits does it use to represent floating-point numbers?

A. 32 bits.

**B. 19 bits.**

C. 16 bits.

D. 20 bits.

### 5. What is the default numeric representation that’s used by PyTorch to execute the operations for the training process?

**A. FP32.**

B. FP8.

C. FP64.

D. INT32.

### 6. What is the goal of the mixed precision approach?

**A. Mixed precision tries to adopt lower-precision formats during the training process’ execution.**

B. Mixed precision tries to adopt higher-precision formats during the training process’ execution.

C. Mixed precision avoids the usage of lower-precision formats during the training process’ execution.

D. Mixed precision avoids the usage of higher-precision formats during the training process’ execution.

### 7. What are the main advantages of using an AMP approach rather than a manual implementation?

A. Simple usage and reduction of performance improvement.

B. Simple usage and reduction of power consumption.

C. Complex usage and avoidance of errors involving numeric representation.

**D. Simple usage and avoidance of errors involving numeric representation.**

### 8. Besides the lack of lower-precision operations, which of the following options is another reason to not use a pure lower-precision approach in the training process?

A. Low performance improvement.

B. High energy consumption.

**C. Loss of information on the gradient.**

D. High usage of main memory.
