# Chapter 6 - Simplifying the Model

## Quiz Answers

### 1. What are the two steps to take when simplifying a workflow?
A. Reduction and compression.

B. Pruning and reduction.

**C. Pruning and compression.**

D. Reduction and zipping.

### 2. A pruning technique usually has the following dimensions:

**A. Criterion, scope, and method.**

B. Algorithm, scope, and magnitude.

C. Criterion, constraints, and targets.

D. Algorithm, constraints, and targets

### 3. Concerning the compression phase, we can assert that

A. It receives a compressed model as input and verifies the model’s integrity.

B. It receives a compressed model as input and generates a model partially comprised only of the non-pruned parameters.

**C. It receives a pruned model as input and generates a new brain model comprised only of the non-pruned parameters.**

D. It receives a pruned model as input and evaluates the pruning degree applied to that model.

### 4. We can execute the model simplifying process on

A. Pre-trained models only.

B. Pre-trained and non-trained models only.

C. Non-trained models only.

**D. Non-trained, pre-trained, and trained models.**

### 5. What is one of the main goals of simplifying a trained model?

A. Accelerate the training process.

**B. Deploy it on resource-constrained environments.**

C. Improve the model’s accuracy.

D. There is no reason to simplify a trained model.

### 6. Consider the following configuration list passed to the prunner:

`config_list = [{ 'op_types': ['Conv2d'],
                 'exclude_op_names': ['layer2'],
                 'sparse_ratio': 0.25 }]`

Which of the following actions would the prunner take?

A. The pruner will try to nullify 75% of all network parameters.

B. The pruner will try to nullify 25% of the parameters of all fully connected layers.

**C. The pruner will try to nullify 25% of the parameters of convolutional layers, except the one labeled as “layer2”.**

D. The pruner will try to nullify 75% of the parameters of the convolutional layers, except the one labeled as “layer2.

### 7. What is more likely to happen to the model’s accuracy after executing the simplified workflow?

A. The model’s accuracy tends to increase.

B. The model’s accuracy surely increases.

**C. The model’s accuracy tends to reduce.**

D. The model’s accuracy stays the same.

### 8. It is necessary to execute a warmup phase before simplifying

**A. Non-trained models.**

B. Trained models.

C. Pre-trained models.

D. None of the above
