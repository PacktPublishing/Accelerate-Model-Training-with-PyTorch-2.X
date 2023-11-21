SIF_IMAGE="/tmp/pytorch-baseline_gpu.sif"
TRAINING_SCRIPT="nccl_distributed-efficientnet_cifar10.py"
NGPU="8"
TORCHRUN_COMMAND="torchrun --nnodes 1 --nproc-per-node $NGPU --master-addr localhost $TRAINING_SCRIPT --backend nccl"

apptainer exec --nv $SIF_IMAGE $TORCHRUN_COMMAND