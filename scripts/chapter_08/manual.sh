NNODES="2"
NPROC_PER_NODE="1"
MASTER_ADDR="10.28.166.39"
NODE_RANK=$1
BACKEND="mpi"
TRAINING_SCRIPT="pytorch_ddp.py"
TORCHRUN_COMMAND="torchrun --nnodes $NNODES --nproc-per-node $NPROC_PER_NODE --master-addr $MASTER_ADDR --node-rank $NODE_RANK $TRAINING_SCRIPT --backend $BACKEND"

apptainer exec /tmp/pytorch-distributed_mpi.sif $TORCHRUN_COMMAND
