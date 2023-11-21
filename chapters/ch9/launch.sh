SIF_IMAGE=$1
TRAINING_SCRIPT=$2
BACKEND=$3
NNODES="1"
NPROC_PER_NODE="2"
MASTER_ADDR="localhost"
TORCHRUN_COMMAND="torchrun --nnodes $NNODES --nproc-per-node $NPROC_PER_NODE --master-addr $MASTER_ADDR $TRAINING_SCRIPT --backend $BACKEND"

apptainer exec $SIF_IMAGE $TORCHRUN_COMMAND
