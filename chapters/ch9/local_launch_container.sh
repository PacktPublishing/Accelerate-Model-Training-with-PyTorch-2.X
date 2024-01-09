TRAINING_SCRIPT=$1 
NPROC_PER_NODE=$2 
SIF_IMAGE=$3 
NNODES= "1" 
MASTER_ADDR= "localhost" 
TORCHRUN_COMMAND="torchrun --nnodes $NNODES --nproc-per-node $NPROC_PER_NODE --master-addr $MASTER_ADDR $TRAINING_SCRIPT" 

apptainer exec $SIF_IMAGE $TORCHRUN_COMMAND 
