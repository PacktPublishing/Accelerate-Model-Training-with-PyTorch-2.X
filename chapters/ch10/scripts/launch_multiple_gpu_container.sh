TRAINING_SCRIPT=$1
NGPU=$2
SIF_IMAGE=$3 
TORCHRUN_COMMAND="torchrun --nnodes 1 --nproc-per-node $NGPU --master-addr localhost $TRAINING_SCRIPT"

apptainer exec --nv $SIF_IMAGE $TORCHRUN_COMMAND