TRAINING_SCRIPT=$1
NGPU=$2
TORCHRUN_COMMAND="torchrun --nnodes 1 --nproc-per-node $NGPU --master-addr localhost $TRAINING_SCRIPT"

$TORCHRUN_COMMAND