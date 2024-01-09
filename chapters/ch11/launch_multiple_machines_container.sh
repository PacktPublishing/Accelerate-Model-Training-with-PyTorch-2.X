TRAINING_SCRIPT=$1
SIF_IMAGE=$2
NPROCS="16"
HOSTS="machine1:8,machine2:8"
COMMAND="python $TRAINING_SCRIPT"

export MASTER_ADDR="machine1"
export MASTER_PORT="12345"

mpirun -x MASTER_ADDR -x MASTER_PORT --np $NPROCS --host $HOSTS apptainer exec --nv $SIF_IMAGE $COMMAND
