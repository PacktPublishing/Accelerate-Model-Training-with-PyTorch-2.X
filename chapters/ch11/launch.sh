SIF_IMAGE=$1
TRAINING_SCRIPT=$2
BACKEND=$3
NPROCS="16"
HOSTS="machine1:8,machine2:8"
COMMAND="python $TRAINING_SCRIPT --backend $BACKEND"

export MASTER_ADDR="machine1"
export MASTER_PORT="12345"

mpirun -x MASTER_ADDR -x MASTER_PORT --np $NPROCS --host $HOSTS apptainer exec --nv $SIF_IMAGE $COMMAND
