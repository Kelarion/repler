#!/bin/bash
module load openmind/singularity/3.2.0        # load singularity module
singularity build /om2/user/malleman/everything.simg docker://petronetto/docker-python-deep-learning
echo "done"
