#!/bin/bash
ARGS=$@
docker run --rm \
-v $SAKURA_HOME/sakura:/opt/miniconda/envs/zak38/lib/python3.8/site-packages/sakura \
-v $PWD:$PWD \
-v $ZAKURO_LOGS:/opt/zakuro/logs \
-e PYENV=zak38 \
--gpus all \
-it zakuroai/sakura bash -c "cd $PWD && sakura $ARGS"