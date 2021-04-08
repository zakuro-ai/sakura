docker run --rm \
  --gpus all \
  --shm-size=70g \
  -v $(pwd)/data:/workspace/data -v $(pwd)/demo.py:/workspace/demo.py -it zakuroai/sakura python demo.py
