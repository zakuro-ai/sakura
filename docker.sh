docker rmi -f zakuroai/sakura
docker build . -t zakuroai/sakura
docker run --rm \
  --gpus all \
  --shm-size=70g \
  -v $(pwd)/data:/workspace/data -v $(pwd)/mnist_demo:/workspace/mnist_demo -it zakuroai/sakura python -m mnist_demo
