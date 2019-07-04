#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/root/armnn/build
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/root/armnn/protobuf-3.5.1_arm64/lib
export CPLUS_INCLUDE_PATH=/home/root/armnn/include:$CPLUS_INCLUDE_PATH
chmod a+x build/
chmod a+x home/root/armnn/armnn-mnist/
cd /home/root/armnn/armnn-mnist
