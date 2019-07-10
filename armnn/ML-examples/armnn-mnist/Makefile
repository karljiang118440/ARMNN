ARMNN_LIB = /media/jcq/study/ARMNN/armnn/armnn/build
ARMNN_INC = /media/jcq/study/ARMNN/armnn/armnn/include
BOOST_ROOT=/media/jcq/study/ARMNN/armnn/boost_arm64_install
PROTOBUF=/media/jcq/study/ARMNN/armnn/protobuf-3.5.1_arm64
CXX=/media/jcq/study/tools/CrossTools/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++

all: mnist_tf
#all: mnist_caffe mnist_tf

mnist_caffe: mnist_caffe.cpp mnist_loader.hpp
	$(CXX) -O3 -std=c++14 -I$(ARMNN_INC) -I$(BOOST_ROOT) mnist_caffe.cpp -o mnist_caffe -L$(PROTOBUF)/lib -L$(ARMNN_LIB) -lprotobuf -larmnn -larmnnCaffeParser

mnist_tf: mnist_tf_v1.cpp mnist_loader.hpp
	$(CXX) -O3 -std=c++14 -I$(ARMNN_INC) -I$(BOOST_ROOT)  mnist_tf_v1.cpp -o mnist_tf_v1 -L$(PROTOBUF)/lib -L$(ARMNN_LIB) -lprotobuf -larmnn -larmnnTfParser -lpthread

clean:
	-rm -f mnist_tf mnist_caffe

test: mnist_caffe mnist_tf
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(ARMNN_LIB) ./mnist_caffe
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(ARMNN_LIB) ./mnist_tf_v1
