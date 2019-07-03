ARMNN_LIB = /media/jcq/study/ARMNN/armnn/armnn/build
ARMNN_INC = /media/jcq/study/ARMNN/armnn/armnn/include
BOOST_ROOT=/media/jcq/study/tools/boost_1_64_0/boost_arm64_install
PROTOBUF=/media/jcq/study/softwares/protobuf3.5.1/protobuf-3.5.1/arm64_build
CXX=/media/jcq/study/tools/CrossTools/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++

all: mnist_tf
#all: mnist_caffe mnist_tf

mnist_caffe: mnist_caffe.cpp mnist_loader.hpp
	$(CXX) -O3 -std=c++14 -I$(ARMNN_INC) -I$(BOOST_ROOT) mnist_caffe.cpp -o mnist_caffe -L$(PROTOBUF)/lib -L$(ARMNN_LIB) -lprotobuf -larmnn -larmnnCaffeParser

mnist_tf: mnist_tf.cpp mnist_loader.hpp
	$(CXX) -O3 -std=c++14 -I$(ARMNN_INC) -I$(BOOST_ROOT)  mnist_tf.cpp -o mnist_tf -L$(PROTOBUF)/lib -L$(ARMNN_LIB) -lprotobuf -larmnn -larmnnTfParser -lpthread

clean:
	-rm -f mnist_tf mnist_caffe

test: mnist_caffe mnist_tf
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(ARMNN_LIB) ./mnist_caffe
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(ARMNN_LIB) ./mnist_tf
