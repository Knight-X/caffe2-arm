EIGEN3_INCLUDE = /usr/local/include/eigen3/
CAFFE2_LIB = /usr/local/lib/

CAFFE2_LINKFLAGS = -L${CAFFE2_LIB} -lCaffe2_CPU -lglog -lprotobuf -lgflags

TARGET = intro

SOURCE_FILES = src/intro.cc

CXXFLAGS = -std=c++11 -I${EIGEN3_INCLUDE} 

all: ${TARGET}

${TARGET} : src/intro.o
	$(CXX) $< $(CAFFE2_LINKFLAGS) -o $@
clean:
	rm *.o
	rm *.so
	rm ${TARGET}

src/intro.o : src/intro.cc
	$(CXX) src/intro.cc $(CXXFLAGS)  -c -o src/intro.o



object:
	g++ -g -fPIC -std=c++11 -mfpu=neon -march=armv7-a -mthumb -mfloat-abi=hard -I/usr/local/include -I/usr/local/include/eigen3 -I../ComputeLibrary -c operators/softmax.cc operators/maxpool.cc operators/conv.cc operators/relu.cc operators/fc.cc
softmax:
	g++ -g -fPIC -std=c++11 -mfpu=neon -march=armv7-a -mthumb -mfloat-abi=hard -I/usr/local/include -I/usr/local/include/eigen3 -I../ComputeLibrary -c operators/softmax.cc operators/maxpool.cc operators/conv.cc operators/relu.cc operators/fc.cc

maxpool:
	g++ -g -fPIC -std=c++11 -mfpu=neon -march=armv7-a -mthumb -mfloat-abi=hard -I/usr/local/include -I/usr/local/include/eigen3 -I../ComputeLibrary -c operators/maxpool.cc
conv:
	g++ -g -fPIC -std=c++11 -mfpu=neon -march=armv7-a -mthumb -mfloat-abi=hard -I/usr/local/include -I/usr/local/include/eigen3 -I../ComputeLibrary -c operators/conv.cc
relu:
	g++ -g -fPIC -std=c++11 -mfpu=neon -march=armv7-a -mthumb -mfloat-abi=hard -I/usr/local/include -I/usr/local/include/eigen3 -I../ComputeLibrary -c operators/relu.cc
fc:
	g++ -g -fPIC -std=c++11 -mfpu=neon -march=armv7-a -mthumb -mfloat-abi=hard -I/usr/local/include -I/usr/local/include/eigen3 -I../ComputeLibrary -c operators/fc.cc

shared:
	g++ -shared -o arm.so conv.o fc.o maxpool.o softmax.o relu.o -lCaffe2_CPU -lprotobuf -lgflags -lglog -larm_compute
