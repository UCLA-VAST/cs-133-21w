SRCS = lib/cnn.h lib/cnn.cpp lib/opencl.cpp lib/main.cpp

DSA := xilinx_aws-vu9p-f1-04261818_dynamic_5_0
DEVICE := $(AWS_PLATFORM)
VP := $(XILINX_VITIS)/bin/v++
CLFLAGS = -t $(TARGET) --platform $(DEVICE) --save-temps
LDFLAGS += -lOpenCL

export OPENCL_PLATFORM = Xilinx
export OPENCL_DEVICE = $(DSA)
export OPENCL_BINARY = cnn.$(TARGET).$(DSA).xclbin

REQUIRED_FILES = cnn.cl params.sh lab4-report.pdf

swsim: emconfig
	XCL_EMULATION_MODE=sw_emu $(MAKE) TARGET=sw_emu internal

hwsim: emconfig
	XCL_EMULATION_MODE=hw_emu $(MAKE) TARGET=hw_emu internal

estimate:
	$(MAKE) TARGET=hw cnn.hw.$(DSA).xo

cnn.$(TARGET).$(DSA).xo: ./cnn.cpp
	$(VP) $(CLFLAGS) -c -k CnnKernel -I'$(<D)' -o'$@' '$<'

cnn.$(TARGET).$(DSA).xclbin: cnn.$(TARGET).$(DSA).xo
	$(VP) $(CLFLAGS) -l $(LDCLFLAGS) --nk CnnKernel:1 -o'$@' $(+)

emconfig: emconfig.json

emconfig.json:
	emconfigutil --platform $(DEVICE) --od .

internal: cnn $(OPENCL_SOURCE) cnn.$(TARGET).$(DSA).xclbin
	OPENCL_WORKGROUP_GLOBAL='1' OPENCL_WORKGROUP_LOCAL='1' ./$<

cnn: $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.a %.o, $^) $(LDFLAGS)

clean:
	$(RM) cnn cnn.*.xo

#CXX = # specify your compiler here
LDFLAGS += # specify your library linking options here
CXXFLAGS += -std=c++17 -O3 $(LDFLAGS)
