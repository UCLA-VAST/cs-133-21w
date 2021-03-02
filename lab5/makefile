SRCS = lib/fixed-point/cnn.h lib/fixed-point/cnn.cpp \
	   lib/fixed-point/opencl.cpp lib/fixed-point/main.cpp
FAST = lib/cnn.h lib/cnn.cpp lib/opencl.cpp lib/main.cpp

DEVICE := $(AWS_PLATFORM)
DSA := $(basename $(notdir $(DEVICE)))
VP := $(XILINX_VITIS)/bin/v++
CLFLAGS = -t $(TARGET) --platform $(DEVICE) --save-temps
LDFLAGS += -lOpenCL

export OPENCL_PLATFORM = Xilinx
export OPENCL_DEVICE = $(DSA)

REQUIRED_FILES = cnn.cl params.sh lab5-report.pdf

fastsim: emconfig
	OPENCL_BINARY=cnn_fast.sw_emu.$(DSA).xclbin XCL_EMULATION_MODE=sw_emu $(MAKE) TARGET=sw_emu internal-fast

swsim: emconfig
	OPENCL_BINARY=cnn.sw_emu.$(DSA).xclbin XCL_EMULATION_MODE=sw_emu $(MAKE) TARGET=sw_emu internal

hwsim: emconfig
	OPENCL_BINARY=cnn.hw_emu.$(DSA).xclbin XCL_EMULATION_MODE=hw_emu $(MAKE) TARGET=hw_emu internal

estimate:
	$(MAKE) TARGET=hw cnn.hw.$(DSA).xo
	grep -m 1 -B 1 -A 4 "Latency (absolute)" _x/*/CnnKernel/CnnKernel/solution/syn/report/CnnKernel_YourCode_csynth.rpt

resource:
	$(MAKE) TARGET=hw cnn.hw.$(DSA).xo
	grep -m 1 -B 1 -A 24 "== Utilization Estimates" _x/*/CnnKernel/CnnKernel/solution/syn/report/CnnKernel_YourCode_csynth.rpt

cnn.$(TARGET).$(DSA).xo: ./cnn-krnl.cpp
	$(VP) $(CLFLAGS) -c -k CnnKernel -I'$(<D)' -o'$@' '$<'

cnn.$(TARGET).$(DSA).xclbin: cnn.$(TARGET).$(DSA).xo
	$(VP) $(CLFLAGS) -l $(LDCLFLAGS) --nk CnnKernel:1 -o'$@' $(+)

cnn_fast.$(TARGET).$(DSA).xo: ./cnn-krnl.cpp
	$(VP) $(CLFLAGS) -c -k CnnKernel -I'$(<D)' -DFASTSIM -o'$@' '$<'

cnn_fast.$(TARGET).$(DSA).xclbin: cnn_fast.$(TARGET).$(DSA).xo
	$(VP) $(CLFLAGS) -l $(LDCLFLAGS) --nk CnnKernel:1 -o'$@' $(+)

emconfig: emconfig.json

emconfig.json:
	emconfigutil --platform $(DEVICE) --od .

internal: cnn $(OPENCL_SOURCE) cnn.$(TARGET).$(DSA).xclbin
	OPENCL_WORKGROUP_GLOBAL='1' OPENCL_WORKGROUP_LOCAL='1' ./$<

internal-fast: cnn_fast $(OPENCL_SOURCE) cnn_fast.$(TARGET).$(DSA).xclbin
	OPENCL_WORKGROUP_GLOBAL='1' OPENCL_WORKGROUP_LOCAL='1' ./$<

cnn: $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.a %.o, $^) $(LDFLAGS)

cnn_fast: $(FAST)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.a %.o, $^) $(LDFLAGS)

clean:
	$(RM) cnn cnn.*.xo cnn.*.xclbin cnn.*.xo.* cnn.*.xclbin.* *.log v++* *.json *.xml
	$(RM) cnn_fast cnn_fast.*.xo cnn_fast.*.xclbin cnn_fast.*.xo.* cnn_fast.*.xclbin.*
	$(RM) -r _x

#CXX = # specify your compiler here
LDFLAGS += # specify your library linking options here
CXXFLAGS += -std=c++11 -O3 $(LDFLAGS)
