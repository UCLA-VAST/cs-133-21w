SRCS = lib/cnn.h lib/cnn.cpp lib/opencl.cpp lib/main.cpp

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS += -framework OpenCL
    export OPENCL_PLATFORM ?= Apple
else
    LDFLAGS += -lOpenCL
    export OPENCL_PLATFORM ?= Intel(R) CPU Runtime for OpenCL(TM) Applications
endif

REQUIRED_FILES = cnn.cl params.sh lab3-report.pdf
export OPENCL_SOURCE ?= cnn.cl

test: cnn $(OPENCL_SOURCE)
	. ./params.sh; ./$<

test-seq:
	@$(MAKE) --no-print-directory test SEQUENTIAL=

test-vadd: vadd lib/vadd.cl
	OPENCL_SOURCE=lib/vadd.cl ./$<

cnn: $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.a %.o, $^) $(LDFLAGS)

vadd: lib/opencl.cpp lib/vadd.cpp
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.a %.o, $^) $(LDFLAGS)

clean:
	$(RM) vadd cnn

#CXX = # specify your compiler here
LDFLAGS += # specify your library linking options here
CXXFLAGS += -std=c++17 -O3 $(LDFLAGS)
