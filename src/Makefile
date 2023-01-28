CXX = g++
CFLAGS = -g -Wall

modules := Neuron InputGate ForgetGate OutputGate
obj := $(modules).o
src := $(objs).cpp
headers := $(obj).h

test:test.cpp $(obj)
	$(CXX) $(CFLAGS) -c $(src) -o $@
