CXX = g++
CFLAGS = -g -Wall
.PHONY:clean

# Files
SRC_FILES = Neuron.cpp ForgetGate.cpp InputGate.cpp OutputGate.cpp
OBJ_FILES = Neuron.o ForgetGate.o InputGate.o OutputGate.o

# Valgrind
VALGRND_FLAGS = --leak-check=yes --show-leak-kinds=all --track-origins=yes

# Default Target using implicit compilation
all: $(OBJ_FILES) $(SRC_FILES)

$(OBJ_FILES): %.o: %.cpp

# Test file
test: test.cpp $(OBJ_FILES)
	$(CXX) $(CFLAGS) -o $@ $< $(SRC_FILES)

clean :
	rm -f *.o