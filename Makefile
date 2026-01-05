CXX = g++
CXXFLAGS = -std=c++11 -I./include -I/Users/sjh/miniforge3/envs/noetic/include/eigen3 -I/Users/sjh/miniforge3/envs/noetic/include
LDFLAGS = -lpcl_common -lpcl_io -lpcl_kdtree -lpcl_visualization -lceres -lpthread

SRC = src/loam_ceres_demo.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = mini_loam_demo

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)