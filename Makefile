CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Isrc
SRCS     = src/main.cpp src/board.cpp src/zobrist.cpp
TARGET   = chess

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	del /f $(TARGET).exe