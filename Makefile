CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Isrc -Isrc/core -Isrc/net -Isrc/syzygy
SRCS = src/main.cpp src/uci.cpp \
       src/core/board.cpp src/core/zobrist.cpp src/core/attacks.cpp \
       src/core/movegen.cpp src/core/eval.cpp src/core/search.cpp \
       src/syzygy/syzygy.cpp
CSRCS = src/syzygy/tbprobe.c src/syzygy/tbchess.c
HDRS = $(wildcard src/*.h) $(wildcard src/core/*.h) \
       $(wildcard src/net/*.h) $(wildcard src/syzygy/*.h)

TARGET = chess

$(TARGET): $(SRCS) $(CSRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) $(SRCS) $(CSRCS) -o $(TARGET)

clean:
	del /f $(TARGET).exe