CXX      = g++
CC       = gcc
CXXFLAGS = -std=c++17 -O2 -Wall -Isrc -Isrc/core -Isrc/net -Isrc/syzygy
CFLAGS   = -std=c11   -O2 -Isrc/syzygy        # C11 for stdatomic.h; no -Wall to mute Fathom's noise

# C++ engine sources
SRCS = src/main.cpp src/uci.cpp \
       src/core/board.cpp src/core/zobrist.cpp src/core/attacks.cpp \
       src/core/movegen.cpp src/core/eval.cpp src/core/search.cpp \
       src/syzygy/syzygy.cpp
# C sources (Fathom) — compiled as C, NOT C++
CSRCS = src/syzygy/tbprobe.c

OBJS  = $(SRCS:.cpp=.o)
COBJS = $(CSRCS:.c=.o)

HDRS = $(wildcard src/*.h) $(wildcard src/core/*.h) \
       $(wildcard src/net/*.h) $(wildcard src/syzygy/*.h)

TARGET_ROOT = chess.exe
TARGET_FAST = ../fast_chess/chess.exe
TARGETS = $(TARGET_ROOT) $(TARGET_FAST)

all: $(TARGETS)

$(TARGET_ROOT): $(OBJS) $(COBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(COBJS) -o $(TARGET_ROOT)

$(TARGET_FAST): $(TARGET_ROOT)
	copy /Y $(subst /,\,$(TARGET_ROOT)) $(subst /,\,$(TARGET_FAST))

# C++ objects
%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# C objects (Fathom) — note $(CC) and $(CFLAGS), compiled as C11
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	del /f $(subst /,\,$(TARGET_ROOT))
	del /f $(subst /,\,$(TARGET_FAST))
	del /f $(subst /,\,$(OBJS)) $(subst /,\,$(COBJS))