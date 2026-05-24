#include "board.h"
#include "zobrist.h"
#include <iostream>

int main() {
    Board b;
    Zobrist::init();
    std::cout<<b.toFEN()<<std::endl;
    return 0;
}