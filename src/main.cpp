#include "board.h"
#include "zobrist.h"
#include "attacks.h"
#include <iostream>

int main() {
    Zobrist::init();
    Attacks::init();
    Board b;
    std::cout << b.toFEN() << '\n';
    return 0;
}