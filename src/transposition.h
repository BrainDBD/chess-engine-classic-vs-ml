#pragma once
#include "types.h"
#include <cstdint>
#include <vector>

enum TTFlag : uint8_t {
    TT_NONE  = 0,
    TT_EXACT = 1,   // score is exact — raised alpha, didn't cause beta cutoff
    TT_LOWER = 2,   // score is a lower bound — caused beta cutoff, real score >= this
    TT_UPPER = 3,   // score is an upper bound — failed low, real score <= this
};

struct TTEntry {
    uint64_t hash  = 0;
    int      score = 0;
    Move     move  = Move::none();
    int8_t   depth = 0;
    TTFlag   flag  = TT_NONE;
};

class TranspositionTable {
public:
    explicit TranspositionTable(size_t sizeMB = 16) {
        const size_t bytes   = sizeMB * 1024 * 1024;
        const size_t entries = bytes / sizeof(TTEntry);
        // Round down to power of 2 so index masking works
        size_ = 1;
        while (size_ * 2 <= entries) size_ *= 2;
        table_.resize(size_);
    }

    void clear() { table_.assign(size_, TTEntry{}); }

    const TTEntry* probe(uint64_t hash) const {
        const TTEntry& entry = table_[hash & (size_ - 1)];
        return (entry.flag != TT_NONE && entry.hash == hash) ? &entry : nullptr;
    }

    void store(uint64_t hash, int score, Move move, int depth, TTFlag flag) {
        TTEntry& entry = table_[hash & (size_ - 1)];
        // Replace if: new position, or searching deeper than what's stored
        if (entry.hash != hash || depth >= entry.depth)
            entry = { hash, score, move, static_cast<int8_t>(depth), flag };
    }

private:
    std::vector<TTEntry> table_;
    size_t size_;
};
