#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class Dataset {
public:
    Dataset() = default;

    // Carica il file binario di token (es. data/train.bin) in memoria RAM
    bool load_binary(const std::string& path);

    // Estrae un batch casuale:
    // - inputs:  buffer di dimensione [batch_size * seq_len]
    // - targets: buffer di dimensione [batch_size * seq_len] (spostato di 1 token in avanti)
    void sample_batch(std::size_t batch_size,
                      std::size_t seq_len,
                      std::vector<int>& inputs,
                      std::vector<int>& targets);

    std::size_t num_tokens() const { return tokens_.size(); }

private:
    std::vector<std::uint16_t> tokens_;
    std::uint64_t rng_state_ = 1337ULL;

    // Generatore pseudo-casuale veloce e deterministico (xorshift64*)
    std::uint64_t next_u64();
};
