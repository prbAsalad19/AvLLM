#include "dataset.hpp"

#include <cstdio>
#include <cstdlib>

bool Dataset::load_binary(const std::string& path) {
    // Apre il file in lettura binaria ("rb")
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "Errore: impossibile aprire il dataset '%s'\n", path.c_str());
        return false;
    }

    // Sposta il cursore alla fine per sapere quanti byte pesa il file
    if (std::fseek(fp, 0, SEEK_END) != 0) {
        std::fclose(fp);
        return false;
    }
    long file_bytes = std::ftell(fp);
    if (file_bytes <= 0) {
        std::fprintf(stderr, "Errore: il file '%s' è vuoto o non valido\n", path.c_str());
        std::fclose(fp);
        return false;
    }
    std::rewind(fp);

    // Ogni token è un uint16_t (2 byte)
    std::size_t count = static_cast<std::size_t>(file_bytes) / sizeof(std::uint16_t);
    if (count < 2) {
        std::fprintf(stderr, "Errore: servono almeno 2 token in '%s'\n", path.c_str());
        std::fclose(fp);
        return false;
    }

    tokens_.resize(count);
    std::size_t read_items = std::fread(tokens_.data(), sizeof(std::uint16_t), count, fp);
    std::fclose(fp);

    if (read_items != count) {
        std::fprintf(stderr, "Errore: letti %zu token su %zu attesi da '%s'\n",
                     read_items, count, path.c_str());
        tokens_.clear();
        return false;
    }

    return true;
}

std::uint64_t Dataset::next_u64() {
    // Algoritmo xorshift64*: molto veloce per scegliere indici di partenza dei batch
    std::uint64_t x = rng_state_;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state_ = x;
    return x * 0x2545F4914F6CDD1DULL;
}

void Dataset::sample_batch(std::size_t batch_size,
                         std::size_t seq_len,
                         std::vector<int>& inputs,
                         std::vector<int>& targets) {
    // Ridimensiona i vettori di output se necessario
    const std::size_t total = batch_size * seq_len;
    if (inputs.size() != total) inputs.resize(total);
    if (targets.size() != total) targets.resize(total);

    // Ci serve almeno seq_len + 1 token perché l'ultimo target legge tokens_[idx + seq_len]
    const std::size_t max_start = tokens_.size() - seq_len - 1;

    for (std::size_t b = 0; b < batch_size; ++b) {
        std::size_t start = static_cast<std::size_t>(next_u64() % (max_start + 1));
        std::size_t base_out = b * seq_len;

        for (std::size_t t = 0; t < seq_len; ++t) {
            inputs[base_out + t]  = static_cast<int>(tokens_[start + t]);
            targets[base_out + t] = static_cast<int>(tokens_[start + t + 1]);
        }
    }
}