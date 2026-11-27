#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct BpeMergeRule {
    int left = 0;
    int right = 0;
    int merged_id = 0;
};

class BpeTokenizer {
public:
    // Inizializza con vocabolario base a byte (0..255)
    BpeTokenizer();

    // Addestra regole BPE fino a raggiungere `target_vocab_size` (>= 256)
    bool train_from_text(const std::string& text, int target_vocab_size, bool verbose = true);

    // Converte una stringa UTF-8/byte in sequenza di ID token
    std::vector<int> encode(const std::string& text) const;

    // Riconverte una sequenza di ID token in stringa
    std::string decode(const std::vector<int>& tokens) const;

    // Salva e carica il modello BPE su disco
    bool save(const std::string& path) const;
    bool load(const std::string& path);

    int vocab_size() const { return static_cast<int>(id_to_bytes_.size()); }

private:
    std::vector<std::string> id_to_bytes_;
    std::vector<BpeMergeRule> merges_;

    static std::pair<int, int> find_best_pair(
        const std::vector<std::vector<int>>& sequences,
        const std::vector<int>& pair_counts_hint);
};