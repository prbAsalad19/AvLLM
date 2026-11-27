#include "bpe_tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <unordered_map>

namespace {

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const noexcept {
        return (static_cast<std::size_t>(static_cast<uint32_t>(p.first)) << 32) ^
               static_cast<std::size_t>(static_cast<uint32_t>(p.second));
    }
};

} // namespace

BpeTokenizer::BpeTokenizer() {
    id_to_bytes_.resize(256);
    for (int i = 0; i < 256; ++i) {
        id_to_bytes_[i] = std::string(1, static_cast<char>(static_cast<unsigned char>(i)));
    }
}

bool BpeTokenizer::train_from_text(const std::string& text, int target_vocab_size, bool verbose) {
    if (text.empty() || target_vocab_size < 256) {
        return false;
    }

    id_to_bytes_.resize(256);
    for (int i = 0; i < 256; ++i) {
        id_to_bytes_[i] = std::string(1, static_cast<char>(static_cast<unsigned char>(i)));
    }
    merges_.clear();

    // Ogni riga è una sequenza indipendente: una fusione BPE non può attraversare
    // il confine tra due righe/documenti. Il carattere newline viene mantenuto come
    // token byte nella propria sequenza, così il tokenizer può comunque codificarlo.
    std::vector<std::vector<int>> sequences;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t found = text.find('\n', begin);
        const std::size_t end = (found == std::string::npos) ? text.size() : found + 1;
        std::vector<int> seq;
        seq.reserve(end - begin);
        for (std::size_t i = begin; i < end; ++i) {
            seq.push_back(static_cast<int>(static_cast<unsigned char>(text[i])));
        }
        if (!seq.empty()) sequences.push_back(std::move(seq));
        begin = end;
    }

    while (vocab_size() < target_vocab_size) {
        std::unordered_map<std::pair<int, int>, std::size_t, PairHash> counts;
        for (const auto& seq : sequences) {
            for (std::size_t i = 0; i + 1 < seq.size(); ++i) {
                counts[{seq[i], seq[i + 1]}] += 1;
            }
        }

        if (counts.empty()) {
            break;
        }

        std::pair<int, int> best_pair{0, 0};
        std::size_t best_count = 0;
        for (const auto& kv : counts) {
            if (kv.second > best_count) {
                best_count = kv.second;
                best_pair = kv.first;
            }
        }

        if (best_count < 2) {
            break;
        }

        const int new_id = vocab_size();
        id_to_bytes_.push_back(id_to_bytes_[best_pair.first] + id_to_bytes_[best_pair.second]);
        merges_.push_back(BpeMergeRule{best_pair.first, best_pair.second, new_id});

        // Applica la fusione a tutte le sequenze
        for (auto& seq : sequences) {
            if (seq.size() < 2) continue;
            std::vector<int> next_seq;
            next_seq.reserve(seq.size());
            std::size_t i = 0;
            while (i < seq.size()) {
                if (i + 1 < seq.size() &&
                    seq[i] == best_pair.first &&
                    seq[i + 1] == best_pair.second) {
                    next_seq.push_back(new_id);
                    i += 2;
                } else {
                    next_seq.push_back(seq[i]);
                    ++i;
                }
            }
            seq.swap(next_seq);
        }

        if (verbose && (merges_.size() % 100 == 0 || vocab_size() == target_vocab_size)) {
            std::printf("BPE vocab: %d / %d (ultima fusione count=%zu)\n",
                        vocab_size(), target_vocab_size, best_count);
        }
    }

    return true;
}

std::vector<int> BpeTokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    tokens.reserve(text.size());
    for (unsigned char ch : text) {
        tokens.push_back(static_cast<int>(ch));
    }

    // Applica le regole nell'ordine esatto in cui sono state apprese
    for (const auto& rule : merges_) {
        if (tokens.size() < 2) break;
        std::vector<int> next_tokens;
        next_tokens.reserve(tokens.size());
        std::size_t i = 0;
        while (i < tokens.size()) {
            if (i + 1 < tokens.size() &&
                tokens[i] == rule.left &&
                tokens[i + 1] == rule.right) {
                next_tokens.push_back(rule.merged_id);
                i += 2;
            } else {
                next_tokens.push_back(tokens[i]);
                ++i;
            }
        }
        tokens.swap(next_tokens);
    }

    return tokens;
}

std::string BpeTokenizer::decode(const std::vector<int>& tokens) const {
    std::string out;
    for (int id : tokens) {
        if (id >= 0 && id < vocab_size()) {
            out += id_to_bytes_[static_cast<std::size_t>(id)];
        }
    }
    return out;
}

bool BpeTokenizer::save(const std::string& path) const {
    std::FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;

    const uint32_t magic = 0x42504531; // "BPE1"
    const uint32_t n_merges = static_cast<uint32_t>(merges_.size());

    bool ok = (std::fwrite(&magic, sizeof(magic), 1, fp) == 1) &&
              (std::fwrite(&n_merges, sizeof(n_merges), 1, fp) == 1);

    for (const auto& rule : merges_) {
        ok = ok && (std::fwrite(&rule.left, sizeof(rule.left), 1, fp) == 1) &&
                   (std::fwrite(&rule.right, sizeof(rule.right), 1, fp) == 1) &&
                   (std::fwrite(&rule.merged_id, sizeof(rule.merged_id), 1, fp) == 1);
    }

    std::fclose(fp);
    return ok;
}

bool BpeTokenizer::load(const std::string& path) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;

    uint32_t magic = 0;
    uint32_t n_merges = 0;
    if (std::fread(&magic, sizeof(magic), 1, fp) != 1 ||
        std::fread(&n_merges, sizeof(n_merges), 1, fp) != 1 ||
        magic != 0x42504531) {
        std::fclose(fp);
        return false;
    }

    id_to_bytes_.resize(256);
    for (int i = 0; i < 256; ++i) {
        id_to_bytes_[i] = std::string(1, static_cast<char>(static_cast<unsigned char>(i)));
    }
    merges_.clear();
    merges_.reserve(n_merges);

    for (uint32_t i = 0; i < n_merges; ++i) {
        BpeMergeRule rule{};
        if (std::fread(&rule.left, sizeof(rule.left), 1, fp) != 1 ||
            std::fread(&rule.right, sizeof(rule.right), 1, fp) != 1 ||
            std::fread(&rule.merged_id, sizeof(rule.merged_id), 1, fp) != 1) {
            std::fclose(fp);
            return false;
        }
        if (rule.merged_id != 256 + static_cast<int>(i) ||
            rule.left < 0 || rule.left >= static_cast<int>(id_to_bytes_.size()) ||
            rule.right < 0 || rule.right >= static_cast<int>(id_to_bytes_.size())) {
            std::fclose(fp);
            return false;
        }
        id_to_bytes_.push_back(id_to_bytes_[rule.left] + id_to_bytes_[rule.right]);
        merges_.push_back(rule);
    }

    std::fclose(fp);
    return true;
}