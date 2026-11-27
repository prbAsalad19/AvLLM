// Usage:
// ./build/prepare_bpe <latin.txt> <italian.txt> <tokenizer.bpe> <train.bin> <val.bin> [vocab_size]
// The split is performed by lines before tokenizer training: validation lines never
// contribute to BPE training. The same tokenizer is then used for both binary sets.
#include "bpe_tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {
struct Split { std::string train, val; };

bool read_file(const std::string& path, std::string& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return !out.empty() && !in.bad();
}

Split split_lines(const std::string& text, double val_fraction = 0.05) {
    std::vector<std::string> lines;
    std::size_t begin = 0;
    while (begin < text.size()) {
        std::size_t end = text.find('\n', begin);
        if (end == std::string::npos) end = text.size(); else ++end;
        if (end > begin) lines.emplace_back(text.substr(begin, end - begin));
        begin = end;
    }
    Split s;
    if (lines.size() < 2) { s.train = text; return s; }
    const std::size_t val_count = std::max<std::size_t>(1, static_cast<std::size_t>(lines.size() * val_fraction));
    // Deterministic interleaved split so a long book does not put all validation at its end.
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (((i + 1) * val_count) / lines.size() != (i * val_count) / lines.size()) s.val += lines[i];
        else s.train += lines[i];
    }
    if (s.train.empty() || s.val.empty()) { s.train = text; s.val.clear(); }
    return s;
}

bool write_tokens(const std::string& path, const BpeTokenizer& tok, const std::string& text) {
    const std::vector<int> ids = tok.encode(text);
    std::vector<std::uint16_t> out;
    out.reserve(ids.size());
    for (int id : ids) {
        if (id < 0 || id > 65535) return false;
        out.push_back(static_cast<std::uint16_t>(id));
    }
    std::FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    const bool ok = out.empty() || std::fwrite(out.data(), sizeof(std::uint16_t), out.size(), fp) == out.size();
    const bool close_ok = std::fclose(fp) == 0;
    std::printf("%s: %zu token (%zu byte)\n", path.c_str(), out.size(), out.size() * sizeof(std::uint16_t));
    return ok && close_ok;
}
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "Uso: %s <latin.txt> <italian.txt> <tokenizer.bpe> <train.bin> <val.bin> [vocab_size]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int vocab_size = argc >= 7 ? std::atoi(argv[6]) : 2048;
    if (vocab_size < 256 || vocab_size > 65535) {
        std::fprintf(stderr, "vocab_size deve essere tra 256 e 65535\n"); return EXIT_FAILURE;
    }
    std::string latin, italian;
    if (!read_file(argv[1], latin)) { std::fprintf(stderr, "Impossibile leggere %s\n", argv[1]); return EXIT_FAILURE; }
    if (!read_file(argv[2], italian)) { std::fprintf(stderr, "Impossibile leggere %s\n", argv[2]); return EXIT_FAILURE; }

    Split la = split_lines(latin), it = split_lines(italian);
    std::string train_text = la.train + "\n\n" + it.train;
    std::string val_text = la.val + "\n\n" + it.val;
    std::printf("Corpus train: %zu byte (LA %zu, IT %zu); validation: %zu byte\n",
                train_text.size(), la.train.size(), it.train.size(), val_text.size());

    BpeTokenizer tok;
    std::printf("Training BPE: vocab target %d\n", vocab_size);
    if (!tok.train_from_text(train_text, vocab_size, true)) {
        std::fprintf(stderr, "Training BPE fallito\n"); return EXIT_FAILURE;
    }
    if (!tok.save(argv[3])) { std::fprintf(stderr, "Impossibile salvare tokenizer %s\n", argv[3]); return EXIT_FAILURE; }
    std::printf("Tokenizer salvato: %s (vocab=%d)\n", argv[3], tok.vocab_size());
    if (!write_tokens(argv[4], tok, train_text)) { std::fprintf(stderr, "Scrittura train.bin fallita\n"); return EXIT_FAILURE; }
    if (val_text.empty()) {
        std::fprintf(stderr, "ATTENZIONE: validation vuota; controlla che i corpus contengano più righe.\n");
        return EXIT_FAILURE;
    }
    if (!write_tokens(argv[5], tok, val_text)) { std::fprintf(stderr, "Scrittura val.bin fallita\n"); return EXIT_FAILURE; }
    return EXIT_SUCCESS;
}
