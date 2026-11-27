// Compilazione:
// g++ -std=c++17 -O3 src/prepare_data.cpp -o build/prepare_data
// Uso:
// ./build/prepare_data data/raw/input.txt data/train.bin data/val.bin

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Uso: %s <input_txt> <output_train_bin> [output_val_bin]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const std::string input_path = argv[1];
    const std::string train_path = argv[2];
    const std::string val_path   = (argc >= 4) ? argv[3] : "";

    std::FILE* in_fp = std::fopen(input_path.c_str(), "rb");
    if (!in_fp) {
        std::fprintf(stderr, "Errore: impossibile aprire '%s'\n", input_path.c_str());
        return EXIT_FAILURE;
    }

    if (std::fseek(in_fp, 0, SEEK_END) != 0) {
        std::fclose(in_fp);
        return EXIT_FAILURE;
    }
    long raw_size = std::ftell(in_fp);
    if (raw_size <= 0) {
        std::fprintf(stderr, "Errore: file '%s' vuoto o non valido\n", input_path.c_str());
        std::fclose(in_fp);
        return EXIT_FAILURE;
    }
    std::rewind(in_fp);

    std::vector<unsigned char> raw_bytes(static_cast<std::size_t>(raw_size));
    std::size_t read_n = std::fread(raw_bytes.data(), 1, raw_bytes.size(), in_fp);
    std::fclose(in_fp);

    if (read_n != raw_bytes.size()) {
        std::fprintf(stderr, "Errore durante la lettura di '%s'\n", input_path.c_str());
        return EXIT_FAILURE;
    }

    // Converte ogni byte (0..255) in uint16_t per combaciare con Dataset::load_binary
    std::vector<std::uint16_t> tokens;
    tokens.reserve(raw_bytes.size());
    for (unsigned char b : raw_bytes) {
        tokens.push_back(static_cast<std::uint16_t>(b));
    }

    // Se è indicato anche val.bin, teniamo il 90% per train e il 10% per validation
    std::size_t train_count = tokens.size();
    if (!val_path.empty() && tokens.size() > 20) {
        train_count = (tokens.size() * 9) / 10;
    }

    std::FILE* tr_fp = std::fopen(train_path.c_str(), "wb");
    if (!tr_fp) {
        std::fprintf(stderr, "Errore: impossibile creare '%s'\n", train_path.c_str());
        return EXIT_FAILURE;
    }
    std::size_t written_tr = std::fwrite(tokens.data(), sizeof(std::uint16_t), train_count, tr_fp);
    std::fclose(tr_fp);

    if (written_tr != train_count) {
        std::fprintf(stderr, "Errore scrittura '%s'\n", train_path.c_str());
        return EXIT_FAILURE;
    }

    std::printf("Scritti %zu token in '%s'\n", written_tr, train_path.c_str());

    if (!val_path.empty() && train_count < tokens.size()) {
        const std::size_t val_count = tokens.size() - train_count;
        std::FILE* va_fp = std::fopen(val_path.c_str(), "wb");
        if (!va_fp) {
            std::fprintf(stderr, "Errore: impossibile creare '%s'\n", val_path.c_str());
            return EXIT_FAILURE;
        }
        std::size_t written_va = std::fwrite(tokens.data() + train_count,
                                             sizeof(std::uint16_t),
                                             val_count, va_fp);
        std::fclose(va_fp);
        std::printf("Scritti %zu token in '%s'\n", written_va, val_path.c_str());
    }

    return 0;
}