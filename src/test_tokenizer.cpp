#include "bpe_tokenizer.hpp"

#include <iostream>
#include <string>
#include <vector>

int main() {
    BpeTokenizer tok;

    if (!tok.load("data/tokenizer.bpe")) {
        std::cerr << "Errore caricamento tokenizer\n";
        return 1;
    }

    const std::vector<std::string> tests = {
        "Ciclismo, perché è importante?",
        "Ciao, come stai? È una bella giornata.",
        "Gallia est omnis divisa in partes tres.",
        "Romae imperium maximum erat.",
        "Italiano e latino: linguae diversae.",
        "Accenti: à è é ì ò ù. Apostrofi: l'acqua.",
        "Prima riga\nSeconda riga\n"
    };

    bool all_ok = true;

    for (const auto& original : tests) {
        const auto encoded = tok.encode(original);
        const auto decoded = tok.decode(encoded);
        const bool ok = original == decoded;

        std::cout << (ok ? "[OK] " : "[FAIL] ")
                  << original << '\n';

        if (!ok) {
            std::cout << "Decoded: " << decoded << '\n';
            all_ok = false;
        }
    }

    return all_ok ? 0 : 1;
}