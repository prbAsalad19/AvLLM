#include "bpe_tokenizer.hpp"
#include "config.hpp"
#include "model.hpp"
#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

int sample_from_logits(const float* logits, int V, float temperature, std::mt19937& rng) {
    if (temperature <= 1e-6f) {
        int best_idx = 0;
        float best_val = logits[0];
        for (int i = 1; i < V; ++i) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_idx = i;
            }
        }
        return best_idx;
    }

    float max_l = logits[0];
    for (int i = 1; i < V; ++i) {
        if (logits[i] > max_l) max_l = logits[i];
    }

    std::vector<double> probs(static_cast<std::size_t>(V), 0.0);
    double sum_p = 0.0;
    for (int i = 0; i < V; ++i) {
        double scaled = (static_cast<double>(logits[i]) - static_cast<double>(max_l)) /
                        static_cast<double>(temperature);
        double p = std::exp(scaled);
        probs[static_cast<std::size_t>(i)] = p;
        sum_p += p;
    }

    std::uniform_real_distribution<double> dist(0.0, sum_p);
    double r = dist(rng);
    double acc = 0.0;
    for (int i = 0; i < V; ++i) {
        acc += probs[static_cast<std::size_t>(i)];
        if (r <= acc) {
            return i;
        }
    }
    return V - 1;
}

void print_usage(const char* prog) {
    std::fprintf(stderr,
                 "Uso: %s <checkpoint.bin> [prompt] [num_tokens] [temperature] [seed] [--tokenizer path] [--seq N]\n"
                 "Esempio:\n"
                 "  HSA_OVERRIDE_GFX_VERSION=10.3.0 %s checkpoints/ckpt_step_000500.bin \"In principio \" 300 0.8 42\n",
                 prog, prog);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::string ckpt_path;
    std::string tokenizer_path = "data/tokenizer.bpe";
    std::string prompt = "The ";
    std::size_t gen_tokens = 300;
    float temperature = 0.8f;
    unsigned int seed = 42U;
    std::size_t seq_override = 0;
    int pos_idx = 0;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--tokenizer") && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (!std::strcmp(argv[i], "--seq") && i + 1 < argc) {
            seq_override = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        } else if (argv[i][0] != '-') {
            switch (pos_idx++) {
                case 0: ckpt_path = argv[i]; break;
                case 1: prompt = argv[i]; break;
                case 2: gen_tokens = static_cast<std::size_t>(std::strtoull(argv[i], nullptr, 10)); break;
                case 3: temperature = std::strtof(argv[i], nullptr); break;
                case 4: seed = static_cast<unsigned int>(std::strtoul(argv[i], nullptr, 10)); break;
                default:
                    std::fprintf(stderr, "Argomento posizionale extra ignorato: %s\n", argv[i]);
                    break;
            }
        } else {
            std::fprintf(stderr, "Argomento non riconosciuto: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
    }

    if (ckpt_path.empty()) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    BpeTokenizer tok;
    if (!tok.load(tokenizer_path)) {
        std::fprintf(stderr, "Errore: impossibile caricare '%s'\n", tokenizer_path.c_str());
        return EXIT_FAILURE;
    }

    TrainConfig cfg;
    cfg.vocab_size = static_cast<std::size_t>(tok.vocab_size());
    if (seq_override > 0) {
        cfg.seq_len = seq_override;
    }
    cfg.batch_size = 1;

    if (!cfg.validate()) {
        return EXIT_FAILURE;
    }

    MiniLLM model(cfg);
    std::size_t ckpt_step = 0;
    if (!model.load_checkpoint(ckpt_path, ckpt_step)) {
        std::fprintf(stderr, "Errore: impossibile caricare '%s'\n", ckpt_path.c_str());
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "Caricato checkpoint '%s' (step %zu, vocab=%zu, seq=%zu)\n",
                 ckpt_path.c_str(), ckpt_step, cfg.vocab_size, cfg.seq_len);

    std::vector<int> context = tok.encode(prompt);
    if (context.empty()) {
        context.push_back(static_cast<int>(' '));
    }
    context.reserve(context.size() + gen_tokens);

    std::printf("%s", prompt.c_str());
    std::fflush(stdout);

    const std::size_t T = cfg.seq_len;
    const std::size_t V = cfg.vocab_size;
    const std::size_t batch_elems = cfg.batch_size * T;
    const std::size_t logits_elems = batch_elems * V;

    if (model.logits_tensor().size() < logits_elems) {
        std::fprintf(stderr, "Errore: dimensione logits (%zu) inferiore ad attesa (%zu)\n",
                     model.logits_tensor().size(), logits_elems);
        return EXIT_FAILURE;
    }

    std::vector<int> h_inputs(batch_elems, 0);
    std::vector<int> h_targets(batch_elems, 0);

    int* d_inputs = nullptr;
    int* d_targets = nullptr;

    if (GPU_MALLOC(reinterpret_cast<void**>(&d_inputs), batch_elems * sizeof(int)) != GPU_SUCCESS ||
        GPU_MALLOC(reinterpret_cast<void**>(&d_targets), batch_elems * sizeof(int)) != GPU_SUCCESS) {
        std::fprintf(stderr, "Errore: GPU malloc fallita in generate\n");
        if (d_inputs) (void)GPU_FREE(d_inputs);
        if (d_targets) (void)GPU_FREE(d_targets);
        return EXIT_FAILURE;
    }

    std::vector<float> h_logits(logits_elems, 0.0f);
    std::mt19937 rng(seed);

    for (std::size_t step = 0; step < gen_tokens; ++step) {
        std::fill(h_inputs.begin(), h_inputs.end(), 0);
        const std::size_t win_len = std::min(context.size(), T);
        const std::size_t start_pos = context.size() - win_len;
        const std::size_t dst_offset = T - win_len;

        for (std::size_t i = 0; i < win_len; ++i) {
            h_inputs[dst_offset + i] = context[start_pos + i];
        }

        (void)GPU_MEMCPY(d_inputs, h_inputs.data(), batch_elems * sizeof(int), GPU_MEMCPY_HOST_TO_DEVICE);
        (void)GPU_MEMCPY(d_targets, h_targets.data(), batch_elems * sizeof(int), GPU_MEMCPY_HOST_TO_DEVICE);

        (void)model.forward_and_loss(d_inputs, d_targets);
        model.logits_tensor().copy_to_host(h_logits.data(), logits_elems);

        const float* last_pos_logits = h_logits.data() + (T - 1) * V;
        const int next_tok = sample_from_logits(last_pos_logits, static_cast<int>(V), temperature, rng);

        context.push_back(next_tok);
        const std::string piece = tok.decode({next_tok});
        std::printf("%s", piece.c_str());
        std::fflush(stdout);
    }

    std::printf("\n");
    (void)GPU_FREE(d_inputs);
    (void)GPU_FREE(d_targets);
    return EXIT_SUCCESS;
}
