#include "config.hpp"
#include "dataset.hpp"
#include "model.hpp"
#include "tensor.hpp"
#include "bpe_tokenizer.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
std::atomic<bool> g_stop_requested{false};
void handle_signal(int) { g_stop_requested.store(true, std::memory_order_relaxed); }
}

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    TrainConfig cfg;
    std::string resume_path;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--resume") && i + 1 < argc) resume_path = argv[++i];
        else if (!std::strcmp(argv[i], "--tokenizer") && i + 1 < argc) cfg.tokenizer_path = argv[++i];
        else if (!std::strcmp(argv[i], "--train") && i + 1 < argc) cfg.train_data_path = argv[++i];
        else if (!std::strcmp(argv[i], "--val") && i + 1 < argc) cfg.val_data_path = argv[++i];
        else if (!std::strcmp(argv[i], "--steps") && i + 1 < argc) cfg.max_steps = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--batch") && i + 1 < argc) cfg.batch_size = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--seq") && i + 1 < argc) cfg.seq_len = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--save-interval") && i + 1 < argc) cfg.save_interval = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--eval-interval") && i + 1 < argc) cfg.eval_interval = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--eval-batches") && i + 1 < argc) cfg.eval_batches = std::strtoull(argv[++i], nullptr, 10);
        else if (argv[i][0] != '-') cfg.train_data_path = argv[i];
        else { std::fprintf(stderr, "Argomento non riconosciuto: %s\n", argv[i]); return EXIT_FAILURE; }
    }

    BpeTokenizer tokenizer;
    if (!tokenizer.load(cfg.tokenizer_path)) {
        std::fprintf(stderr, "Impossibile caricare tokenizer '%s'\n", cfg.tokenizer_path.c_str()); return EXIT_FAILURE;
    }
    cfg.vocab_size = static_cast<std::size_t>(tokenizer.vocab_size());
    if (!cfg.validate()) return EXIT_FAILURE;

    Dataset train_ds, val_ds;
    if (!train_ds.load_binary(cfg.train_data_path)) return EXIT_FAILURE;
    if (!val_ds.load_binary(cfg.val_data_path)) return EXIT_FAILURE;
    const std::size_t required = cfg.seq_len + 1;
    if (train_ds.num_tokens() < required || val_ds.num_tokens() < required) {
        std::fprintf(stderr, "Dataset troppo piccolo: servono almeno seq_len+1 (%zu) token in train e validation\n", required);
        return EXIT_FAILURE;
    }

    std::printf("Tokenizer: %s | vocab=%zu\n", cfg.tokenizer_path.c_str(), cfg.vocab_size);
    std::printf("Train tokens: %zu | validation tokens: %zu\n", train_ds.num_tokens(), val_ds.num_tokens());
    std::printf("Parametri stimati: ~%zu\n", cfg.estimate_params());
    MiniLLM model(cfg);
    std::size_t start_step = 0;
    if (!resume_path.empty()) {
        if (!model.load_checkpoint(resume_path, start_step)) {
            std::fprintf(stderr, "Impossibile riprendere da '%s'\n", resume_path.c_str()); return EXIT_FAILURE;
        }
        std::printf("Ripreso checkpoint '%s' allo step %zu\n", resume_path.c_str(), start_step);
    }
    if (start_step >= cfg.max_steps) { std::printf("Training già completato (%zu/%zu).\n", start_step, cfg.max_steps); return 0; }

    const std::size_t elems = cfg.batch_size * cfg.seq_len;
    std::vector<int> inputs(elems), targets(elems), eval_inputs(elems), eval_targets(elems);
    int *d_inputs = nullptr, *d_targets = nullptr;
    if (GPU_MALLOC(reinterpret_cast<void**>(&d_inputs), elems * sizeof(int)) != GPU_SUCCESS ||
        GPU_MALLOC(reinterpret_cast<void**>(&d_targets), elems * sizeof(int)) != GPU_SUCCESS) {
        std::fprintf(stderr, "GPU malloc fallita\n");
        if (d_inputs) (void)GPU_FREE(d_inputs);
        if (d_targets) (void)GPU_FREE(d_targets);
        return EXIT_FAILURE;
    }

    auto save = [&](std::size_t step) {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/ckpt_step_%06zu.bin", cfg.checkpoint_dir.c_str(), step);
        if (!model.save_checkpoint(path, step)) std::fprintf(stderr, "Errore salvataggio checkpoint %s\n", path);
        else std::printf("Checkpoint salvato: %s\n", path);
    };

    std::printf("Training avviato. Ctrl+C salva a fine step.\n");
    const auto t0 = std::chrono::steady_clock::now();
    std::size_t last_step = start_step;
    double last_elapsed = 0.0;
    bool interrupted = false;
    for (std::size_t step = start_step; step < cfg.max_steps; ++step) {
        train_ds.sample_batch(cfg.batch_size, cfg.seq_len, inputs, targets);
        (void)GPU_MEMCPY(d_inputs, inputs.data(), elems * sizeof(int), GPU_MEMCPY_HOST_TO_DEVICE);
        (void)GPU_MEMCPY(d_targets, targets.data(), elems * sizeof(int), GPU_MEMCPY_HOST_TO_DEVICE);
        const float loss = model.forward_and_loss(d_inputs, d_targets);
        model.backward_and_step(d_inputs, static_cast<int>(step + 1));
        last_step = step + 1;

        if (last_step % cfg.log_interval == 0 || last_step == 1 || last_step == cfg.max_steps) {
            (void)GPU_DEVICE_SYNCHRONIZE();
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("step %zu/%zu | train loss %.5f | Dt: %.1fs | elapsed: %.1fs\n", last_step, cfg.max_steps, loss, elapsed - last_elapsed, elapsed);
            last_elapsed = elapsed;
            std::fflush(stdout);
        }
        if (cfg.eval_interval && last_step % cfg.eval_interval == 0) {
            double sum = 0.0;
            for (std::size_t b = 0; b < cfg.eval_batches; ++b) {
                val_ds.sample_batch(cfg.batch_size, cfg.seq_len, eval_inputs, eval_targets);
                (void)GPU_MEMCPY(d_inputs, eval_inputs.data(), elems * sizeof(int), GPU_MEMCPY_HOST_TO_DEVICE);
                (void)GPU_MEMCPY(d_targets, eval_targets.data(), elems * sizeof(int), GPU_MEMCPY_HOST_TO_DEVICE);
                sum += model.forward_and_loss(d_inputs, d_targets);
            }
            (void)GPU_DEVICE_SYNCHRONIZE();
            std::printf("             validation loss %.5f (%zu batch)\n", sum / cfg.eval_batches, cfg.eval_batches);
        }
        if (cfg.save_interval && last_step % cfg.save_interval == 0) 
        {
            (void)GPU_DEVICE_SYNCHRONIZE();
            save(last_step);
        }
        if (g_stop_requested.load(std::memory_order_relaxed)) { interrupted = true; break; }
    }
    if (interrupted) 
    { 
        (void)GPU_DEVICE_SYNCHRONIZE(); save(last_step); std::printf("Interrotto in modo controllato allo step %zu.\n", last_step); 
    }
    else if (last_step > start_step && (!cfg.save_interval || last_step % cfg.save_interval != 0))
    {
        (void)GPU_DEVICE_SYNCHRONIZE();
        double sum = 0.0;
        for (std::size_t b = 0; b < cfg.eval_batches; ++b) {
            val_ds.sample_batch(cfg.batch_size, cfg.seq_len, eval_inputs, eval_targets);
            (void)GPU_MEMCPY(d_inputs, eval_inputs.data(), elems * sizeof(int), GPU_MEMCPY_HOST_TO_DEVICE);
            (void)GPU_MEMCPY(d_targets, eval_targets.data(), elems * sizeof(int), GPU_MEMCPY_HOST_TO_DEVICE);
            sum += model.forward_and_loss(d_inputs, d_targets);
        }
        std::printf("             validation loss %.5f (%zu batch)\n", sum / cfg.eval_batches, cfg.eval_batches);
        save(last_step);
    }

    (void)GPU_FREE(d_inputs); (void)GPU_FREE(d_targets);
    return EXIT_SUCCESS;
}
