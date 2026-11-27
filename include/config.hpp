#pragma once

#include <cstddef>
#include <string>

struct TrainConfig {
    std::size_t vocab_size = 256;
    std::size_t seq_len = 256;
    std::size_t d_model = 384;
    std::size_t n_heads = 6;
    std::size_t n_layers = 6;
    std::size_t mlp_hidden = 1536;

    std::size_t batch_size = 32;
    std::size_t max_steps = 20000;
    float learning_rate = 3e-4f;
    float weight_decay = 0.01f;
    float grad_clip = 1.0f;

    std::size_t log_interval = 5;
    std::size_t eval_interval = 500;
    std::size_t eval_batches = 20;
    std::size_t save_interval = 2000;
    unsigned long long seed = 42ULL;

    std::string train_data_path = "data/train.bin";
    std::string val_data_path = "data/val.bin";
    std::string tokenizer_path = "data/tokenizer.bpe";
    std::string checkpoint_dir = "checkpoints_15M";

    bool validate() const;
    std::size_t estimate_params() const;
};
