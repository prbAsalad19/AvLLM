#include "config.hpp"

#include <cstdio>

bool TrainConfig::validate() const {
    if (vocab_size < 256 || vocab_size > 65535 || seq_len == 0 || d_model == 0 ||
        n_heads == 0 || n_layers == 0) {
        std::fprintf(stderr, "Errore config: dimensioni architettura non valide\n");
        return false;
    }
    if (d_model % n_heads != 0) {
        std::fprintf(stderr, "Errore config: d_model (%zu) deve essere divisibile per n_heads (%zu)\n",
                     d_model, n_heads);
        return false;
    }
    if (mlp_hidden == 0 || batch_size == 0 || max_steps == 0 || eval_batches == 0) {
        std::fprintf(stderr, "Errore config: batch_size, mlp_hidden, max_steps ed eval_batches devono essere > 0\n");
        return false;
    }
    if (learning_rate <= 0.0f || weight_decay < 0.0f || grad_clip < 0.0f) {
        std::fprintf(stderr, "Errore config: learning_rate deve essere > 0; weight_decay e grad_clip >= 0\n");
        return false;
    }
    if (log_interval == 0) {
        std::fprintf(stderr, "Errore config: log_interval deve essere > 0\n");
        return false;
    }
    return true;
}

std::size_t TrainConfig::estimate_params() const {
    const std::size_t tok_emb = vocab_size * d_model;
    const std::size_t pos_emb = seq_len * d_model;
    const std::size_t per_layer = (4 * d_model) + (3 * d_model * d_model + 3 * d_model) +
        (d_model * d_model + d_model) + (d_model * mlp_hidden + mlp_hidden) +
        (mlp_hidden * d_model + d_model);
    return tok_emb + pos_emb + n_layers * per_layer + 2 * d_model + d_model * vocab_size;
}
