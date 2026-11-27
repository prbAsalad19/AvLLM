#pragma once

#include "config.hpp"
#include "tensor.hpp"

#include <cstddef>
#include <string>
#include <vector>

struct TransformerBlockWeights {
    GpuTensor ln1_weight; // [d_model]
    GpuTensor ln1_bias;   // [d_model]
    GpuTensor w_qkv;      // [d_model * (3 * d_model)]
    GpuTensor w_out;      // [d_model * d_model]

    GpuTensor ln2_weight; // [d_model]
    GpuTensor ln2_bias;   // [d_model]
    GpuTensor w_fc1;      // [d_model * mlp_hidden]
    GpuTensor w_fc2;      // [mlp_hidden * d_model]
};

struct TransformerBlockGrads {
    GpuTensor ln1_weight;
    GpuTensor ln1_bias;
    GpuTensor w_qkv;
    GpuTensor w_out;

    GpuTensor ln2_weight;
    GpuTensor ln2_bias;
    GpuTensor w_fc1;
    GpuTensor w_fc2;

    void zero();
};

struct TransformerBlockAdam {
    GpuTensor m_ln1_w, v_ln1_w;
    GpuTensor m_ln1_b, v_ln1_b;
    GpuTensor m_w_qkv, v_w_qkv;
    GpuTensor m_w_out, v_w_out;

    GpuTensor m_ln2_w, v_ln2_w;
    GpuTensor m_ln2_b, v_ln2_b;
    GpuTensor m_w_fc1, v_w_fc1;
    GpuTensor m_w_fc2, v_w_fc2;
};

struct TransformerBlockActs {
    GpuTensor x_in;          // [BT * C]
    GpuTensor ln1_out;       // [BT * C]
    GpuTensor ln1_mean;      // [BT]
    GpuTensor ln1_rstd;      // [BT]
    GpuTensor qkv;           // [BT * 3C]
    GpuTensor q;             // [BT * C]
    GpuTensor k;             // [BT * C]
    GpuTensor v;             // [BT * C]
    GpuTensor attn_probs;    // [B * n_heads * T * T]
    GpuTensor attn_merged;   // [BT * C]
    GpuTensor attn_proj;     // [BT * C]
    GpuTensor x_mid;         // [BT * C]
    GpuTensor ln2_out;       // [BT * C]
    GpuTensor ln2_mean;      // [BT]
    GpuTensor ln2_rstd;      // [BT]
    GpuTensor mlp_h1;        // [BT * H]
    GpuTensor mlp_act;       // [BT * H]
    GpuTensor mlp_out;       // [BT * C]
    GpuTensor x_out;         // [BT * C]
};

struct TransformerWeights {
    GpuTensor tok_emb;         // [vocab_size * d_model]
    GpuTensor pos_emb;         // [seq_len * d_model]
    std::vector<TransformerBlockWeights> blocks;
    GpuTensor final_ln_weight; // [d_model]
    GpuTensor final_ln_bias;   // [d_model]
    GpuTensor lm_head;         // [d_model * vocab_size]
};

struct TransformerGrads {
    GpuTensor tok_emb;
    GpuTensor pos_emb;
    std::vector<TransformerBlockGrads> blocks;
    GpuTensor final_ln_weight;
    GpuTensor final_ln_bias;
    GpuTensor lm_head;

    void zero_all();
};

struct AdamWMoments {
    GpuTensor m_tok_emb, v_tok_emb;
    GpuTensor m_pos_emb, v_pos_emb;
    std::vector<TransformerBlockAdam> blocks;
    GpuTensor m_final_ln_w, v_final_ln_w;
    GpuTensor m_final_ln_b, v_final_ln_b;
    GpuTensor m_lm_head, v_lm_head;
};

struct TransformerActivations {
    GpuTensor x_emb;         // [BT * C]
    std::vector<TransformerBlockActs> blocks;
    GpuTensor final_ln_out;  // [BT * C]
    GpuTensor final_ln_mean; // [BT]
    GpuTensor final_ln_rstd; // [BT]
    GpuTensor logits;        // [BT * V]
    GpuTensor dlogits;       // [BT * V]
    GpuTensor loss;          // [1]

    // Buffer di lavoro riutilizzati nel backward (zero hipMalloc nel loop!)
    GpuTensor d_x;
    GpuTensor d_branch;
    GpuTensor d_norm;
    GpuTensor d_mlp_h1;
    GpuTensor d_mlp_act;
    GpuTensor d_attn_merged;
    GpuTensor d_q;
    GpuTensor d_k;
    GpuTensor d_v;
    GpuTensor d_qkv;
};

class MiniLLM {
public:
    explicit MiniLLM(const TrainConfig& cfg);
    ~MiniLLM();

    MiniLLM(const MiniLLM&) = delete;
    MiniLLM& operator=(const MiniLLM&) = delete;

    float forward_and_loss(const int* d_tokens, const int* d_targets);
    void backward_and_step(const int* d_tokens, int step_idx);

    bool save_checkpoint(const std::string& path, std::size_t step) const;
    bool load_checkpoint(const std::string& path, std::size_t& out_step);

    const TrainConfig& config() const { return cfg_; }
    const GpuTensor& logits_tensor() const { return acts_.logits; }

private:
    TrainConfig cfg_;
    GpuBlasHandle blas_handle_ = nullptr;
    TransformerWeights weights_;
    TransformerGrads grads_;
    AdamWMoments adam_state_;
    TransformerActivations acts_;

    void allocate_buffers();
    void init_weights_random();

    void matmul_row_major(const float* A, const float* B, float* C,
                          int M, int N, int K,
                          float alpha = 1.0f, float beta = 0.0f);
};