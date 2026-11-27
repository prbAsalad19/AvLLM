#include "model.hpp"
#include "kernels.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

inline void check_blas(GpuBlasStatus status, const char* file, int line) {
    if (status != GPU_BLAS_SUCCESS) {
        std::fprintf(stderr, "GPU BLAS Error [%s:%d]: status %d\n",
                     file, line, static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

#define GPU_BLAS_ASSERT(cmd) check_blas((cmd), __FILE__, __LINE__)

inline void check_gpu_model(GpuError err, const char* file, int line) {
    if (err != GPU_SUCCESS) {
        std::fprintf(stderr, "GPU Model Error [%s:%d]: %s\n",
                     file, line, GPU_GET_ERROR_STRING(err));
        std::exit(EXIT_FAILURE);
    }
}

#define GPU_MODEL_ASSERT(cmd) check_gpu_model((cmd), __FILE__, __LINE__)

void fill_normal_host(std::vector<float>& buf, float mean, float stddev, unsigned long long seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(mean, stddev);
    for (float& v : buf) {
        v = dist(rng);
    }
}

void gemm_rm(GpuBlasHandle handle,
             bool transA, bool transB,
             const float* A, const float* B, float* C,
             int M, int N, int K,
             float alpha = 1.0f, float beta = 0.0f) {
    GpuBlasOperation opB = transB ? GPU_BLAS_OP_TRANSPOSE : GPU_BLAS_OP_NONE;
    GpuBlasOperation opA = transA ? GPU_BLAS_OP_TRANSPOSE : GPU_BLAS_OP_NONE;

    int lda_col = transB ? K : N;
    int ldb_col = transA ? M : K;
    int ldc_col = N;

    GPU_BLAS_ASSERT(GPU_BLAS_SGEMM(
        handle,
        opB,
        opA,
        N, M, K,
        &alpha,
        B, lda_col,
        A, ldb_col,
        &beta,
        C, ldc_col));
}

bool write_tensor_bin(std::FILE* fp, const GpuTensor& t) {
    std::vector<float> host_buf = t.to_host();
    std::size_t n = host_buf.size();
    return std::fwrite(&n, sizeof(n), 1, fp) == 1 &&
           (n == 0 || std::fwrite(host_buf.data(), sizeof(float), n, fp) == n);
}

bool read_tensor_bin(std::FILE* fp, GpuTensor& t) {
    std::size_t n = 0;
    if (std::fread(&n, sizeof(n), 1, fp) != 1) return false;
    if (n != t.size()) {
        std::fprintf(stderr, "Errore checkpoint: dimensione tensore %zu != attesa %zu\n",
                     n, t.size());
        return false;
    }
    if (n == 0) return true;
    std::vector<float> host_buf(n);
    if (std::fread(host_buf.data(), sizeof(float), n, fp) != n) return false;
    t.copy_from_host(host_buf);
    return true;
}

} // namespace

void TransformerBlockGrads::zero() {
    ln1_weight.zero();
    ln1_bias.zero();
    w_qkv.zero();
    w_out.zero();
    ln2_weight.zero();
    ln2_bias.zero();
    w_fc1.zero();
    w_fc2.zero();
}

void TransformerGrads::zero_all() {
    tok_emb.zero();
    pos_emb.zero();
    for (auto& b : blocks) {
        b.zero();
    }
    final_ln_weight.zero();
    final_ln_bias.zero();
    lm_head.zero();
}

MiniLLM::MiniLLM(const TrainConfig& cfg) : cfg_(cfg) {
    if (!cfg_.validate()) {
        std::exit(EXIT_FAILURE);
    }
    GPU_BLAS_ASSERT(GPU_BLAS_CREATE(&blas_handle_));
    allocate_buffers();
    init_weights_random();
    grads_.zero_all();
}

MiniLLM::~MiniLLM() {
    if (blas_handle_ != nullptr) {
        GPU_BLAS_ASSERT(GPU_BLAS_DESTROY(blas_handle_));
        blas_handle_ = nullptr;
    }
}

void MiniLLM::allocate_buffers() {
    const std::size_t L = cfg_.n_layers;
    const std::size_t V = cfg_.vocab_size;
    const std::size_t T = cfg_.seq_len;
    const std::size_t C = cfg_.d_model;
    const std::size_t H = cfg_.mlp_hidden;
    const std::size_t BT = cfg_.batch_size * T;
    const std::size_t BH_TT = cfg_.batch_size * cfg_.n_heads * T * T;

    // 1. Pesi globali
    weights_.tok_emb.resize(V * C);
    weights_.pos_emb.resize(T * C);
    weights_.final_ln_weight.resize(C);
    weights_.final_ln_bias.resize(C);
    weights_.lm_head.resize(C * V);

    grads_.tok_emb.resize(V * C);
    grads_.pos_emb.resize(T * C);
    grads_.final_ln_weight.resize(C);
    grads_.final_ln_bias.resize(C);
    grads_.lm_head.resize(C * V);

    adam_state_.m_tok_emb.resize(V * C); adam_state_.v_tok_emb.resize(V * C);
    adam_state_.m_pos_emb.resize(T * C); adam_state_.v_pos_emb.resize(T * C);
    adam_state_.m_final_ln_w.resize(C);  adam_state_.v_final_ln_w.resize(C);
    adam_state_.m_final_ln_b.resize(C);  adam_state_.v_final_ln_b.resize(C);
    adam_state_.m_lm_head.resize(C * V); adam_state_.v_lm_head.resize(C * V);

    adam_state_.m_tok_emb.zero(); adam_state_.v_tok_emb.zero();
    adam_state_.m_pos_emb.zero(); adam_state_.v_pos_emb.zero();
    adam_state_.m_final_ln_w.zero(); adam_state_.v_final_ln_w.zero();
    adam_state_.m_final_ln_b.zero(); adam_state_.v_final_ln_b.zero();
    adam_state_.m_lm_head.zero(); adam_state_.v_lm_head.zero();

    // 2. Blocchi Transformer [0..L-1]
    weights_.blocks.resize(L);
    grads_.blocks.resize(L);
    adam_state_.blocks.resize(L);
    acts_.blocks.resize(L);

    for (std::size_t l = 0; l < L; ++l) {
        auto& wb = weights_.blocks[l];
        wb.ln1_weight.resize(C);
        wb.ln1_bias.resize(C);
        wb.w_qkv.resize(C * (3 * C));
        wb.w_out.resize(C * C);
        wb.ln2_weight.resize(C);
        wb.ln2_bias.resize(C);
        wb.w_fc1.resize(C * H);
        wb.w_fc2.resize(H * C);

        auto& gb = grads_.blocks[l];
        gb.ln1_weight.resize(C);
        gb.ln1_bias.resize(C);
        gb.w_qkv.resize(C * (3 * C));
        gb.w_out.resize(C * C);
        gb.ln2_weight.resize(C);
        gb.ln2_bias.resize(C);
        gb.w_fc1.resize(C * H);
        gb.w_fc2.resize(H * C);

        auto& ab = adam_state_.blocks[l];
        ab.m_ln1_w.resize(C);         ab.v_ln1_w.resize(C);
        ab.m_ln1_b.resize(C);         ab.v_ln1_b.resize(C);
        ab.m_w_qkv.resize(C * 3 * C); ab.v_w_qkv.resize(C * 3 * C);
        ab.m_w_out.resize(C * C);     ab.v_w_out.resize(C * C);
        ab.m_ln2_w.resize(C);         ab.v_ln2_w.resize(C);
        ab.m_ln2_b.resize(C);         ab.v_ln2_b.resize(C);
        ab.m_w_fc1.resize(C * H);     ab.v_w_fc1.resize(C * H);
        ab.m_w_fc2.resize(H * C);     ab.v_w_fc2.resize(H * C);

        ab.m_ln1_w.zero(); ab.v_ln1_w.zero();
        ab.m_ln1_b.zero(); ab.v_ln1_b.zero();
        ab.m_w_qkv.zero(); ab.v_w_qkv.zero();
        ab.m_w_out.zero(); ab.v_w_out.zero();
        ab.m_ln2_w.zero(); ab.v_ln2_w.zero();
        ab.m_ln2_b.zero(); ab.v_ln2_b.zero();
        ab.m_w_fc1.zero(); ab.v_w_fc1.zero();
        ab.m_w_fc2.zero(); ab.v_w_fc2.zero();

        auto& act = acts_.blocks[l];
        act.x_in.resize(BT * C);
        act.ln1_out.resize(BT * C);
        act.ln1_mean.resize(BT);
        act.ln1_rstd.resize(BT);
        act.qkv.resize(BT * 3 * C);
        act.q.resize(BT * C);
        act.k.resize(BT * C);
        act.v.resize(BT * C);
        act.attn_probs.resize(BH_TT);
        act.attn_merged.resize(BT * C);
        act.attn_proj.resize(BT * C);
        act.x_mid.resize(BT * C);
        act.ln2_out.resize(BT * C);
        act.ln2_mean.resize(BT);
        act.ln2_rstd.resize(BT);
        act.mlp_h1.resize(BT * H);
        act.mlp_act.resize(BT * H);
        act.mlp_out.resize(BT * C);
        act.x_out.resize(BT * C);
    }

    // 3. Attivazioni globali + workspace backward riutilizzabile
    acts_.x_emb.resize(BT * C);
    acts_.final_ln_out.resize(BT * C);
    acts_.final_ln_mean.resize(BT);
    acts_.final_ln_rstd.resize(BT);
    acts_.logits.resize(BT * V);
    acts_.dlogits.resize(BT * V);
    acts_.loss.resize(1);

    acts_.d_x.resize(BT * C);
    acts_.d_branch.resize(BT * C);
    acts_.d_norm.resize(BT * C);
    acts_.d_mlp_h1.resize(BT * H);
    acts_.d_mlp_act.resize(BT * H);
    acts_.d_attn_merged.resize(BT * C);
    acts_.d_q.resize(BT * C);
    acts_.d_k.resize(BT * C);
    acts_.d_v.resize(BT * C);
    acts_.d_qkv.resize(BT * 3 * C);
}

void MiniLLM::init_weights_random() {
    const float init_std = 0.02f;
    // Scaling residuale GPT-2 style per stabilizzare i modelli più profondi
    const float res_std = init_std / std::sqrt(2.0f * static_cast<float>(cfg_.n_layers));
    unsigned long long s = cfg_.seed;

    auto upload_normal = [&](GpuTensor& dst, float std_dev, unsigned long long salt) {
        std::vector<float> host_buf(dst.size());
        fill_normal_host(host_buf, 0.0f, std_dev, s + salt);
        dst.copy_from_host(host_buf);
    };

    upload_normal(weights_.tok_emb, init_std, 1);
    upload_normal(weights_.pos_emb, init_std, 2);
    upload_normal(weights_.lm_head, init_std, 3);

    std::vector<float> ones(cfg_.d_model, 1.0f);
    weights_.final_ln_weight.copy_from_host(ones);
    weights_.final_ln_bias.zero();

    for (std::size_t l = 0; l < cfg_.n_layers; ++l) {
        auto& wb = weights_.blocks[l];
        unsigned long long base = 100ULL + l * 10ULL;

        wb.ln1_weight.copy_from_host(ones);
        wb.ln1_bias.zero();
        upload_normal(wb.w_qkv, init_std, base + 1);
        upload_normal(wb.w_out, res_std,  base + 2);

        wb.ln2_weight.copy_from_host(ones);
        wb.ln2_bias.zero();
        upload_normal(wb.w_fc1, init_std, base + 3);
        upload_normal(wb.w_fc2, res_std,  base + 4);
    }
}

void MiniLLM::matmul_row_major(const float* A, const float* B, float* C,
                               int M, int N, int K,
                               float alpha, float beta) {
    gemm_rm(blas_handle_, false, false, A, B, C, M, N, K, alpha, beta);
}

float MiniLLM::forward_and_loss(const int* d_tokens, const int* d_targets) {
    const int B = static_cast<int>(cfg_.batch_size);
    const int T = static_cast<int>(cfg_.seq_len);
    const int C = static_cast<int>(cfg_.d_model);
    const int NH = static_cast<int>(cfg_.n_heads);
    const int HD = C / NH;
    const int H = static_cast<int>(cfg_.mlp_hidden);
    const int V = static_cast<int>(cfg_.vocab_size);
    const int BT = B * T;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));

    // 1. Embedding
    kernels::launch_embedding_forward(
        weights_.tok_emb.data(),
        weights_.pos_emb.data(),
        d_tokens,
        acts_.x_emb.data(),
        B, T, C, V);

    const float* cur_x = acts_.x_emb.data();

    // 2. Stack di N blocchi Transformer Pre-LayerNorm
    for (std::size_t l = 0; l < cfg_.n_layers; ++l) {
        const auto& wb = weights_.blocks[l];
        auto& act = acts_.blocks[l];

        GPU_MODEL_ASSERT(GPU_MEMCPY(act.x_in.data(), cur_x, act.x_in.bytes(), GPU_MEMCPY_DEVICE_TO_DEVICE));

        // --- Sotto-blocco 1: LN1 -> Causal Self-Attention -> Residual ---
        kernels::launch_layernorm_forward(
            act.x_in.data(), wb.ln1_weight.data(), wb.ln1_bias.data(),
            act.ln1_out.data(), act.ln1_mean.data(), act.ln1_rstd.data(),
            BT, C);

        matmul_row_major(act.ln1_out.data(), wb.w_qkv.data(), act.qkv.data(), BT, 3 * C, C);
        kernels::launch_split_qkv(act.qkv.data(), act.q.data(), act.k.data(), act.v.data(),
                                  B, T, NH, HD);
        kernels::launch_causal_attention_scores_softmax(act.q.data(), act.k.data(),
                                                        act.attn_probs.data(),
                                                        B * NH, T, HD, scale);
        kernels::launch_apply_attn_to_v_and_merge(act.attn_probs.data(), act.v.data(),
                                                  act.attn_merged.data(), B, T, NH, HD);
        matmul_row_major(act.attn_merged.data(), wb.w_out.data(), act.attn_proj.data(), BT, C, C);
        kernels::launch_add_forward(act.x_in.data(), act.attn_proj.data(), act.x_mid.data(), BT * C);

        // --- Sotto-blocco 2: LN2 -> MLP (GELU) -> Residual ---
        kernels::launch_layernorm_forward(
            act.x_mid.data(), wb.ln2_weight.data(), wb.ln2_bias.data(),
            act.ln2_out.data(), act.ln2_mean.data(), act.ln2_rstd.data(),
            BT, C);

        matmul_row_major(act.ln2_out.data(), wb.w_fc1.data(), act.mlp_h1.data(), BT, H, C);
        kernels::launch_gelu_forward(act.mlp_h1.data(), act.mlp_act.data(), BT * H);
        matmul_row_major(act.mlp_act.data(), wb.w_fc2.data(), act.mlp_out.data(), BT, C, H);
        kernels::launch_add_forward(act.x_mid.data(), act.mlp_out.data(), act.x_out.data(), BT * C);

        cur_x = act.x_out.data();
    }

    // 3. Final LayerNorm + LM Head + Loss
    kernels::launch_layernorm_forward(
        cur_x, weights_.final_ln_weight.data(), weights_.final_ln_bias.data(),
        acts_.final_ln_out.data(), acts_.final_ln_mean.data(), acts_.final_ln_rstd.data(),
        BT, C);

    matmul_row_major(acts_.final_ln_out.data(), weights_.lm_head.data(), acts_.logits.data(), BT, V, C);

    acts_.loss.zero();
    kernels::launch_softmax_ce_loss_and_grad(
        acts_.logits.data(),
        d_targets,
        acts_.dlogits.data(),
        acts_.loss.data(),
        BT, V);

    float h_loss = 0.0f;
    acts_.loss.copy_to_host(&h_loss, 1);
    return h_loss;
}

void MiniLLM::backward_and_step(const int* d_tokens, int step_idx) {
    const int B = static_cast<int>(cfg_.batch_size);
    const int T = static_cast<int>(cfg_.seq_len);
    const int C = static_cast<int>(cfg_.d_model);
    const int NH = static_cast<int>(cfg_.n_heads);
    const int HD = C / NH;
    const int H = static_cast<int>(cfg_.mlp_hidden);
    const int V = static_cast<int>(cfg_.vocab_size);
    const int BT = B * T;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));

    grads_.zero_all();

    // 1. LM Head backward
    gemm_rm(blas_handle_, true, false,
            acts_.final_ln_out.data(), acts_.dlogits.data(), grads_.lm_head.data(),
            C, V, BT);
    gemm_rm(blas_handle_, false, true,
            acts_.dlogits.data(), weights_.lm_head.data(), acts_.d_norm.data(),
            BT, C, V);

    // 2. Final LayerNorm backward -> produce d_x in ingresso all'ultimo blocco
    const float* last_block_out = (cfg_.n_layers > 0)
                                      ? acts_.blocks[cfg_.n_layers - 1].x_out.data()
                                      : acts_.x_emb.data();
    kernels::launch_layernorm_backward(
        acts_.d_norm.data(),
        last_block_out,
        weights_.final_ln_weight.data(),
        acts_.final_ln_mean.data(),
        acts_.final_ln_rstd.data(),
        acts_.d_x.data(),
        grads_.final_ln_weight.data(),
        grads_.final_ln_bias.data(),
        BT, C);

    // 3. Backward attraverso i blocchi dal più profondo (L-1) a 0
    for (int l = static_cast<int>(cfg_.n_layers) - 1; l >= 0; --l) {
        const auto& wb = weights_.blocks[l];
        auto& gb = grads_.blocks[l];
        const auto& act = acts_.blocks[l];

        // --- Backward MLP + Residuale 2 (x_out = x_mid + mlp_out) ---
        // d_x contiene dL/dx_out; il ramo MLP riceve lo stesso gradiente
        gemm_rm(blas_handle_, true, false,
                act.mlp_act.data(), acts_.d_x.data(), gb.w_fc2.data(),
                H, C, BT);
        gemm_rm(blas_handle_, false, true,
                acts_.d_x.data(), wb.w_fc2.data(), acts_.d_mlp_act.data(),
                BT, H, C);

        kernels::launch_gelu_backward(act.mlp_h1.data(), acts_.d_mlp_act.data(),
                                      acts_.d_mlp_h1.data(), BT * H);

        gemm_rm(blas_handle_, true, false,
                act.ln2_out.data(), acts_.d_mlp_h1.data(), gb.w_fc1.data(),
                C, H, BT);
        gemm_rm(blas_handle_, false, true,
                acts_.d_mlp_h1.data(), wb.w_fc1.data(), acts_.d_norm.data(),
                BT, C, H);

        kernels::launch_layernorm_backward(
            acts_.d_norm.data(),
            act.x_mid.data(),
            wb.ln2_weight.data(),
            act.ln2_mean.data(),
            act.ln2_rstd.data(),
            acts_.d_branch.data(),
            gb.ln2_weight.data(),
            gb.ln2_bias.data(),
            BT, C);

        // d_x ora diventa dL/dx_mid = dL/dx_out + d_branch
        kernels::launch_add_forward(acts_.d_x.data(), acts_.d_branch.data(), acts_.d_x.data(), BT * C);

        // --- Backward Attention + Residuale 1 (x_mid = x_in + attn_proj) ---
        gemm_rm(blas_handle_, true, false,
                act.attn_merged.data(), acts_.d_x.data(), gb.w_out.data(),
                C, C, BT);
        gemm_rm(blas_handle_, false, true,
                acts_.d_x.data(), wb.w_out.data(), acts_.d_attn_merged.data(),
                BT, C, C);

        acts_.d_v.zero(); // azzera dv perché usa atomicAdd nel kernel
        kernels::launch_causal_attention_backward(
            acts_.d_attn_merged.data(),
            act.q.data(),
            act.k.data(),
            act.v.data(),
            act.attn_probs.data(),
            acts_.d_q.data(),
            acts_.d_k.data(),
            acts_.d_v.data(),
            B, T, NH, HD, scale);

        kernels::launch_scatter_dqkv(acts_.d_q.data(), acts_.d_k.data(), acts_.d_v.data(),
                                     acts_.d_qkv.data(), B, T, NH, HD);

        gemm_rm(blas_handle_, true, false,
                act.ln1_out.data(), acts_.d_qkv.data(), gb.w_qkv.data(),
                C, 3 * C, BT);
        gemm_rm(blas_handle_, false, true,
                acts_.d_qkv.data(), wb.w_qkv.data(), acts_.d_norm.data(),
                BT, C, 3 * C);

        kernels::launch_layernorm_backward(
            acts_.d_norm.data(),
            act.x_in.data(),
            wb.ln1_weight.data(),
            act.ln1_mean.data(),
            act.ln1_rstd.data(),
            acts_.d_branch.data(),
            gb.ln1_weight.data(),
            gb.ln1_bias.data(),
            BT, C);

        // d_x diventa dL/dx_in per il blocco precedente
        kernels::launch_add_forward(acts_.d_x.data(), acts_.d_branch.data(), acts_.d_x.data(), BT * C);
    }

    // 4. Embedding backward
    kernels::launch_embedding_backward(
        d_tokens,
        acts_.d_x.data(),
        grads_.tok_emb.data(),
        grads_.pos_emb.data(),
        B, T, C, V);

    // 5. Aggiornamento pesi con AdamW (weight decay solo sulle matrici 2D, non su LayerNorm)
    const float lr = cfg_.learning_rate;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float wd = cfg_.weight_decay;

    auto step_adam = [&](GpuTensor& w, GpuTensor& g, GpuTensor& m, GpuTensor& v, float decay) {
        kernels::launch_adamw_step(
            w.data(), g.data(), m.data(), v.data(),
            static_cast<int>(w.size()),
            lr, beta1, beta2, eps, decay, step_idx);
    };

    step_adam(weights_.tok_emb, grads_.tok_emb, adam_state_.m_tok_emb, adam_state_.v_tok_emb, wd);
    step_adam(weights_.pos_emb, grads_.pos_emb, adam_state_.m_pos_emb, adam_state_.v_pos_emb, wd);
    step_adam(weights_.final_ln_weight, grads_.final_ln_weight,
              adam_state_.m_final_ln_w, adam_state_.v_final_ln_w, 0.0f);
    step_adam(weights_.final_ln_bias, grads_.final_ln_bias,
              adam_state_.m_final_ln_b, adam_state_.v_final_ln_b, 0.0f);
    step_adam(weights_.lm_head, grads_.lm_head, adam_state_.m_lm_head, adam_state_.v_lm_head, wd);

    for (std::size_t l = 0; l < cfg_.n_layers; ++l) {
        auto& wb = weights_.blocks[l];
        auto& gb = grads_.blocks[l];
        auto& ab = adam_state_.blocks[l];

        step_adam(wb.ln1_weight, gb.ln1_weight, ab.m_ln1_w, ab.v_ln1_w, 0.0f);
        step_adam(wb.ln1_bias,   gb.ln1_bias,   ab.m_ln1_b, ab.v_ln1_b, 0.0f);
        step_adam(wb.w_qkv,      gb.w_qkv,      ab.m_w_qkv, ab.v_w_qkv, wd);
        step_adam(wb.w_out,      gb.w_out,      ab.m_w_out, ab.v_w_out, wd);

        step_adam(wb.ln2_weight, gb.ln2_weight, ab.m_ln2_w, ab.v_ln2_w, 0.0f);
        step_adam(wb.ln2_bias,   gb.ln2_bias,   ab.m_ln2_b, ab.v_ln2_b, 0.0f);
        step_adam(wb.w_fc1,      gb.w_fc1,      ab.m_w_fc1, ab.v_w_fc1, wd);
        step_adam(wb.w_fc2,      gb.w_fc2,      ab.m_w_fc2, ab.v_w_fc2, wd);
    }
}

bool MiniLLM::save_checkpoint(const std::string& path, std::size_t step) const {
    std::FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) { std::fprintf(stderr, "Errore: impossibile creare checkpoint '%s'\n", path.c_str()); return false; }
    const unsigned int magic = 0x4D4C4C4D; // MLLM
    const unsigned int version = 3;       // weights + AdamW moments
    const std::size_t n_layers = cfg_.n_layers;
    bool ok = std::fwrite(&magic, sizeof(magic), 1, fp) == 1 &&
              std::fwrite(&version, sizeof(version), 1, fp) == 1 &&
              std::fwrite(&step, sizeof(step), 1, fp) == 1 &&
              std::fwrite(&n_layers, sizeof(n_layers), 1, fp) == 1;

    auto write_pair = [&](const GpuTensor& m, const GpuTensor& v) {
        return write_tensor_bin(fp, m) && write_tensor_bin(fp, v);
    };
    auto write_weights = [&]() {
        bool good = write_tensor_bin(fp, weights_.tok_emb) && write_tensor_bin(fp, weights_.pos_emb) &&
            write_tensor_bin(fp, weights_.final_ln_weight) && write_tensor_bin(fp, weights_.final_ln_bias) &&
            write_tensor_bin(fp, weights_.lm_head);
        for (const auto& b : weights_.blocks) good = good && write_tensor_bin(fp, b.ln1_weight) &&
            write_tensor_bin(fp, b.ln1_bias) && write_tensor_bin(fp, b.w_qkv) && write_tensor_bin(fp, b.w_out) &&
            write_tensor_bin(fp, b.ln2_weight) && write_tensor_bin(fp, b.ln2_bias) &&
            write_tensor_bin(fp, b.w_fc1) && write_tensor_bin(fp, b.w_fc2);
        return good;
    };
    auto write_optimizer = [&]() {
        bool good = write_pair(adam_state_.m_tok_emb, adam_state_.v_tok_emb) &&
            write_pair(adam_state_.m_pos_emb, adam_state_.v_pos_emb) &&
            write_pair(adam_state_.m_final_ln_w, adam_state_.v_final_ln_w) &&
            write_pair(adam_state_.m_final_ln_b, adam_state_.v_final_ln_b) &&
            write_pair(adam_state_.m_lm_head, adam_state_.v_lm_head);
        for (const auto& b : adam_state_.blocks) good = good &&
            write_pair(b.m_ln1_w, b.v_ln1_w) && write_pair(b.m_ln1_b, b.v_ln1_b) &&
            write_pair(b.m_w_qkv, b.v_w_qkv) && write_pair(b.m_w_out, b.v_w_out) &&
            write_pair(b.m_ln2_w, b.v_ln2_w) && write_pair(b.m_ln2_b, b.v_ln2_b) &&
            write_pair(b.m_w_fc1, b.v_w_fc1) && write_pair(b.m_w_fc2, b.v_w_fc2);
        return good;
    };
    ok = ok && write_weights() && write_optimizer();
    const bool close_ok = std::fclose(fp) == 0;
    return ok && close_ok;
}

bool MiniLLM::load_checkpoint(const std::string& path, std::size_t& out_step) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) { std::fprintf(stderr, "Errore: impossibile leggere checkpoint '%s'\n", path.c_str()); return false; }
    unsigned int magic = 0, version = 0;
    std::size_t n_layers = 0;
    if (std::fread(&magic, sizeof(magic), 1, fp) != 1 ||
        std::fread(&version, sizeof(version), 1, fp) != 1 ||
        std::fread(&out_step, sizeof(out_step), 1, fp) != 1 ||
        std::fread(&n_layers, sizeof(n_layers), 1, fp) != 1 ||
        magic != 0x4D4C4C4D || version != 3 || n_layers != cfg_.n_layers) {
        std::fprintf(stderr, "Checkpoint incompatibile: richiesto formato v3 e %zu layer (trovati v%u, %zu)\n",
                     cfg_.n_layers, version, n_layers);
        std::fclose(fp); return false;
    }
    auto read_pair = [&](GpuTensor& m, GpuTensor& v) {
        return read_tensor_bin(fp, m) && read_tensor_bin(fp, v);
    };
    auto read_weights = [&]() {
        bool good = read_tensor_bin(fp, weights_.tok_emb) && read_tensor_bin(fp, weights_.pos_emb) &&
            read_tensor_bin(fp, weights_.final_ln_weight) && read_tensor_bin(fp, weights_.final_ln_bias) &&
            read_tensor_bin(fp, weights_.lm_head);
        for (auto& b : weights_.blocks) good = good && read_tensor_bin(fp, b.ln1_weight) &&
            read_tensor_bin(fp, b.ln1_bias) && read_tensor_bin(fp, b.w_qkv) && read_tensor_bin(fp, b.w_out) &&
            read_tensor_bin(fp, b.ln2_weight) && read_tensor_bin(fp, b.ln2_bias) &&
            read_tensor_bin(fp, b.w_fc1) && read_tensor_bin(fp, b.w_fc2);
        return good;
    };
    auto read_optimizer = [&]() {
        bool good = read_pair(adam_state_.m_tok_emb, adam_state_.v_tok_emb) &&
            read_pair(adam_state_.m_pos_emb, adam_state_.v_pos_emb) &&
            read_pair(adam_state_.m_final_ln_w, adam_state_.v_final_ln_w) &&
            read_pair(adam_state_.m_final_ln_b, adam_state_.v_final_ln_b) &&
            read_pair(adam_state_.m_lm_head, adam_state_.v_lm_head);
        for (auto& b : adam_state_.blocks) good = good &&
            read_pair(b.m_ln1_w, b.v_ln1_w) && read_pair(b.m_ln1_b, b.v_ln1_b) &&
            read_pair(b.m_w_qkv, b.v_w_qkv) && read_pair(b.m_w_out, b.v_w_out) &&
            read_pair(b.m_ln2_w, b.v_ln2_w) && read_pair(b.m_ln2_b, b.v_ln2_b) &&
            read_pair(b.m_w_fc1, b.v_w_fc1) && read_pair(b.m_w_fc2, b.v_w_fc2);
        return good;
    };
    const bool ok = read_weights() && read_optimizer();
    std::fclose(fp);
    return ok;
}
