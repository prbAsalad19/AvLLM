#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

inline void check_hip(hipError_t err, const char* file, int line) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "HIP Kernel Error [%s:%d]: %s\n",
                     file, line, hipGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

#define HIP_K_CHECK(cmd) check_hip((cmd), __FILE__, __LINE__)

constexpr int BLOCK_SIZE = 256;

__global__ void embedding_forward_kernel(const float* __restrict__ tok_emb,
                                       const float* __restrict__ pos_emb,
                                       const int* __restrict__ tokens,
                                       float* __restrict__ out,
                                       int B, int T, int d_model, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * d_model;
    if (idx >= total) return;

    int c = idx % d_model;
    int t = (idx / d_model) % T;
    int bt = idx / d_model; // indice combinato [0 .. B*T - 1]

    int tok_id = tokens[bt];
    float v_tok = (tok_id >= 0 && tok_id < vocab_size)
                     ? tok_emb[tok_id * d_model + c]
                     : 0.0f;
    float v_pos = pos_emb[t * d_model + c];

    out[idx] = v_tok + v_pos;
}

__global__ void add_forward_kernel(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ out,
                                 int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void gelu_forward_kernel(const float* __restrict__ x,
                                  float* __restrict__ out,
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
        constexpr float COEFF = 0.044715f;
        float val = x[idx];
        float inner = SQRT_2_OVER_PI * (val + COEFF * val * val * val);
        out[idx] = 0.5f * val * (1.0f + tanhf(inner));
    }
}

// Un thread per riga (ciascuna riga corrisponde a un token su cui predire il prossimo)
__global__ void softmax_ce_loss_and_grad_kernel(const float* __restrict__ logits,
                                                const int* __restrict__ targets,
                                                float* __restrict__ dlogits,
                                                float* __restrict__ loss_out,
                                                int N, int V) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const float* row_logits = logits + row * V;
    float* row_grad = dlogits + row * V;

    // 1. Massimo della riga per stabilità numerica (evita overflow in exp)
    float max_val = row_logits[0];
    for (int j = 1; j < V; ++j) {
        float v = row_logits[j];
        if (v > max_val) max_val = v;
    }

    // 2. Somma degli esponenziali shiftati
    float sum_exp = 0.0f;
    for (int j = 0; j < V; ++j) {
        float e = expf(row_logits[j] - max_val);
        row_grad[j] = e; // temporaneamente salviamo exp(z_j - max)
        sum_exp += e;
    }

    int target = targets[row];
    float inv_sum = 1.0f / sum_exp;
    float inv_N = 1.0f / static_cast<float>(N);

    // 3. Probabilità softmax e gradiente rispetto ai logits: (p_j - y_j) / N
    for (int j = 0; j < V; ++j) {
        float prob = row_grad[j] * inv_sum;
        float indicator = (j == target) ? 1.0f : 0.0f;
        row_grad[j] = (prob - indicator) * inv_N;
    }

    // 4. Cross-entropy loss per questa riga: -log(prob_target)
    float log_prob_target = (row_logits[target] - max_val) - logf(sum_exp);
    float row_loss = -log_prob_target;
    atomicAdd(loss_out, row_loss * inv_N);
}

__global__ void adamw_step_kernel(float* __restrict__ param,
                                float* __restrict__ grad,
                                float* __restrict__ m,
                                float* __restrict__ v,
                                int n,
                                float lr,
                                float beta1,
                                float beta2,
                                float eps,
                                float weight_decay,
                                float inv_bc1,
                                float inv_bc2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = grad[idx];
    float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
    float v_new = beta2 * v[idx] + (1.0f - beta2) * g * g;

    m[idx] = m_new;
    v[idx] = v_new;

    // Bias correction (step_idx parte da 1)
    float m_hat = m_new * inv_bc1;
    float v_hat = v_new * inv_bc2;

    float p = param[idx];
    // Decoupled weight decay (AdamW) + aggiornamento gradiente
    p = p - lr * weight_decay * p;
    p = p - lr * (m_hat / (sqrtf(v_hat) + eps));

    param[idx] = p;
    grad[idx] = 0.0f; // resetta il gradiente per lo step successivo
}

} // namespace

namespace kernels {

void launch_embedding_forward(const float* tok_emb,
                              const float* pos_emb,
                              const int* tokens,
                              float* out,
                              int B, int T, int d_model, int vocab_size,
                              hipStream_t stream) {
    int total = B * T * d_model;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(embedding_forward_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       tok_emb, pos_emb, tokens, out, B, T, d_model, vocab_size);
    HIP_K_CHECK(hipGetLastError());
}

void launch_add_forward(const float* a,
                      const float* b,
                      float* out,
                      int n,
                      hipStream_t stream) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(add_forward_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       a, b, out, n);
    HIP_K_CHECK(hipGetLastError());
}

void launch_gelu_forward(const float* x,
                       float* out,
                       int n,
                       hipStream_t stream) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(gelu_forward_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       x, out, n);
    HIP_K_CHECK(hipGetLastError());
}

void launch_softmax_ce_loss_and_grad(const float* logits,
                                     const int* targets,
                                     float* dlogits,
                                     float* loss_out,
                                     int N, int V,
                                     hipStream_t stream) {
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(softmax_ce_loss_and_grad_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       logits, targets, dlogits, loss_out, N, V);
    HIP_K_CHECK(hipGetLastError());
}

void launch_adamw_step(float* param,
                       float* grad,
                       float* m,
                       float* v,
                       int n,
                       float lr,
                       float beta1,
                       float beta2,
                       float eps,
                       float weight_decay,
                       int step_idx,
                       hipStream_t stream) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int s = (step_idx < 1) ? 1 : step_idx;
    float bc1 = 1.0f - std::pow(beta1, static_cast<float>(s));
    float bc2 = 1.0f - std::pow(beta2, static_cast<float>(s));
    float inv_bc1 = 1.0f / bc1;
    float inv_bc2 = 1.0f / bc2;

    hipLaunchKernelGGL(adamw_step_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       param, grad, m, v, n, lr, beta1, beta2, eps, weight_decay, inv_bc1, inv_bc2);
    HIP_K_CHECK(hipGetLastError());
}

__global__ void gelu_backward_kernel(const float* __restrict__ x,
                                   const float* __restrict__ dout,
                                   float* __restrict__ dx,
                                   int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float COEFF = 0.044715f;

    float v = x[idx];
    float v2 = v * v;
    float inner = SQRT_2_OVER_PI * (v + COEFF * v * v2);
    float th = tanhf(inner);
    float sech2 = 1.0f - th * th;
    float d_inner = SQRT_2_OVER_PI * (1.0f + 3.0f * COEFF * v2);

    // Derivata di 0.5 * v * (1 + tanh(inner)) rispetto a v
    float dgelu_dx = 0.5f * (1.0f + th) + 0.5f * v * sech2 * d_inner;
    dx[idx] = dout[idx] * dgelu_dx;
}

__global__ void embedding_backward_kernel(const int* __restrict__ tokens,
                                        const float* __restrict__ dx_emb,
                                        float* __restrict__ grad_tok_emb,
                                        float* __restrict__ grad_pos_emb,
                                        int B, int T, int d_model, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * d_model;
    if (idx >= total) return;

    int c = idx % d_model;
    int t = (idx / d_model) % T;
    int bt = idx / d_model;

    int tok_id = tokens[bt];
    float g = dx_emb[idx];

    // atomicAdd perché più posizioni/batch possono puntare allo stesso token o stessa posizione t
    atomicAdd(&grad_pos_emb[t * d_model + c], g);
    if (tok_id >= 0 && tok_id < vocab_size) {
        atomicAdd(&grad_tok_emb[tok_id * d_model + c], g);
    }
}

void launch_gelu_backward(const float* x,
                        const float* dout,
                        float* dx,
                        int n,
                        hipStream_t stream) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(gelu_backward_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       x, dout, dx, n);
    HIP_K_CHECK(hipGetLastError());
}

void launch_embedding_backward(const int* tokens,
                             const float* dx_emb,
                             float* grad_tok_emb,
                             float* grad_pos_emb,
                             int B, int T, int d_model, int vocab_size,
                             hipStream_t stream) {
    int total = B * T * d_model;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(embedding_backward_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       tokens, dx_emb, grad_tok_emb, grad_pos_emb, B, T, d_model, vocab_size);
    HIP_K_CHECK(hipGetLastError());
}

__global__ void split_qkv_kernel(const float* __restrict__ qkv,
                               float* __restrict__ q,
                               float* __restrict__ k,
                               float* __restrict__ v,
                               int B, int T, int n_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int C = n_heads * head_dim;
    int total = B * T * C;
    if (idx >= total) return;

    int hd = idx % head_dim;
    int h = (idx / head_dim) % n_heads;
    int t = (idx / C) % T;
    int b = idx / (T * C);

    // Layout in qkv: [b, t, {0=Q, 1=K, 2=V}, h, hd]
    int qkv_base = ((b * T + t) * 3) * C + h * head_dim + hd;
    // Layout in q, k, v: [b, h, t, hd]
    int out_idx = ((b * n_heads + h) * T + t) * head_dim + hd;

    q[out_idx] = qkv[qkv_base + 0 * C];
    k[out_idx] = qkv[qkv_base + 1 * C];
    v[out_idx] = qkv[qkv_base + 2 * C];
}

__global__ void scatter_dqkv_kernel(const float* __restrict__ dq,
                                  const float* __restrict__ dk,
                                  const float* __restrict__ dv,
                                  float* __restrict__ dqkv,
                                  int B, int T, int n_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int C = n_heads * head_dim;
    int total = B * T * C;
    if (idx >= total) return;

    int hd = idx % head_dim;
    int h = (idx / head_dim) % n_heads;
    int t = (idx / C) % T;
    int b = idx / (T * C);

    int in_idx = ((b * n_heads + h) * T + t) * head_dim + hd;
    int qkv_base = ((b * T + t) * 3) * C + h * head_dim + hd;

    dqkv[qkv_base + 0 * C] = dq[in_idx];
    dqkv[qkv_base + 1 * C] = dk[in_idx];
    dqkv[qkv_base + 2 * C] = dv[in_idx];
}

// Un thread per riga (b_h, t): calcola Q[t] . K[s] * scale per s <= t, maschera s > t, e fa softmax
__global__ void causal_attention_scores_softmax_kernel(const float* __restrict__ q,
                                                       const float* __restrict__ k,
                                                       float* __restrict__ attn_probs,
                                                       int BH, int T, int head_dim,
                                                       float scale) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = BH * T;
    if (row >= total_rows) return;

    int bh = row / T;
    int t = row % T;

    const float* q_vec = q + (bh * T + t) * head_dim;
    const float* k_base = k + bh * T * head_dim;
    float* probs_row = attn_probs + row * T;

    float max_score = -1e30f;
    for (int s = 0; s < T; ++s) {
        float val;
        if (s > t) {
            val = -1e9f; // maschera causale: impedisce di guardare il futuro
        } else {
            const float* k_vec = k_base + s * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_vec[d] * k_vec[d];
            }
            val = dot * scale;
        }
        probs_row[s] = val;
        if (val > max_score) max_score = val;
    }

    float sum_exp = 0.0f;
    for (int s = 0; s < T; ++s) {
        float e = (s > t) ? 0.0f : expf(probs_row[s] - max_score);
        probs_row[s] = e;
        sum_exp += e;
    }

    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (int s = 0; s < T; ++s) {
        probs_row[s] *= inv_sum;
    }
}

__global__ void apply_attn_to_v_and_merge_kernel(const float* __restrict__ attn_probs,
                                                 const float* __restrict__ v,
                                                 float* __restrict__ merged_out,
                                                 int B, int T, int n_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int C = n_heads * head_dim;
    int total = B * T * C;
    if (idx >= total) return;

    int hd = idx % head_dim;
    int h = (idx / head_dim) % n_heads;
    int t = (idx / C) % T;
    int b = idx / (T * C);

    int bh = b * n_heads + h;
    const float* p_row = attn_probs + (bh * T + t) * T;
    const float* v_bh = v + bh * T * head_dim;

    float acc = 0.0f;
    for (int s = 0; s <= t; ++s) {
        acc += p_row[s] * v_bh[s * head_dim + hd];
    }

    // Scrive in merged_out con layout [b, t, h, hd] -> [b, t, c]
    merged_out[((b * T + t) * n_heads + h) * head_dim + hd] = acc;
}

void launch_split_qkv(const float* qkv,
                      float* q,
                      float* k,
                      float* v,
                      int B, int T, int n_heads, int head_dim,
                      hipStream_t stream) {
    int total = B * T * n_heads * head_dim;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(split_qkv_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       qkv, q, k, v, B, T, n_heads, head_dim);
    HIP_K_CHECK(hipGetLastError());
}

void launch_scatter_dqkv(const float* dq,
                         const float* dk,
                         const float* dv,
                         float* dqkv,
                         int B, int T, int n_heads, int head_dim,
                         hipStream_t stream) {
    int total = B * T * n_heads * head_dim;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(scatter_dqkv_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       dq, dk, dv, dqkv, B, T, n_heads, head_dim);
    HIP_K_CHECK(hipGetLastError());
}

void launch_causal_attention_scores_softmax(const float* q,
                                            const float* k,
                                            float* attn_probs,
                                            int BH, int T, int head_dim,
                                            float scale,
                                            hipStream_t stream) {
    int total_rows = BH * T;
    int blocks = (total_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(causal_attention_scores_softmax_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       q, k, attn_probs, BH, T, head_dim, scale);
    HIP_K_CHECK(hipGetLastError());
}

void launch_apply_attn_to_v_and_merge(const float* attn_probs,
                                      const float* v,
                                      float* merged_out,
                                      int B, int T, int n_heads, int head_dim,
                                      hipStream_t stream) {
    int total = B * T * n_heads * head_dim;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(apply_attn_to_v_and_merge_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       attn_probs, v, merged_out, B, T, n_heads, head_dim);
    HIP_K_CHECK(hipGetLastError());
}

} // namespace kernels