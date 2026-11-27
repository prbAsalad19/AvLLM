#ifndef AVLLM_INCLUDE_KERNELS_HPP
#define AVLLM_INCLUDE_KERNELS_HPP

#include "gpu_backend.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int BLOCK_SIZE = 256;

inline void check_gpu_kernel(GpuError err, const char* file, int line) {
    if (err != GPU_SUCCESS) {
        std::fprintf(stderr, "GPU Kernel Error [%s:%d]: %s\n",
                     file, line, GPU_GET_ERROR_STRING(err));
        std::exit(EXIT_FAILURE);
    }
}

#define GPU_K_CHECK(cmd) check_gpu_kernel((cmd), __FILE__, __LINE__)

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
    int bt = idx / d_model;

    int tok_id = tokens[bt];
    float te = (tok_id >= 0 && tok_id < vocab_size) ? tok_emb[tok_id * d_model + c] : 0.0f;
    float pe = pos_emb[t * d_model + c];
    out[idx] = te + pe;
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

    atomicAdd(&grad_pos_emb[t * d_model + c], g);
    if (tok_id >= 0 && tok_id < vocab_size) {
        atomicAdd(&grad_tok_emb[tok_id * d_model + c], g);
    }
}

// Un thread per riga n in [0, N)
__global__ void layernorm_forward_kernel(const float* __restrict__ x,
                                       const float* __restrict__ weight,
                                       const float* __restrict__ bias,
                                       float* __restrict__ y,
                                       float* __restrict__ mean_out,
                                       float* __restrict__ rstd_out,
                                       int N, int C, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const float* x_row = x + row * C;
    float* y_row = y + row * C;

    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        sum += x_row[c];
    }
    float mean = sum / static_cast<float>(C);

    float var_sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        float diff = x_row[c] - mean;
        var_sum += diff * diff;
    }
    float rstd = rsqrtf(var_sum / static_cast<float>(C) + eps);

    mean_out[row] = mean;
    rstd_out[row] = rstd;

    for (int c = 0; c < C; ++c) {
        float norm = (x_row[c] - mean) * rstd;
        y_row[c] = norm * weight[c] + bias[c];
    }
}

// Un thread per riga n in [0, N): calcola dx e accumula dweight e dbias via atomicAdd
__global__ void layernorm_backward_kernel(const float* __restrict__ dout,
                                        const float* __restrict__ x,
                                        const float* __restrict__ weight,
                                        const float* __restrict__ mean,
                                        const float* __restrict__ rstd,
                                        float* __restrict__ dx,
                                        float* __restrict__ dweight,
                                        float* __restrict__ dbias,
                                        int N, int C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const float* dout_row = dout + row * C;
    const float* x_row = x + row * C;
    float* dx_row = dx + row * C;

    float m = mean[row];
    float rs = rstd[row];

    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;

    for (int c = 0; c < C; ++c) {
        float norm = (x_row[c] - m) * rs;
        float dy = dout_row[c];
        float dnorm = dy * weight[c];

        dnorm_mean += dnorm;
        dnorm_norm_mean += dnorm * norm;

        atomicAdd(&dbias[c], dy);
        atomicAdd(&dweight[c], dy * norm);
    }

    float inv_C = 1.0f / static_cast<float>(C);
    dnorm_mean *= inv_C;
    dnorm_norm_mean *= inv_C;

    for (int c = 0; c < C; ++c) {
        float norm = (x_row[c] - m) * rs;
        float dnorm = dout_row[c] * weight[c];
        dx_row[c] = (dnorm - dnorm_mean - norm * dnorm_norm_mean) * rs;
    }
}

__global__ void add_forward_kernel(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ out,
                                 int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] + b[idx];
}

__global__ void gelu_forward_kernel(const float* __restrict__ x,
                                  float* __restrict__ out,
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float COEFF = 0.044715f;

    float v = x[idx];
    float inner = SQRT_2_OVER_PI * (v + COEFF * v * v * v);
    out[idx] = 0.5f * v * (1.0f + tanhf(inner));
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

    float dgelu_dx = 0.5f * (1.0f + th) + 0.5f * v * sech2 * d_inner;
    dx[idx] = dout[idx] * dgelu_dx;
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

    int qkv_base = ((b * T + t) * 3) * C + h * head_dim + hd;
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
            val = -1e9f;
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

    merged_out[((b * T + t) * n_heads + h) * head_dim + hd] = acc;
}

// Kernel 1 del backward attention: un thread per (bh, t)
// Calcola dQ[bh, t, :] e accumula dV[bh, s, :] per s <= t
__global__ void causal_attn_bwd_dq_dv_kernel(const float* __restrict__ d_merged,
                                             const float* __restrict__ q,
                                             const float* __restrict__ k,
                                             const float* __restrict__ v,
                                             const float* __restrict__ attn_probs,
                                             float* __restrict__ dq,
                                             float* __restrict__ dv,
                                             int B, int T, int n_heads, int head_dim,
                                             float scale) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int BH = B * n_heads;
    if (row >= BH * T) return;

    int bh = row / T;
    int t = row % T;
    int b = bh / n_heads;
    int h = bh % n_heads;

    const float* d_out_t = d_merged + ((b * T + t) * n_heads + h) * head_dim;
    const float* p_row = attn_probs + row * T;
    const float* k_bh = k + bh * T * head_dim;
    const float* v_bh = v + bh * T * head_dim;

    float* dq_t = dq + (bh * T + t) * head_dim;
    float* dv_bh = dv + bh * T * head_dim;

    // 1. Calcola dot_p_dp = sum_{s<=t} p[t, s] * dp[t, s], dove dp[t, s] = d_out_t . v[s]
    //    e contemporaneamente accumula dv[s] += p[t, s] * d_out_t
    float dot_p_dp = 0.0f;
    for (int s = 0; s <= t; ++s) {
        float p_ts = p_row[s];
        const float* v_s = v_bh + s * head_dim;
        float* dv_s = dv_bh + s * head_dim;

        float dp = 0.0f;
        for (int hd = 0; hd < head_dim; ++hd) {
            float g_val = d_out_t[hd];
            dp += g_val * v_s[hd];
            atomicAdd(&dv_s[hd], p_ts * g_val);
        }
        dot_p_dp += p_ts * dp;
    }

    // 2. Calcola dq[t, :] = sum_{s<=t} d_score[t, s] * k[s, :]
    for (int hd = 0; hd < head_dim; ++hd) {
        dq_t[hd] = 0.0f;
    }

    for (int s = 0; s <= t; ++s) {
        float p_ts = p_row[s];
        const float* v_s = v_bh + s * head_dim;
        const float* k_s = k_bh + s * head_dim;

        float dp = 0.0f;
        for (int hd = 0; hd < head_dim; ++hd) {
            dp += d_out_t[hd] * v_s[hd];
        }
        float d_score = p_ts * (dp - dot_p_dp) * scale;
        for (int hd = 0; hd < head_dim; ++hd) {
            dq_t[hd] += d_score * k_s[hd];
        }
    }
}

__global__ void causal_attn_bwd_dot_kernel(
    const float* __restrict__ d_merged,
    const float* __restrict__ v,
    const float* __restrict__ attn_probs,
    float* __restrict__ scratch,
    int B, int T, int n_heads, int head_dim)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int BH = B * n_heads;

    if (row >= BH * T) return;

    int bh = row / T;
    int t = row % T;
    int b = bh / n_heads;
    int h = bh % n_heads;

    const float* d_out_t =
        d_merged + ((b * T + t) * n_heads + h) * head_dim;

    const float* p_row = attn_probs + row * T;
    const float* v_bh = v + bh * T * head_dim;

    float dot_p_dp = 0.0f;

    for (int m = 0; m <= t; ++m) {
        const float* v_m = v_bh + m * head_dim;

        float dp_m = 0.0f;

        for (int hd = 0; hd < head_dim; ++hd) {
            dp_m += d_out_t[hd] * v_m[hd];
        }

        dot_p_dp += p_row[m] * dp_m;
    }

    scratch[row] = dot_p_dp;
}

// Kernel 2 del backward attention: un thread per (bh, s)
// Calcola dK[bh, s, :] sommando su tutte le query t >= s (zero atomicAdd su dK!)
__global__ void causal_attn_bwd_dk_kernel(
    const float* __restrict__ d_merged,
    const float* __restrict__ q,
    const float* __restrict__ v,
    const float* __restrict__ attn_probs,
    const float* __restrict__ dot_p_dp,
    float* __restrict__ dk,
    int B, int T, int n_heads, int head_dim,
    float scale)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int BH = B * n_heads;

    if (col >= BH * T) return;

    int bh = col / T;
    int s = col % T;
    int b = bh / n_heads;
    int h = bh % n_heads;

    const float* q_bh = q + bh * T * head_dim;
    const float* v_bh = v + bh * T * head_dim;

    const float* v_s = v_bh + s * head_dim;

    float* dk_s = dk + (bh * T + s) * head_dim;

    for (int hd = 0; hd < head_dim; ++hd) {
        dk_s[hd] = 0.0f;
    }

    for (int t = s; t < T; ++t) {
        const float* d_out_t =
            d_merged + ((b * T + t) * n_heads + h) * head_dim;

        const float* p_row = attn_probs + (bh * T + t) * T;
        const float* q_t = q_bh + t * head_dim;

        float dp_s = 0.0f;

        for (int hd = 0; hd < head_dim; ++hd) {
            dp_s += d_out_t[hd] * v_s[hd];
        }

        float d_score =
            p_row[s] * (dp_s - dot_p_dp[bh * T + t]) * scale;

        for (int hd = 0; hd < head_dim; ++hd) {
            dk_s[hd] += d_score * q_t[hd];
        }
    }
}

__global__ void softmax_ce_kernel(const float* __restrict__ logits,
                                const int* __restrict__ targets,
                                float* __restrict__ dlogits,
                                float* __restrict__ loss_out,
                                int N, int V) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const float* row_logits = logits + row * V;
    float* row_dlogits = dlogits + row * V;
    int target = targets[row];

    float max_val = -1e30f;
    for (int j = 0; j < V; ++j) {
        float v = row_logits[j];
        if (v > max_val) max_val = v;
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < V; ++j) {
        sum_exp += expf(row_logits[j] - max_val);
    }

    float inv_sum = 1.0f / sum_exp;
    float inv_N = 1.0f / static_cast<float>(N);

    for (int j = 0; j < V; ++j) {
        float prob = expf(row_logits[j] - max_val) * inv_sum;
        float grad = prob - ((j == target) ? 1.0f : 0.0f);
        row_dlogits[j] = grad * inv_N;
    }

    if (target >= 0 && target < V) {
        float target_logit = row_logits[target];
        float nll = -(target_logit - max_val - logf(sum_exp));
        atomicAdd(loss_out, nll * inv_N);
    }
}

__global__ void adamw_kernel(float* __restrict__ param,
                           const float* __restrict__ grad,
                           float* __restrict__ m,
                           float* __restrict__ v,
                           int n,
                           float lr,
                           float beta1,
                           float beta2,
                           float eps,
                           float weight_decay,
                           float bias_correction1,
                           float bias_correction2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = grad[idx];
    float p = param[idx];

    float m_val = beta1 * m[idx] + (1.0f - beta1) * g;
    float v_val = beta2 * v[idx] + (1.0f - beta2) * (g * g);

    m[idx] = m_val;
    v[idx] = v_val;

    float m_hat = m_val / bias_correction1;
    float v_hat = v_val / bias_correction2;

    p -= lr * weight_decay * p;
    p -= lr * (m_hat / (sqrtf(v_hat) + eps));

    param[idx] = p;
}

} // namespace

namespace kernels {

inline void launch_embedding_forward(const float* tok_emb,
                              const float* pos_emb,
                              const int* tokens,
                              float* out,
                              int B, int T, int d_model, int vocab_size,
                              hipStream_t stream = hipStreamDefault) {
    int total = B * T * d_model;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(embedding_forward_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       tok_emb, pos_emb, tokens, out, B, T, d_model, vocab_size);
    GPU_K_CHECK(hipGetLastError());
}

inline void launch_embedding_backward(const int* tokens,
                             const float* dx_emb,
                             float* grad_tok_emb,
                             float* grad_pos_emb,
                             int B, int T, int d_model, int vocab_size,
                             hipStream_t stream = hipStreamDefault) {
    int total = B * T * d_model;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(embedding_backward_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       tokens, dx_emb, grad_tok_emb, grad_pos_emb, B, T, d_model, vocab_size);
    GPU_K_CHECK(hipGetLastError());
}

inline void launch_layernorm_forward(const float* x,
                              const float* weight,
                              const float* bias,
                              float* y,
                              float* mean_out,
                              float* rstd_out,
                              int N, int C,
                              float eps = 1e-5f,
                              hipStream_t stream = hipStreamDefault) {
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(layernorm_forward_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       x, weight, bias, y, mean_out, rstd_out, N, C, eps);
    GPU_K_CHECK(hipGetLastError());
}

inline void launch_layernorm_backward(const float* dout,
                               const float* x,
                               const float* weight,
                               const float* mean,
                               const float* rstd,
                               float* dx,
                               float* dweight,
                               float* dbias,
                               int N, int C,
                               hipStream_t stream = hipStreamDefault) {
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(layernorm_backward_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       dout, x, weight, mean, rstd, dx, dweight, dbias, N, C);
    GPU_K_CHECK(hipGetLastError());
}

inline void launch_add_forward(const float* a,
                      const float* b,
                      float* out,
                      int n,
                      hipStream_t stream = hipStreamDefault) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(add_forward_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       a, b, out, n);
    GPU_K_CHECK(hipGetLastError());
}

inline void launch_gelu_forward(const float* x,
                       float* out,
                       int n,
                       hipStream_t stream = hipStreamDefault) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(gelu_forward_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       x, out, n);
    GPU_K_CHECK(hipGetLastError());
}

inline void launch_gelu_backward(const float* x,
                        const float* dout,
                        float* dx,
                        int n,
                        hipStream_t stream = hipStreamDefault) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(gelu_backward_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       x, dout, dx, n);
    GPU_K_CHECK(hipGetLastError());
}

inline void launch_split_qkv(const float* qkv,
                      float* q,
                      float* k,
                      float* v,
                      int B, int T, int n_heads, int head_dim,
                      hipStream_t stream = hipStreamDefault) {
    int total = B * T * n_heads * head_dim;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(split_qkv_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       qkv, q, k, v, B, T, n_heads, head_dim);
    GPU_K_CHECK(hipGetLastError());
}

inline void launch_scatter_dqkv(const float* dq,
                         const float* dk,
                         const float* dv,
                         float* dqkv,
                         int B, int T, int n_heads, int head_dim,
                         hipStream_t stream = hipStreamDefault) {
    int total = B * T * n_heads * head_dim;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(scatter_dqkv_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       dq, dk, dv, dqkv, B, T, n_heads, head_dim);
    GPU_K_CHECK(hipGetLastError());
}

inline void launch_causal_attention_scores_softmax(const float* q,
                                            const float* k,
                                            float* attn_probs,
                                            int BH, int T, int head_dim,
                                            float scale,
                                            hipStream_t stream = hipStreamDefault) {
    int total_rows = BH * T;
    int blocks = (total_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(causal_attention_scores_softmax_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       q, k, attn_probs, BH, T, head_dim, scale);
    GPU_K_CHECK(hipGetLastError());
}

inline void launch_apply_attn_to_v_and_merge(const float* attn_probs,
                                      const float* v,
                                      float* merged_out,
                                      int B, int T, int n_heads, int head_dim,
                                      hipStream_t stream = hipStreamDefault) {
    int total = B * T * n_heads * head_dim;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(apply_attn_to_v_and_merge_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       attn_probs, v, merged_out, B, T, n_heads, head_dim);
    GPU_K_CHECK(hipGetLastError());
}

inline void launch_causal_attention_backward(
    const float* d_merged,
    const float* q,
    const float* k,
    const float* v,
    const float* attn_probs,
    float* dq,
    float* dk,
    float* dv,
    int B, int T, int n_heads, int head_dim,
    float scale,
    hipStream_t stream = hipStreamDefault)
{
    int total_rows = B * n_heads * T;
    int blocks = (total_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 1. Precompute dot_p_dp.
    // dq is temporarily used as scratch memory.
    hipLaunchKernelGGL(
        causal_attn_bwd_dot_kernel,
        dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
        d_merged, v, attn_probs, dq,
        B, T, n_heads, head_dim
    );
    GPU_K_CHECK(hipGetLastError());

    // 2. Compute dK using precomputed values.
    hipLaunchKernelGGL(
        causal_attn_bwd_dk_kernel,
        dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
        d_merged, q, v, attn_probs, dq, dk,
        B, T, n_heads, head_dim, scale
    );
    GPU_K_CHECK(hipGetLastError());

    // 3. Compute dQ and dV.
    // This overwrites the temporary contents of dq.
    hipLaunchKernelGGL(
        causal_attn_bwd_dq_dv_kernel,
        dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
        d_merged, q, k, v, attn_probs,
        dq, dv, B, T, n_heads, head_dim, scale
    );
    GPU_K_CHECK(hipGetLastError());
}

inline void launch_softmax_ce_loss_and_grad(const float* logits,
                                     const int* targets,
                                     float* dlogits,
                                     float* loss_out,
                                     int N, int V,
                                     hipStream_t stream = hipStreamDefault) {
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(softmax_ce_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       logits, targets, dlogits, loss_out, N, V);
    GPU_K_CHECK(hipGetLastError());
}

inline void launch_adamw_step(float* param,
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
                       hipStream_t stream = hipStreamDefault) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step_idx));
    float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step_idx));

    hipLaunchKernelGGL(adamw_kernel, dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                       param, grad, m, v, n, lr, beta1, beta2, eps, weight_decay, bc1, bc2);
    GPU_K_CHECK(hipGetLastError());
}

} // namespace kernels

#endif // AVLLM_INCLUDE_KERNELS_HPP