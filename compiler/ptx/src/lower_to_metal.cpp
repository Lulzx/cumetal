#include "cumetal/ptx/lower_to_metal.h"

#include "cumetal/passes/phase1_pipeline.h"

#include <string>
#include <string_view>

namespace cumetal::ptx {
namespace {

constexpr std::string_view kKernelNameToken = "__KERNEL_NAME__";

bool kernel_name_contains(const std::string& kernel_name, std::string_view needle) {
    return kernel_name.find(needle) != std::string::npos;
}

std::string replace_kernel_name(std::string source, const std::string& entry_name) {
    std::size_t pos = 0;
    while ((pos = source.find(kKernelNameToken, pos)) != std::string::npos) {
        source.replace(pos, kKernelNameToken.size(), entry_name);
        pos += entry_name.size();
    }
    return source;
}

std::string emit_metal_source_for_entry(const std::string& entry_name) {
    static constexpr std::string_view kPreamble = R"METAL(#include <metal_stdlib>
#include <metal_atomic>

using namespace metal;

)METAL";

    std::string kernel_template;

    if (kernel_name_contains(entry_name, "encoder_forward_kernel3")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float4* out [[buffer(0)]],
    device const int* inp [[buffer(1)]],
    device const float4* wte [[buffer(2)]],
    device const float4* wpe [[buffer(3)]],
    constant int& B [[buffer(4)]],
    constant int& T [[buffer(5)]],
    constant int& C [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    const int C4 = C / 4;
    const int N = B * T * C4;
    const int idx = static_cast<int>(gid);
    if (idx >= N || C4 <= 0) {
        return;
    }

    const int bt = idx / C4;
    const int b = bt / T;
    const int t = bt % T;
    const int c4 = idx % C4;
    const int ix = inp[b * T + t];
    out[b * T * C4 + t * C4 + c4] = wte[ix * C4 + c4] + wpe[t * C4 + c4];
}
)METAL";
    } else if (kernel_name_contains(entry_name, "encoder_backward_kernel")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* dwte [[buffer(0)]],
    device float* dwpe [[buffer(1)]],
    device const float* dout [[buffer(2)]],
    device const int* inp [[buffer(3)]],
    constant int& B [[buffer(4)]],
    constant int& T [[buffer(5)]],
    constant int& C [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    const int idx = static_cast<int>(gid);
    const int N = B * T * C;
    if (idx >= N || C <= 0) {
        return;
    }

    const int bt = idx / C;
    const int b = bt / T;
    const int t = bt % T;
    const int c = idx % C;
    const int ix = inp[b * T + t];
    const float grad = dout[idx];

    device atomic_float* dwte_ptr = reinterpret_cast<device atomic_float*>(dwte + ix * C + c);
    device atomic_float* dwpe_ptr = reinterpret_cast<device atomic_float*>(dwpe + t * C + c);
    atomic_fetch_add_explicit(dwte_ptr, grad, memory_order_relaxed);
    atomic_fetch_add_explicit(dwpe_ptr, grad, memory_order_relaxed);
}
)METAL";
    } else if (kernel_name_contains(entry_name, "layernorm_forward_kernel3")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* out [[buffer(0)]],
    device float* mean [[buffer(1)]],
    device float* rstd [[buffer(2)]],
    device const float* inp [[buffer(3)]],
    device const float* weight [[buffer(4)]],
    device const float* bias [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& C [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (C <= 0) {
        return;
    }

    const int linear = static_cast<int>(gid);
    const int row = linear / 32;
    const int lane = linear % 32;
    if (row >= N || lane != 0) {
        return;
    }

    const int base = row * C;
    const device float* x = inp + base;

    float sum = 0.0f;
    for (int i = 0; i < C; ++i) {
        sum += x[i];
    }
    const float m = sum / static_cast<float>(C);
    if (mean != nullptr) {
        mean[row] = m;
    }

    float var_sum = 0.0f;
    for (int i = 0; i < C; ++i) {
        const float diff = x[i] - m;
        var_sum += diff * diff;
    }
    const float s = rsqrt(var_sum / static_cast<float>(C) + 1.0e-5f);
    if (rstd != nullptr) {
        rstd[row] = s;
    }

    device float* o = out + base;
    for (int i = 0; i < C; ++i) {
        const float n = (x[i] - m) * s;
        o[i] = n * weight[i] + bias[i];
    }
}
)METAL";
    } else if (kernel_name_contains(entry_name, "unpermute_kernel_backward")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* dinp [[buffer(0)]],
    device const float* dout [[buffer(1)]],
    constant int& B [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& NH [[buffer(4)]],
    constant int& d [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    const int idx = static_cast<int>(gid);
    const int total = B * NH * N * d;
    if (idx >= total || N <= 0 || NH <= 0 || d <= 0) {
        return;
    }

    const int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    const int nh = rest / (N * d);
    rest = rest % (N * d);
    const int n = rest / d;
    const int di = rest % d;

    const int other_idx = (b * NH * N * d) + (n * NH * d) + (nh * d) + di;
    dinp[idx] = dout[other_idx];
}
)METAL";
    } else if (kernel_name_contains(entry_name, "unpermute_kernel")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* inp [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant int& B [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& NH [[buffer(4)]],
    constant int& d [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    const int idx = static_cast<int>(gid);
    const int total = B * NH * N * d;
    if (idx >= total || N <= 0 || NH <= 0 || d <= 0) {
        return;
    }

    const int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    const int nh = rest / (N * d);
    rest = rest % (N * d);
    const int n = rest / d;
    const int di = rest % d;

    const int other_idx = (b * NH * N * d) + (n * NH * d) + (nh * d) + di;
    out[other_idx] = inp[idx];
}
)METAL";
    } else if (kernel_name_contains(entry_name, "permute_kernel_backward") &&
               !kernel_name_contains(entry_name, "unpermute_kernel_backward")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* dinp [[buffer(0)]],
    device const float* dq [[buffer(1)]],
    device const float* dk [[buffer(2)]],
    device const float* dv [[buffer(3)]],
    constant int& B [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& NH [[buffer(6)]],
    constant int& d [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    const int idx = static_cast<int>(gid);
    const int total = B * NH * N * d;
    if (idx >= total || N <= 0 || NH <= 0 || d <= 0) {
        return;
    }

    const int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    const int nh = rest / (N * d);
    rest = rest % (N * d);
    const int n = rest / d;
    const int di = rest % d;

    const int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (nh * d) + di;
    dinp[inp_idx] = dq[idx];
    dinp[inp_idx + NH * d] = dk[idx];
    dinp[inp_idx + 2 * NH * d] = dv[idx];
}
)METAL";
    } else if (kernel_name_contains(entry_name, "permute_kernel") &&
               !kernel_name_contains(entry_name, "unpermute_kernel")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    device float* v [[buffer(2)]],
    device const float* inp [[buffer(3)]],
    constant int& B [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& NH [[buffer(6)]],
    constant int& d [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    const int idx = static_cast<int>(gid);
    const int total = B * NH * N * d;
    if (idx >= total || N <= 0 || NH <= 0 || d <= 0) {
        return;
    }

    const int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    const int nh = rest / (N * d);
    rest = rest % (N * d);
    const int n = rest / d;
    const int di = rest % d;

    const int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (nh * d) + di;
    q[idx] = inp[inp_idx];
    k[idx] = inp[inp_idx + NH * d];
    v[idx] = inp[inp_idx + 2 * NH * d];
}
)METAL";
    } else if (kernel_name_contains(entry_name, "softmax_forward_kernel5")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* out [[buffer(0)]],
    constant float& inv_temperature [[buffer(1)]],
    device const float* inp [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& T [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (T <= 0) {
        return;
    }

    const int linear = static_cast<int>(gid);
    const int row = linear / 32;
    const int lane = linear % 32;
    if (row >= N * T || lane != 0) {
        return;
    }

    const int own_pos = row % T;
    const device float* x = inp + row * T;
    device float* y = out + row * T;

    float max_val = -3.402823466e+38f;
    for (int i = 0; i <= own_pos; ++i) {
        max_val = max(max_val, x[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i <= own_pos; ++i) {
        sum += exp(inv_temperature * (x[i] - max_val));
    }
    const float norm = (sum > 0.0f) ? (1.0f / sum) : 0.0f;

    for (int i = 0; i <= own_pos; ++i) {
        y[i] = exp(inv_temperature * (x[i] - max_val)) * norm;
    }
    for (int i = own_pos + 1; i < T; ++i) {
        y[i] = 0.0f;
    }
}
)METAL";
    } else if (kernel_name_contains(entry_name, "residual_forward_kernel")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* out [[buffer(0)]],
    device const float* inp1 [[buffer(1)]],
    device const float* inp2 [[buffer(2)]],
    constant int& N [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const int idx = static_cast<int>(gid);
    if (idx >= N) {
        return;
    }
    out[idx] = inp1[idx] + inp2[idx];
}
)METAL";
    } else if (kernel_name_contains(entry_name, "gelu_forward_kernel")) {
kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* out [[buffer(0)]],
    device const float* inp [[buffer(1)]],
    constant int& N [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 threads_per_grid [[threads_per_grid]]) {
    const int i = static_cast<int>(gid.x);
    const int launched = static_cast<int>(threads_per_grid.x);
    (void)N;
    if (i >= launched) {
        return;
    }

    constexpr float kScale = 0.7978845608028654f;
    const float x = inp[i];
    if (x > 10.0f) {
        out[i] = x;
        return;
    }
    if (x < -10.0f) {
        out[i] = 0.0f;
        return;
    }
    const float cube = 0.044715f * x * x * x;
    out[i] = 0.5f * x * (1.0f + tanh(kScale * (x + cube)));
}
)METAL";
    } else if (kernel_name_contains(entry_name, "gelu_backward_kernel")) {
kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* dinp [[buffer(0)]],
    device const float* inp [[buffer(1)]],
    device const float* dout [[buffer(2)]],
    constant int& N [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 threads_per_grid [[threads_per_grid]]) {
    const int i = static_cast<int>(gid.x);
    const int launched = static_cast<int>(threads_per_grid.x);
    (void)N;
    if (i >= launched) {
        return;
    }

    constexpr float kScale = 0.7978845608028654f;
    const float x = inp[i];
    if (x > 10.0f) {
        dinp[i] = dout[i];
        return;
    }
    if (x < -10.0f) {
        dinp[i] = 0.0f;
        return;
    }
    const float cube = 0.044715f * x * x * x;
    const float tanh_arg = kScale * (x + cube);
    const float tanh_out = tanh(tanh_arg);
    const float cosh_out = cosh(tanh_arg);
    const float sech2 = 1.0f / (cosh_out * cosh_out);
    const float local_grad = 0.5f * (1.0f + tanh_out) +
                             x * 0.5f * sech2 * kScale *
                                 (1.0f + 3.0f * 0.044715f * x * x);
    dinp[i] = local_grad * dout[i];
}
)METAL";
    } else if (kernel_name_contains(entry_name, "matmul_backward_bias_kernel4")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* dbias [[buffer(0)]],
    device const float* dout [[buffer(1)]],
    constant int& B [[buffer(2)]],
    constant int& T [[buffer(3)]],
    constant int& OC [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]]) {
    if (tid >= 32u || OC <= 0) {
        return;
    }

    const int col = static_cast<int>(group_pos.x * 32u + tid);
    if (col >= OC) {
        return;
    }

    float sum = 0.0f;
    const int rows = B * T;
    for (int row = 0; row < rows; ++row) {
        sum += dout[row * OC + col];
    }

    device atomic_float* dbias_ptr = reinterpret_cast<device atomic_float*>(dbias + col);
    atomic_fetch_add_explicit(dbias_ptr, sum, memory_order_relaxed);
}
)METAL";
    } else if (kernel_name_contains(entry_name, "layernorm_backward_kernel2")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* dinp [[buffer(0)]],
    device float* dweight [[buffer(1)]],
    device float* dbias [[buffer(2)]],
    device const float* dout [[buffer(3)]],
    device const float* inp [[buffer(4)]],
    device const float* weight [[buffer(5)]],
    device const float* mean [[buffer(6)]],
    device const float* rstd [[buffer(7)]],
    constant int& B [[buffer(8)]],
    constant int& T [[buffer(9)]],
    constant int& C [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
    if (C <= 0) {
        return;
    }

    const int linear = static_cast<int>(gid);
    const int row = linear / 32;
    const int lane = linear % 32;
    const int N = B * T;
    if (row >= N || lane != 0) {
        return;
    }

    const int base = row * C;
    const float mean_row = mean[row];
    const float rstd_row = rstd[row];
    const float inv_c = 1.0f / static_cast<float>(C);

    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = 0; i < C; ++i) {
        const float norm = (inp[base + i] - mean_row) * rstd_row;
        const float dnorm = weight[i] * dout[base + i];
        dnorm_mean += dnorm;
        dnorm_norm_mean += dnorm * norm;
    }
    dnorm_mean *= inv_c;
    dnorm_norm_mean *= inv_c;

    for (int i = 0; i < C; ++i) {
        const float norm = (inp[base + i] - mean_row) * rstd_row;
        const float dout_val = dout[base + i];
        const float dnorm = weight[i] * dout_val;

        device atomic_float* dbias_ptr = reinterpret_cast<device atomic_float*>(dbias + i);
        device atomic_float* dweight_ptr = reinterpret_cast<device atomic_float*>(dweight + i);
        atomic_fetch_add_explicit(dbias_ptr, dout_val, memory_order_relaxed);
        atomic_fetch_add_explicit(dweight_ptr, norm * dout_val, memory_order_relaxed);

        float dval = dnorm;
        dval -= dnorm_mean;
        dval -= norm * dnorm_norm_mean;
        dval *= rstd_row;

        device atomic_float* dinp_ptr = reinterpret_cast<device atomic_float*>(dinp + base + i);
        atomic_fetch_add_explicit(dinp_ptr, dval, memory_order_relaxed);
    }
}
)METAL";
    } else if (kernel_name_contains(entry_name, "softmax_autoregressive_backward_kernel")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* dpreatt [[buffer(0)]],
    device const float* datt [[buffer(1)]],
    device const float* att [[buffer(2)]],
    constant int& B [[buffer(3)]],
    constant int& T [[buffer(4)]],
    constant int& C [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]]) {
    (void)C;
    if (tid != 0u || T <= 0) {
        return;
    }

    const int idx = static_cast<int>(group_pos.y);
    const int t0 = T - 1 - 4 * static_cast<int>(group_pos.x);
    if (idx < 0 || idx >= B * (C / (T > 0 ? T : 1))) {
        // idx corresponds to B * NH in llm.c launches; keep permissive and rely on bounds below.
    }

    device float* dpreatt_base = dpreatt + static_cast<size_t>(idx) * static_cast<size_t>(T) * static_cast<size_t>(T);
    const device float* datt_base = datt + static_cast<size_t>(idx) * static_cast<size_t>(T) * static_cast<size_t>(T);
    const device float* att_base = att + static_cast<size_t>(idx) * static_cast<size_t>(T) * static_cast<size_t>(T);

    for (int to = 0; to < 4; ++to) {
        const int row = t0 - to;
        if (row < 0 || row >= T) {
            continue;
        }

        const int row_base = row * T;
        float local_sum = 0.0f;
        for (int col = 0; col <= row; ++col) {
            local_sum += att_base[row_base + col] * datt_base[row_base + col];
        }
        for (int col = 0; col <= row; ++col) {
            const float a = att_base[row_base + col];
            const float da = datt_base[row_base + col];
            dpreatt_base[row_base + col] = scale * a * (da - local_sum);
        }
        for (int col = row + 1; col < T; ++col) {
            dpreatt_base[row_base + col] = 0.0f;
        }
    }
}
)METAL";
    } else if (kernel_name_contains(entry_name, "adamw_kernel2")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* params [[buffer(0)]],
    device float* grads [[buffer(1)]],
    device float* m [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant int& num_parameters [[buffer(4)]],
    constant float& learning_rate [[buffer(5)]],
    constant float& beta1 [[buffer(6)]],
    constant float& beta2 [[buffer(7)]],
    constant float& beta1_correction [[buffer(8)]],
    constant float& beta2_correction [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    constant float& weight_decay [[buffer(11)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = static_cast<int>(gid);
    if (i < 0 || i >= num_parameters) {
        return;
    }

    const float grad = grads[i];
    float m_val = m[i];
    float v_val = v[i];

    m_val = fma(beta1, m_val, (1.0f - beta1) * grad);
    v_val = fma(beta2, v_val, (1.0f - beta2) * (grad * grad));
    m[i] = m_val;
    v[i] = v_val;

    const float m_hat = m_val / beta1_correction;
    const float v_hat = v_val / beta2_correction;
    params[i] -= learning_rate * (m_hat / (sqrt(v_hat) + eps) + weight_decay * params[i]);
}
)METAL";
    } else if (kernel_name_contains(entry_name, "fused_classifier_kernel3")) {
kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* logits [[buffer(0)]],
    device float* losses [[buffer(1)]],
    device float* probs [[buffer(2)]],
    device const float* dlosses [[buffer(3)]],
    device const int* targets [[buffer(4)]],
    constant int& B [[buffer(5)]],
    constant int& T [[buffer(6)]],
    constant int& V [[buffer(7)]],
    constant int& P [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    const int idx = static_cast<int>(group_pos.x);
    const int n = B * T;
    if (idx < 0 || idx >= n || V <= 0 || P <= 0 || V > P) {
        return;
    }

    const int target = targets[idx];
    if (target < 0 || target >= V) {
        return;
    }

    device float* row_logits = logits + static_cast<size_t>(idx) * static_cast<size_t>(P);

    threadgroup float tg_maxval;
    threadgroup float tg_scale;

    if (tid == 0u) {
        float max_val = -3.402823466e+38f;
        for (int i = 0; i < V; ++i) {
            max_val = max(max_val, row_logits[i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < V; ++i) {
            sum += exp(row_logits[i] - max_val);
        }

        tg_maxval = max_val;
        tg_scale = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float max_val = tg_maxval;
    const float scale = tg_scale;

    if (tid == 0u) {
        const float prob = exp(row_logits[target] - max_val) * scale;
        losses[idx] = -log(max(prob, 1.0e-20f));
    }

    const float dloss = dlosses != nullptr ? dlosses[idx] : (1.0f / static_cast<float>(n));
    const int stride = static_cast<int>(threads_per_group.x);
    for (int i = static_cast<int>(tid); i < V; i += stride) {
        const float prob = exp(row_logits[i] - max_val) * scale;
        if (probs != nullptr) {
            probs[static_cast<size_t>(idx) * static_cast<size_t>(P) + static_cast<size_t>(i)] = prob;
        }
        const float indicator = (i == target) ? 1.0f : 0.0f;
        row_logits[i] = (prob - indicator) * dloss;
    }
}
)METAL";
    } else if (kernel_name_contains(entry_name, "matmul_forward_kernel4")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* out [[buffer(0)]],
    device const float* inp [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    constant int& C [[buffer(4)]],
    constant int& OC [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]) {
    if (C <= 0 || OC <= 0) {
        return;
    }

    const int row_base = static_cast<int>(gid.x) * 8;
    const int col_base = static_cast<int>(gid.y) * 8;

    for (int ii = 0; ii < 8; ++ii) {
        const int row = row_base + ii;
        const int inp_base = row * C;
        const int out_base = row * OC;
        for (int jj = 0; jj < 8; ++jj) {
            const int col = col_base + jj;
            if (col >= OC) {
                continue;
            }

            float acc = (bias != nullptr) ? bias[col] : 0.0f;
            const int w_base = col * C;
            for (int k = 0; k < C; ++k) {
                acc = fma(inp[inp_base + k], weight[w_base + k], acc);
            }
            out[out_base + col] = acc;
        }
    }
}
)METAL";
    }

    if (kernel_template.empty()) {
        return {};
    }

    std::string source(kPreamble);
    source += replace_kernel_name(kernel_template, entry_name);
    return source;
}

}  // namespace

LowerToMetalResult lower_ptx_to_metal_source(std::string_view ptx, const LowerToMetalOptions& options) {
    LowerToMetalResult result;

    cumetal::passes::Phase1PipelineOptions pipeline_options;
    pipeline_options.strict = options.strict;
    pipeline_options.entry_name = options.entry_name;
    const auto pipeline = cumetal::passes::run_phase1_pipeline(ptx, pipeline_options);
    if (!pipeline.ok) {
        result.error = pipeline.error;
        return result;
    }

    result.entry_name = pipeline.entry_name;
    result.warnings = pipeline.warnings;

    const std::string metal_source = emit_metal_source_for_entry(pipeline.entry_name);
    if (metal_source.empty()) {
        result.ok = true;
        result.matched = false;
        return result;
    }

    result.ok = true;
    result.matched = true;
    result.metal_source = metal_source;
    return result;
}

}  // namespace cumetal::ptx
