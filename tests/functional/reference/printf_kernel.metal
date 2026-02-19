#include <metal_stdlib>
using namespace metal;

// Device printf via ring buffer (spec §5.3).
// The printf_buf layout:
//   buf[0] = atomic write-word-count (initial 0; increments by record_words per call)
//   buf[1..capacity-1] = records: [fmt_id: u32][n_args: u32][arg0: u32]...[argN: u32]
// Overflow: if write position + record_size >= capacity, the record is silently dropped.
// The CPU drains the buffer after kernel completion and formats the strings.

static void cumetal_printf_2u(device atomic_uint* buf,
                              uint capacity_words,
                              uint fmt_id,
                              uint arg0,
                              uint arg1) {
    const uint record_words = 4;  // fmt_id + n_args + arg0 + arg1
    uint pos = atomic_fetch_add_explicit(buf, record_words, memory_order_relaxed);
    if (pos + record_words >= capacity_words) {
        return;  // buffer overflow: silently drop
    }
    device uint* words = (device uint*)buf;
    words[pos + 1] = fmt_id;
    words[pos + 2] = 2;  // n_args
    words[pos + 3] = arg0;
    words[pos + 4] = arg1;
}

kernel void printf_kernel(device const float* input     [[buffer(0)]],
                          device float*       output    [[buffer(1)]],
                          device atomic_uint* printf_buf [[buffer(2)]],
                          constant uint&      buf_cap_words [[buffer(3)]],
                          uint tid  [[thread_position_in_grid]]) {
    const float val = input[tid];
    const float result = val * 2.0f + 1.0f;
    output[tid] = result;

    // Log: format_id=0 → "thread %u: %.2f\n" (2 args: tid, result-as-bits)
    cumetal_printf_2u(printf_buf, buf_cap_words, 0, tid, as_type<uint>(result));
}
