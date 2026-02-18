kernel void saxpy(device const float* x [[buffer(0)]],
                  device const float* y [[buffer(1)]],
                  device float* out [[buffer(2)]],
                  constant float& alpha [[buffer(3)]],
                  uint id [[thread_position_in_grid]]) {
    out[id] = alpha * x[id] + y[id];
}
