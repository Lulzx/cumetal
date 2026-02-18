kernel void vector_add(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    c[id] = a[id] + b[id];
}

kernel void scale(device float* data [[buffer(0)]],
                  constant float& alpha [[buffer(1)]],
                  uint id [[thread_position_in_grid]]) {
    data[id] = data[id] * alpha;
}
