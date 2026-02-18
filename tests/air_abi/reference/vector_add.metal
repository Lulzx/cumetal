kernel void vector_add(device float* a [[buffer(0)]],
                       device float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    c[id] = a[id] + b[id];
}
