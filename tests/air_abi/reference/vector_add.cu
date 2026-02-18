extern "C" __global__ void vector_add(float* a, float* b, float* c, int id) {
    c[id] = a[id] + b[id];
}
