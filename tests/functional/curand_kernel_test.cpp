#include <curand_kernel.h>
#include <cstdio>
#include <cmath>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
    else { printf("PASS: %s\n", msg); } \
} while(0)

static void test_xorwow_init_and_generate() {
    curandState_t state;
    curand_init(12345ULL, 0ULL, 0ULL, &state);
    unsigned int val = curand(&state);
    CHECK(val != 0, "curand XORWOW generates non-zero");

    // Generate 1000 values, check they're not all the same
    unsigned int first = curand(&state);
    bool all_same = true;
    for (int i = 0; i < 999; i++) {
        if (curand(&state) != first) { all_same = false; break; }
    }
    CHECK(!all_same, "curand XORWOW produces varied output");
}

static void test_uniform_range() {
    curandState_t state;
    curand_init(42ULL, 0ULL, 0ULL, &state);

    bool in_range = true;
    for (int i = 0; i < 10000; i++) {
        float u = curand_uniform(&state);
        if (u < 0.0f || u >= 1.0f) { in_range = false; break; }
    }
    CHECK(in_range, "curand_uniform in [0, 1)");
}

static void test_uniform_mean() {
    curandState_t state;
    curand_init(123ULL, 0ULL, 0ULL, &state);

    double sum = 0;
    int N = 100000;
    for (int i = 0; i < N; i++) sum += curand_uniform(&state);
    double mean = sum / N;
    CHECK(std::fabs(mean - 0.5) < 0.01, "curand_uniform mean ~0.5");
}

static void test_normal_distribution() {
    curandState_t state;
    curand_init(456ULL, 0ULL, 0ULL, &state);

    double sum = 0, sum_sq = 0;
    int N = 100000;
    for (int i = 0; i < N; i++) {
        float n = curand_normal(&state);
        sum += n;
        sum_sq += n * n;
    }
    double mean = sum / N;
    double var = sum_sq / N - mean * mean;
    CHECK(std::fabs(mean) < 0.02, "curand_normal mean ~0");
    CHECK(std::fabs(var - 1.0) < 0.05, "curand_normal variance ~1");
}

static void test_philox_init_and_generate() {
    curandStatePhilox4_32_10_t state;
    curand_init(789ULL, 0ULL, 0ULL, &state);

    unsigned int val = curand(&state);
    CHECK(val != 0, "curand Philox generates non-zero");

    float u = curand_uniform(&state);
    CHECK(u >= 0.0f && u < 1.0f, "curand_uniform Philox in [0, 1)");
}

static void test_different_seeds_different_output() {
    curandState_t s1, s2;
    curand_init(100ULL, 0ULL, 0ULL, &s1);
    curand_init(200ULL, 0ULL, 0ULL, &s2);

    unsigned int v1 = curand(&s1);
    unsigned int v2 = curand(&s2);
    CHECK(v1 != v2, "different seeds produce different output");
}

static void test_different_sequences_different_output() {
    curandState_t s1, s2;
    curand_init(100ULL, 0ULL, 0ULL, &s1);
    curand_init(100ULL, 1ULL, 0ULL, &s2);

    unsigned int v1 = curand(&s1);
    unsigned int v2 = curand(&s2);
    CHECK(v1 != v2, "different sequences produce different output");
}

static void test_log_normal() {
    curandState_t state;
    curand_init(321ULL, 0ULL, 0ULL, &state);

    bool all_positive = true;
    for (int i = 0; i < 1000; i++) {
        float val = curand_log_normal(&state, 0.0f, 1.0f);
        if (val <= 0.0f) { all_positive = false; break; }
    }
    CHECK(all_positive, "curand_log_normal all positive");
}

int main() {
    test_xorwow_init_and_generate();
    test_uniform_range();
    test_uniform_mean();
    test_normal_distribution();
    test_philox_init_and_generate();
    test_different_seeds_different_output();
    test_different_sequences_different_output();
    test_log_normal();

    printf("\n%s (%d failures)\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
