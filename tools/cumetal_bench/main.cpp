#include "cuda_runtime.h"
#include "metal_backend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kThreadsPerBlock = 256;

struct Options {
    std::string metallib_path = "tests/air_abi/reference/reference.metallib";
    std::string kernel_name = "vector_add";
    std::size_t element_count = 1u << 18;
    int warmup_iterations = 5;
    int measure_iterations = 50;
};

void print_usage(const char* argv0) {
    std::printf(
        "usage: %s [--metallib <path>] [--kernel <name>] [--elements <n>] [--warmup <n>] "
        "[--iterations <n>]\n",
        argv0);
}

bool parse_positive_int(const std::string& text, int* out_value) {
    if (out_value == nullptr) {
        return false;
    }
    if (text.empty()) {
        return false;
    }
    char* end = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (end == nullptr || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    *out_value = static_cast<int>(parsed);
    return true;
}

bool parse_positive_size(const std::string& text, std::size_t* out_value) {
    if (out_value == nullptr) {
        return false;
    }
    if (text.empty()) {
        return false;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (end == nullptr || *end != '\0' || parsed == 0) {
        return false;
    }
    *out_value = static_cast<std::size_t>(parsed);
    return true;
}

bool parse_options(int argc, char** argv, Options* out_options, bool* out_show_help) {
    if (out_options == nullptr || out_show_help == nullptr) {
        return false;
    }

    *out_show_help = false;
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            *out_show_help = true;
            *out_options = options;
            return true;
        }

        if (arg == "--metallib") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "FAIL: missing value for --metallib\n");
                return false;
            }
            options.metallib_path = argv[++i];
            continue;
        }
        if (arg.rfind("--metallib=", 0) == 0) {
            options.metallib_path = arg.substr(std::strlen("--metallib="));
            continue;
        }

        if (arg == "--kernel") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "FAIL: missing value for --kernel\n");
                return false;
            }
            options.kernel_name = argv[++i];
            continue;
        }
        if (arg.rfind("--kernel=", 0) == 0) {
            options.kernel_name = arg.substr(std::strlen("--kernel="));
            continue;
        }

        std::string value;
        if (arg == "--elements") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "FAIL: missing value for --elements\n");
                return false;
            }
            value = argv[++i];
            if (!parse_positive_size(value, &options.element_count)) {
                std::fprintf(stderr, "FAIL: invalid --elements value: %s\n", value.c_str());
                return false;
            }
            continue;
        }
        if (arg.rfind("--elements=", 0) == 0) {
            value = arg.substr(std::strlen("--elements="));
            if (!parse_positive_size(value, &options.element_count)) {
                std::fprintf(stderr, "FAIL: invalid --elements value: %s\n", value.c_str());
                return false;
            }
            continue;
        }

        if (arg == "--warmup") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "FAIL: missing value for --warmup\n");
                return false;
            }
            value = argv[++i];
            if (!parse_positive_int(value, &options.warmup_iterations)) {
                std::fprintf(stderr, "FAIL: invalid --warmup value: %s\n", value.c_str());
                return false;
            }
            continue;
        }
        if (arg.rfind("--warmup=", 0) == 0) {
            value = arg.substr(std::strlen("--warmup="));
            if (!parse_positive_int(value, &options.warmup_iterations)) {
                std::fprintf(stderr, "FAIL: invalid --warmup value: %s\n", value.c_str());
                return false;
            }
            continue;
        }

        if (arg == "--iterations") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "FAIL: missing value for --iterations\n");
                return false;
            }
            value = argv[++i];
            if (!parse_positive_int(value, &options.measure_iterations)) {
                std::fprintf(stderr, "FAIL: invalid --iterations value: %s\n", value.c_str());
                return false;
            }
            continue;
        }
        if (arg.rfind("--iterations=", 0) == 0) {
            value = arg.substr(std::strlen("--iterations="));
            if (!parse_positive_int(value, &options.measure_iterations)) {
                std::fprintf(stderr, "FAIL: invalid --iterations value: %s\n", value.c_str());
                return false;
            }
            continue;
        }

        std::fprintf(stderr, "FAIL: unknown argument: %s\n", arg.c_str());
        return false;
    }

    *out_options = options;
    return true;
}

std::vector<float> make_input_a(std::size_t count) {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = static_cast<float>((i * 3u) % 97u) * 0.25f;
    }
    return values;
}

std::vector<float> make_input_b(std::size_t count) {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = static_cast<float>((i * 11u + 5u) % 89u) * 0.125f;
    }
    return values;
}

bool verify_vector_add(const std::vector<float>& a,
                       const std::vector<float>& b,
                       const std::vector<float>& out) {
    constexpr float kTolerance = 1e-5f;
    if (a.size() != b.size() || a.size() != out.size()) {
        return false;
    }
    for (std::size_t i = 0; i < out.size(); ++i) {
        const float expected = a[i] + b[i];
        if (std::fabs(out[i] - expected) > kTolerance) {
            std::fprintf(stderr,
                         "FAIL: output mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(out[i]),
                         static_cast<double>(expected));
            return false;
        }
    }
    return true;
}

double to_millis(std::chrono::steady_clock::duration duration) {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(duration).count()) /
           1000.0;
}

double benchmark_runtime_path(const Options& options,
                              const std::vector<float>& host_a,
                              const std::vector<float>& host_b) {
    const std::size_t bytes = options.element_count * sizeof(float);

    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return -1.0;
    }

    void* dev_a = nullptr;
    void* dev_b = nullptr;
    void* dev_c = nullptr;

    if (cudaMalloc(&dev_a, bytes) != cudaSuccess || cudaMalloc(&dev_b, bytes) != cudaSuccess ||
        cudaMalloc(&dev_c, bytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return -1.0;
    }

    if (cudaMemcpy(dev_a, host_a.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_b, host_b.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: runtime host->device copy failed\n");
        (void)cudaFree(dev_a);
        (void)cudaFree(dev_b);
        (void)cudaFree(dev_c);
        return -1.0;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
    };

    const cumetalKernel_t kernel{
        .metallib_path = options.metallib_path.c_str(),
        .kernel_name = options.kernel_name.c_str(),
        .arg_count = 3,
        .arg_info = kArgInfo,
    };

    void* arg_a = dev_a;
    void* arg_b = dev_b;
    void* arg_c = dev_c;
    void* launch_args[] = {&arg_a, &arg_b, &arg_c};

    const dim3 block_dim(static_cast<unsigned int>(kThreadsPerBlock), 1, 1);
    const dim3 grid_dim(static_cast<unsigned int>((options.element_count + kThreadsPerBlock - 1) /
                                                   kThreadsPerBlock),
                        1,
                        1);

    for (int i = 0; i < options.warmup_iterations; ++i) {
        if (cudaLaunchKernel(&kernel, grid_dim, block_dim, launch_args, 0, nullptr) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "FAIL: runtime warmup launch failed\n");
            (void)cudaFree(dev_a);
            (void)cudaFree(dev_b);
            (void)cudaFree(dev_c);
            return -1.0;
        }
    }

    std::chrono::steady_clock::duration total = std::chrono::steady_clock::duration::zero();
    for (int i = 0; i < options.measure_iterations; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        if (cudaLaunchKernel(&kernel, grid_dim, block_dim, launch_args, 0, nullptr) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "FAIL: runtime measured launch failed\n");
            (void)cudaFree(dev_a);
            (void)cudaFree(dev_b);
            (void)cudaFree(dev_c);
            return -1.0;
        }
        total += (std::chrono::steady_clock::now() - begin);
    }

    std::vector<float> host_out(options.element_count, 0.0f);
    if (cudaMemcpy(host_out.data(), dev_c, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: runtime device->host copy failed\n");
        (void)cudaFree(dev_a);
        (void)cudaFree(dev_b);
        (void)cudaFree(dev_c);
        return -1.0;
    }

    if (!verify_vector_add(host_a, host_b, host_out)) {
        (void)cudaFree(dev_a);
        (void)cudaFree(dev_b);
        (void)cudaFree(dev_c);
        return -1.0;
    }

    if (cudaFree(dev_a) != cudaSuccess || cudaFree(dev_b) != cudaSuccess || cudaFree(dev_c) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return -1.0;
    }

    return to_millis(total) / static_cast<double>(options.measure_iterations);
}

double benchmark_native_path(const Options& options,
                             const std::vector<float>& host_a,
                             const std::vector<float>& host_b) {
    std::string error;
    if (cumetal::metal_backend::initialize(&error) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: metal backend initialize failed: %s\n", error.c_str());
        return -1.0;
    }

    const std::size_t bytes = options.element_count * sizeof(float);
    std::shared_ptr<cumetal::metal_backend::Buffer> buffer_a;
    std::shared_ptr<cumetal::metal_backend::Buffer> buffer_b;
    std::shared_ptr<cumetal::metal_backend::Buffer> buffer_c;
    if (cumetal::metal_backend::allocate_buffer(bytes, &buffer_a, &error) != cudaSuccess ||
        cumetal::metal_backend::allocate_buffer(bytes, &buffer_b, &error) != cudaSuccess ||
        cumetal::metal_backend::allocate_buffer(bytes, &buffer_c, &error) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: native buffer allocation failed: %s\n", error.c_str());
        return -1.0;
    }

    std::memcpy(buffer_a->contents(), host_a.data(), bytes);
    std::memcpy(buffer_b->contents(), host_b.data(), bytes);
    std::memset(buffer_c->contents(), 0, bytes);

    cumetal::metal_backend::LaunchConfig config{
        .grid = dim3(static_cast<unsigned int>((options.element_count + kThreadsPerBlock - 1) /
                                               kThreadsPerBlock),
                     1,
                     1),
        .block = dim3(static_cast<unsigned int>(kThreadsPerBlock), 1, 1),
        .shared_memory_bytes = 0,
    };

    std::vector<cumetal::metal_backend::KernelArg> args(3);
    args[0].kind = cumetal::metal_backend::KernelArg::Kind::kBuffer;
    args[0].buffer = buffer_a;
    args[1].kind = cumetal::metal_backend::KernelArg::Kind::kBuffer;
    args[1].buffer = buffer_b;
    args[2].kind = cumetal::metal_backend::KernelArg::Kind::kBuffer;
    args[2].buffer = buffer_c;

    for (int i = 0; i < options.warmup_iterations; ++i) {
        if (cumetal::metal_backend::launch_kernel(
                options.metallib_path, options.kernel_name, config, args, nullptr, &error) != cudaSuccess ||
            cumetal::metal_backend::synchronize(&error) != cudaSuccess) {
            std::fprintf(stderr, "FAIL: native warmup launch failed: %s\n", error.c_str());
            return -1.0;
        }
    }

    std::chrono::steady_clock::duration total = std::chrono::steady_clock::duration::zero();
    for (int i = 0; i < options.measure_iterations; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        if (cumetal::metal_backend::launch_kernel(
                options.metallib_path, options.kernel_name, config, args, nullptr, &error) != cudaSuccess ||
            cumetal::metal_backend::synchronize(&error) != cudaSuccess) {
            std::fprintf(stderr, "FAIL: native measured launch failed: %s\n", error.c_str());
            return -1.0;
        }
        total += (std::chrono::steady_clock::now() - begin);
    }

    const float* out = static_cast<const float*>(buffer_c->contents());
    std::vector<float> host_out(options.element_count);
    std::memcpy(host_out.data(), out, bytes);
    if (!verify_vector_add(host_a, host_b, host_out)) {
        return -1.0;
    }

    return to_millis(total) / static_cast<double>(options.measure_iterations);
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    bool show_help = false;
    if (!parse_options(argc, argv, &options, &show_help)) {
        return 64;
    }
    if (show_help) {
        return 0;
    }

    if (!std::filesystem::exists(options.metallib_path)) {
        std::fprintf(stderr, "FAIL: metallib not found: %s\n", options.metallib_path.c_str());
        return 64;
    }

    const std::vector<float> host_a = make_input_a(options.element_count);
    const std::vector<float> host_b = make_input_b(options.element_count);

    const double native_ms = benchmark_native_path(options, host_a, host_b);
    if (native_ms <= 0.0) {
        return 1;
    }

    const double runtime_ms = benchmark_runtime_path(options, host_a, host_b);
    if (runtime_ms <= 0.0) {
        return 1;
    }

    const double ratio = runtime_ms / native_ms;
    std::printf("cumetal_bench results\n");
    std::printf("kernel: %s\n", options.kernel_name.c_str());
    std::printf("elements: %zu\n", options.element_count);
    std::printf("warmup_iterations: %d\n", options.warmup_iterations);
    std::printf("measure_iterations: %d\n", options.measure_iterations);
    std::printf("native_metal_avg_ms: %.4f\n", native_ms);
    std::printf("cumetal_runtime_avg_ms: %.4f\n", runtime_ms);
    std::printf("runtime_vs_native_ratio: %.3fx\n", ratio);
    return 0;
}
