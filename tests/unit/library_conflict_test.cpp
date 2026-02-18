#include "library_conflict.h"

#include <cstdio>
#include <string>
#include <vector>

int main() {
    {
        const std::vector<std::string> images = {
            "/usr/lib/libSystem.B.dylib",
            "/opt/cumetal/lib/libcumetal.dylib",
        };
        const std::string warning =
            cumetal::error::detect_libcuda_conflict_from_paths(images, "/opt/cumetal/lib/libcumetal.dylib");
        if (!warning.empty()) {
            std::fprintf(stderr, "FAIL: no libcuda in image list should not warn\n");
            return 1;
        }
    }

    {
        const std::vector<std::string> images = {
            "/opt/cumetal/lib/libcuda.dylib",
        };
        const std::string warning =
            cumetal::error::detect_libcuda_conflict_from_paths(images, "/opt/cumetal/lib/libcuda.dylib");
        if (!warning.empty()) {
            std::fprintf(stderr, "FAIL: self libcuda should not be treated as conflict\n");
            return 1;
        }
    }

    {
        const std::vector<std::string> images = {
            "/opt/cumetal/lib/libcumetal.dylib",
            "/usr/local/cuda/lib/libcuda.dylib",
        };
        const std::string warning =
            cumetal::error::detect_libcuda_conflict_from_paths(images, "/opt/cumetal/lib/libcumetal.dylib");
        if (warning.empty()) {
            std::fprintf(stderr, "FAIL: external libcuda should trigger conflict warning\n");
            return 1;
        }
        if (warning.find("/usr/local/cuda/lib/libcuda.dylib") == std::string::npos) {
            std::fprintf(stderr, "FAIL: warning should include conflicting image path\n");
            return 1;
        }
        if (warning.find("/opt/cumetal/lib/libcumetal.dylib") == std::string::npos) {
            std::fprintf(stderr, "FAIL: warning should include CuMetal image path\n");
            return 1;
        }
    }

    {
        const std::vector<std::string> images = {
            "/opt/cumetal/lib/libcuda.dylib",
        };
        const std::string warning =
            cumetal::error::detect_libcuda_conflict_from_paths(images, "/opt/cumetal/lib/../lib/libcuda.dylib");
        if (!warning.empty()) {
            std::fprintf(stderr, "FAIL: normalized self path should not trigger conflict\n");
            return 1;
        }
    }

    std::printf("PASS: library conflict detection logic behaves correctly\n");
    return 0;
}
