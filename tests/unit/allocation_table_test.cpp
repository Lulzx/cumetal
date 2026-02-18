#include "allocation_table.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

namespace {

class FakeBuffer final : public cumetal::metal_backend::Buffer {
public:
    FakeBuffer(void* base, std::size_t size) : base_(base), size_(size) {}

    void* contents() const override {
        return base_;
    }

    std::size_t length() const override {
        return size_;
    }

private:
    void* base_;
    std::size_t size_;
};

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    cumetal::rt::AllocationTable table;

    std::uintptr_t memory[256]{};
    void* base = &memory[0];
    auto buffer = std::make_shared<FakeBuffer>(base, sizeof(memory));

    std::string error;
    if (!expect(table.insert(base, sizeof(memory), buffer, &error), "insert base allocation")) {
        return 1;
    }

    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!expect(table.resolve(base, &resolved), "resolve base pointer")) {
        return 1;
    }
    if (!expect(resolved.offset == 0, "base resolve offset is zero")) {
        return 1;
    }
    if (!expect(resolved.remaining_size == sizeof(memory), "base remaining size")) {
        return 1;
    }

    void* offset_ptr = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(base) + 64);
    if (!expect(table.resolve(offset_ptr, &resolved), "resolve offset pointer")) {
        return 1;
    }
    if (!expect(resolved.offset == 64, "offset resolve offset bytes")) {
        return 1;
    }

    auto overlap_buffer = std::make_shared<FakeBuffer>(offset_ptr, 32);
    error.clear();
    if (!expect(!table.insert(offset_ptr, 32, overlap_buffer, &error), "reject overlapping insert")) {
        return 1;
    }
    if (!expect(!error.empty(), "overlap insert returns error message")) {
        return 1;
    }

    void* miss_ptr = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(base) + sizeof(memory) + 8);
    if (!expect(!table.resolve(miss_ptr, &resolved), "resolve miss pointer")) {
        return 1;
    }

    if (!expect(table.erase(base), "erase existing allocation")) {
        return 1;
    }
    if (!expect(!table.resolve(base, &resolved), "resolve after erase fails")) {
        return 1;
    }

    if (!expect(!table.erase(base), "erase missing allocation returns false")) {
        return 1;
    }

    std::printf("PASS: allocation table unit tests\n");
    return 0;
}
