#pragma once

#include "metal_backend.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>

namespace cumetal::rt {

enum class AllocationKind {
    kDevice,
    kHost,
};

class AllocationTable {
public:
    struct ResolvedAllocation {
        std::shared_ptr<metal_backend::Buffer> buffer;
        std::size_t offset = 0;
        std::size_t remaining_size = 0;
        AllocationKind kind = AllocationKind::kDevice;
    };

    bool insert(void* base,
                std::size_t size,
                AllocationKind kind,
                std::shared_ptr<metal_backend::Buffer> buffer,
                std::string* error_message);
    bool erase(void* base);
    bool resolve(const void* ptr, ResolvedAllocation* resolved) const;
    std::size_t total_allocated_size() const;
    void clear();

private:
    struct Entry {
        std::size_t size = 0;
        AllocationKind kind = AllocationKind::kDevice;
        std::shared_ptr<metal_backend::Buffer> buffer;
    };

    std::map<std::uintptr_t, Entry> entries_;
    mutable std::shared_mutex mutex_;
};

}  // namespace cumetal::rt
