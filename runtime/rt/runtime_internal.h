#pragma once

#include "allocation_table.h"

namespace cumetal::rt {

bool resolve_allocation_for_pointer(const void* ptr, AllocationTable::ResolvedAllocation* out);

}  // namespace cumetal::rt
