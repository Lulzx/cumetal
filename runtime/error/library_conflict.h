#pragma once

#include <string>
#include <vector>

namespace cumetal::error {

std::string detect_libcuda_conflict_from_paths(const std::vector<std::string>& image_paths,
                                               const std::string& self_image_path);

std::string detect_loaded_libcuda_conflict(const void* self_symbol);

}  // namespace cumetal::error
