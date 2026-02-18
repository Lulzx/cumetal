#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace cumetal::cache {

bool stage_metallib_bytes(const void* image,
                          std::size_t size,
                          std::filesystem::path* out_path,
                          std::string* error_message);

}  // namespace cumetal::cache
