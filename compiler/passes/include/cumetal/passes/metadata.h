#pragma once

#include "cumetal/ptx/parser.h"

#include <string>
#include <vector>

namespace cumetal::passes {

struct MetadataField {
    std::string key;
    std::string value;
};

struct KernelMetadata {
    std::string kernel_name;
    std::vector<MetadataField> fields;
};

struct MetadataOptions {
    std::string air_version = "2.8";
    std::string language_version = "4.0";
};

KernelMetadata build_kernel_metadata(const cumetal::ptx::EntryFunction& entry,
                                     const MetadataOptions& options = {});

}  // namespace cumetal::passes
