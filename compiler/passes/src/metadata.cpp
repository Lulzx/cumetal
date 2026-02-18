#include "cumetal/passes/metadata.h"

namespace cumetal::passes {

KernelMetadata build_kernel_metadata(const cumetal::ptx::EntryFunction& entry,
                                     const MetadataOptions& options) {
    KernelMetadata metadata;
    metadata.kernel_name = entry.name;
    metadata.fields.push_back({.key = "air.kernel", .value = "true"});
    metadata.fields.push_back({.key = "air.version", .value = options.air_version});
    metadata.fields.push_back({.key = "language.version", .value = options.language_version});
    metadata.fields.push_back({.key = "kernel.arg_count", .value = std::to_string(entry.params.size())});

    for (std::size_t i = 0; i < entry.params.size(); ++i) {
        metadata.fields.push_back(
            {.key = "kernel.arg." + std::to_string(i) + ".type", .value = entry.params[i].type});
        metadata.fields.push_back(
            {.key = "kernel.arg." + std::to_string(i) + ".name", .value = entry.params[i].name});
    }

    return metadata;
}

}  // namespace cumetal::passes
