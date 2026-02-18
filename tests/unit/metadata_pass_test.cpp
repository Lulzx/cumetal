#include "cumetal/passes/metadata.h"
#include "cumetal/ptx/parser.h"

#include <cstdio>
#include <string>

namespace {

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
}

bool has_field(const cumetal::passes::KernelMetadata& metadata,
               const std::string& key,
               const std::string& value) {
    for (const auto& field : metadata.fields) {
        if (field.key == key && field.value == value) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main() {
    const std::string ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry vector_add(
    .param .u64 vector_add_param_0,
    .param .u64 vector_add_param_1,
    .param .u64 vector_add_param_2
)
{
    ret;
}
)PTX";

    const auto parsed = cumetal::ptx::parse_ptx(ptx);
    if (!expect(parsed.ok, "parse PTX for metadata pass")) {
        return 1;
    }
    if (!expect(parsed.module.entries.size() == 1, "one entry parsed")) {
        return 1;
    }

    cumetal::passes::MetadataOptions options;
    options.air_version = "2.9";
    options.language_version = "4.1";
    const auto metadata = cumetal::passes::build_kernel_metadata(parsed.module.entries[0], options);

    if (!expect(metadata.kernel_name == "vector_add", "kernel name propagated")) {
        return 1;
    }
    if (!expect(has_field(metadata, "air.kernel", "true"), "air.kernel field")) {
        return 1;
    }
    if (!expect(has_field(metadata, "air.version", "2.9"), "air.version field")) {
        return 1;
    }
    if (!expect(has_field(metadata, "language.version", "4.1"), "language.version field")) {
        return 1;
    }
    if (!expect(has_field(metadata, "kernel.arg_count", "3"), "kernel arg count field")) {
        return 1;
    }
    if (!expect(has_field(metadata, "kernel.arg.0.type", ".u64"), "arg0 type field")) {
        return 1;
    }
    if (!expect(has_field(metadata, "kernel.arg.0.name", "vector_add_param_0"), "arg0 name field")) {
        return 1;
    }
    if (!expect(has_field(metadata, "kernel.arg.2.name", "vector_add_param_2"), "arg2 name field")) {
        return 1;
    }

    std::printf("PASS: metadata pass unit tests\n");
    return 0;
}
