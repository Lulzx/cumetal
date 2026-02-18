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

}  // namespace

int main() {
    const std::string sample_ptx = R"PTX(
// vector ops
.version 8.0
.target sm_90
.address_size 64

.visible .entry vector_add(
    .param .u64 vector_add_param_0,
    .param .u64 vector_add_param_1,
    .param .u64 vector_add_param_2
)
{
    .reg .pred %p1;
    ld.param.u64 %rd1, [vector_add_param_0];
    ld.param.u64 %rd2, [vector_add_param_1];
    add.s32 %r1, %r2, %r3;
    @%p1 bra L_done;
L_done:
    ret;
}

/* second entry */
.visible .entry scale(
    .param .u64 scale_param_0,
    .param .f32 scale_param_1
)
{
    mul.f32 %f2, %f0, %f1;
    ret;
}
)PTX";

    const auto parsed = cumetal::ptx::parse_ptx(sample_ptx);
    if (!expect(parsed.ok, "parse PTX sample")) {
        return 1;
    }
    if (!expect(parsed.module.version_major == 8 && parsed.module.version_minor == 0,
                "parse .version 8.0")) {
        return 1;
    }
    if (!expect(parsed.module.target == "sm_90", "parse .target sm_90")) {
        return 1;
    }
    if (!expect(parsed.module.entries.size() == 2, "parse two entries")) {
        return 1;
    }

    if (!expect(parsed.module.entries[0].name == "vector_add", "first entry name")) {
        return 1;
    }
    if (!expect(parsed.module.entries[0].params.size() == 3, "vector_add parameter count")) {
        return 1;
    }
    if (!expect(parsed.module.entries[0].params[0].type == ".u64", "vector_add param0 type")) {
        return 1;
    }
    if (!expect(parsed.module.entries[0].params[0].name == "vector_add_param_0",
                "vector_add param0 name")) {
        return 1;
    }
    if (!expect(parsed.module.entries[0].instructions.size() == 5,
                "vector_add instruction count")) {
        return 1;
    }
    if (!expect(parsed.module.entries[0].instructions[0].opcode == "ld.param.u64",
                "first instruction opcode")) {
        return 1;
    }
    if (!expect(parsed.module.entries[0].instructions[0].supported, "first instruction supported")) {
        return 1;
    }
    if (!expect(parsed.module.entries[0].instructions[3].predicate == "@%p1",
                "predicate captured for branch")) {
        return 1;
    }
    if (!expect(parsed.module.entries[0].instructions[3].opcode == "bra",
                "branch opcode captured")) {
        return 1;
    }

    if (!expect(parsed.module.entries[1].name == "scale", "second entry name")) {
        return 1;
    }
    if (!expect(parsed.module.entries[1].params.size() == 2, "scale parameter count")) {
        return 1;
    }
    if (!expect(parsed.module.entries[1].params[1].type == ".f32", "scale param1 type")) {
        return 1;
    }
    if (!expect(parsed.module.entries[1].instructions.size() == 2, "scale instruction count")) {
        return 1;
    }

    const auto no_entry = cumetal::ptx::parse_ptx(".version 8.0\n.target sm_90\n");
    if (!expect(!no_entry.ok, "reject module without entry")) {
        return 1;
    }
    if (!expect(no_entry.error.find("no .entry") != std::string::npos, "error message for missing entry")) {
        return 1;
    }

    const std::string unsupported_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry unsupported(
    .param .u64 p0
)
{
    foo.bar %r1, %r2;
    ret;
}
)PTX";

    const auto tolerant = cumetal::ptx::parse_ptx(unsupported_ptx);
    if (!expect(tolerant.ok, "tolerant parse accepts unknown opcode")) {
        return 1;
    }
    if (!expect(!tolerant.warnings.empty(), "tolerant parse emits warning")) {
        return 1;
    }
    if (!expect(tolerant.module.entries[0].instructions.size() == 2,
                "unsupported entry instruction count")) {
        return 1;
    }
    if (!expect(!tolerant.module.entries[0].instructions[0].supported,
                "unknown opcode marked unsupported")) {
        return 1;
    }

    cumetal::ptx::ParseOptions strict_options;
    strict_options.strict = true;
    const auto strict = cumetal::ptx::parse_ptx(unsupported_ptx, strict_options);
    if (!expect(!strict.ok, "strict parse rejects unknown opcode")) {
        return 1;
    }
    if (!expect(strict.error.find("unsupported opcode") != std::string::npos,
                "strict parse error mentions unsupported opcode")) {
        return 1;
    }

    std::printf("PASS: ptx parser unit tests\n");
    return 0;
}
