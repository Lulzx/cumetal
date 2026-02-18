#include "cumetal/passes/printf_lower.h"
#include "cumetal/ptx/parser.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
}

bool contains_warning(const std::vector<std::string>& warnings, const std::string& needle) {
    for (const auto& warning : warnings) {
        if (warning.find(needle) != std::string::npos) {
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
.visible .entry k(
    .param .u64 p0
)
{
    call.uni (%r0), vprintf, ("tid=%d", %r1);
    call.uni (%r2), printf, ("tid=%d", %r2);
    call.uni (%r3), vprintf, ("%f", %f1);
    call.uni (%r4), foo, (%r4);
    ret;
}
)PTX";

    const auto parsed = cumetal::ptx::parse_ptx(ptx);
    if (!expect(parsed.ok, "parse PTX for printf lowering")) {
        return 1;
    }

    const auto lowered = cumetal::passes::lower_printf_calls(parsed.module.entries[0]);
    if (!expect(lowered.ok, "printf lowering succeeds")) {
        return 1;
    }
    if (!expect(lowered.calls.size() == 3, "three printf/vprintf calls lowered")) {
        return 1;
    }
    if (!expect(lowered.formats.size() == 2, "deduplicated format table has two entries")) {
        return 1;
    }
    if (!expect(lowered.calls[0].format_id == lowered.calls[1].format_id,
                "duplicate format literal reuses format id")) {
        return 1;
    }
    if (!expect(lowered.calls[0].arguments.size() == 1 &&
                    lowered.calls[0].arguments[0] == "%r1",
                "printf arguments captured after format token")) {
        return 1;
    }
    if (!expect(lowered.formats[0].literal, "first format marked literal")) {
        return 1;
    }
    if (!expect(!lowered.formats[0].truncated, "first format not truncated")) {
        return 1;
    }

    const std::string long_format(300, 'a');
    const std::string trunc_ptx =
        ".version 8.0\n"
        ".target sm_90\n"
        ".visible .entry trunc(\n"
        "    .param .u64 p0\n"
        ")\n"
        "{\n"
        "    call.uni (%r0), vprintf, (\"" +
        long_format +
        "\", %r1);\n"
        "    ret;\n"
        "}\n";

    const auto trunc_parsed = cumetal::ptx::parse_ptx(trunc_ptx);
    if (!expect(trunc_parsed.ok, "parse PTX with long format literal")) {
        return 1;
    }

    const auto trunc_lowered = cumetal::passes::lower_printf_calls(trunc_parsed.module.entries[0]);
    if (!expect(trunc_lowered.ok, "printf lowering handles long format literal")) {
        return 1;
    }
    if (!expect(trunc_lowered.formats.size() == 1, "single format in truncation test")) {
        return 1;
    }
    if (!expect(trunc_lowered.formats[0].truncated, "long format literal truncated")) {
        return 1;
    }
    if (!expect(trunc_lowered.formats[0].token.size() == 256, "format truncation obeys 256-byte limit")) {
        return 1;
    }
    if (!expect(contains_warning(trunc_lowered.warnings, "truncated"),
                "truncation warning emitted")) {
        return 1;
    }

    const std::string malformed_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry malformed(
    .param .u64 p0
)
{
    call.uni (%r0), vprintf;
    ret;
}
)PTX";

    const auto malformed_parsed = cumetal::ptx::parse_ptx(malformed_ptx);
    if (!expect(malformed_parsed.ok, "parse PTX with malformed printf call")) {
        return 1;
    }

    const auto tolerant = cumetal::passes::lower_printf_calls(malformed_parsed.module.entries[0]);
    if (!expect(tolerant.ok, "tolerant printf lowering keeps malformed call as warning")) {
        return 1;
    }
    if (!expect(contains_warning(tolerant.warnings, "missing argument tuple"),
                "missing argument tuple warning emitted")) {
        return 1;
    }

    cumetal::passes::PrintfLowerOptions strict_options;
    strict_options.strict = true;
    const auto strict = cumetal::passes::lower_printf_calls(malformed_parsed.module.entries[0], strict_options);
    if (!expect(!strict.ok, "strict printf lowering rejects malformed call")) {
        return 1;
    }
    if (!expect(strict.error.find("missing argument tuple") != std::string::npos,
                "strict error reports missing argument tuple")) {
        return 1;
    }

    std::printf("PASS: printf lower unit tests\n");
    return 0;
}
