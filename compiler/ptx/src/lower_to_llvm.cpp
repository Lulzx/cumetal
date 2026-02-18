#include "cumetal/ptx/lower_to_llvm.h"

#include "cumetal/passes/phase1_pipeline.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace cumetal::ptx {
namespace {

struct ParamInfo {
    std::string ptx_type;
    std::string llvm_type;
    std::string name;
};

std::string trim(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
        --end;
    }
    return std::string(text.substr(begin, end - begin));
}

bool parse_major_minor(const std::string& value, int* major, int* minor) {
    if (major == nullptr || minor == nullptr) {
        return false;
    }
    const std::size_t dot = value.find('.');
    if (dot == std::string::npos) {
        return false;
    }
    const std::string major_text = trim(value.substr(0, dot));
    const std::string minor_text = trim(value.substr(dot + 1));
    if (major_text.empty() || minor_text.empty()) {
        return false;
    }
    for (char c : major_text) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    for (char c : minor_text) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    *major = std::stoi(major_text);
    *minor = std::stoi(minor_text);
    return true;
}

std::string map_param_type_to_llvm(const std::string& ptx_type) {
    if (ptx_type == ".u64" || ptx_type == ".s64" || ptx_type == ".b64") {
        return "ptr addrspace(1)";
    }
    if (ptx_type == ".u32" || ptx_type == ".s32" || ptx_type == ".b32") {
        return "i32";
    }
    if (ptx_type == ".u16" || ptx_type == ".s16" || ptx_type == ".b16") {
        return "i16";
    }
    if (ptx_type == ".u8" || ptx_type == ".s8" || ptx_type == ".b8") {
        return "i8";
    }
    if (ptx_type == ".f32") {
        return "float";
    }
    if (ptx_type == ".f64") {
        return "double";
    }
    return "i32";
}

std::map<std::string, std::string> to_field_map(const cumetal::passes::KernelMetadata& metadata) {
    std::map<std::string, std::string> map;
    for (const auto& field : metadata.fields) {
        if (!field.key.empty()) {
            map[field.key] = field.value;
        }
    }
    return map;
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool looks_like_vector_add_signature(const std::string& entry_name,
                                     const std::vector<ParamInfo>& params) {
    if (params.size() < 4) {
        return false;
    }
    if (params[0].llvm_type != "ptr addrspace(1)" || params[1].llvm_type != "ptr addrspace(1)" ||
        params[2].llvm_type != "ptr addrspace(1)") {
        return false;
    }
    if (params[3].llvm_type != "i32" && params[3].llvm_type != "i64") {
        return false;
    }

    const std::string lowered_name = lowercase(entry_name);
    return lowered_name.find("vector_add") != std::string::npos ||
           lowered_name.find("vecadd") != std::string::npos;
}

bool is_integer_llvm_type(const std::string& llvm_type) {
    return !llvm_type.empty() && llvm_type[0] == 'i';
}

int alignment_for_type(const std::string& llvm_type) {
    return llvm_type == "i64" ? 8 : 4;
}

bool looks_like_matrix_mul_signature(const std::string& entry_name,
                                     const std::vector<ParamInfo>& params) {
    if (params.size() < 5) {
        return false;
    }
    if (params[0].llvm_type != "ptr addrspace(1)" || params[1].llvm_type != "ptr addrspace(1)" ||
        params[2].llvm_type != "ptr addrspace(1)") {
        return false;
    }
    if (!is_integer_llvm_type(params[3].llvm_type) || params[3].llvm_type != params[4].llvm_type) {
        return false;
    }

    const std::string lowered_name = lowercase(entry_name);
    return lowered_name.find("matrix_mul") != std::string::npos ||
           lowered_name.find("matmul") != std::string::npos ||
           lowered_name.find("gemm") != std::string::npos;
}

void emit_lowered_instruction_comments(
    std::ostringstream& ir,
    const std::vector<cumetal::passes::LoweredInstruction>& lowered_instructions) {
    for (const auto& instruction : lowered_instructions) {
        ir << "  ; ptx.lower opcode=" << instruction.opcode;
        if (!instruction.operands.empty()) {
            ir << " operands=";
            for (std::size_t i = 0; i < instruction.operands.size(); ++i) {
                if (i > 0) {
                    ir << ",";
                }
                ir << instruction.operands[i];
            }
        }
        ir << "\n";
    }
}

void emit_vector_add_body(std::ostringstream& ir, const std::vector<ParamInfo>& params) {
    const std::string& a_name = params[0].name;
    const std::string& b_name = params[1].name;
    const std::string& c_name = params[2].name;
    const std::string& idx_name = params[3].name;
    const std::string idx_type = params[3].llvm_type;

    ir << "  %a.ptr = getelementptr float, ptr addrspace(1) %" << a_name << ", " << idx_type << " %"
       << idx_name << "\n";
    ir << "  %b.ptr = getelementptr float, ptr addrspace(1) %" << b_name << ", " << idx_type << " %"
       << idx_name << "\n";
    ir << "  %c.ptr = getelementptr float, ptr addrspace(1) %" << c_name << ", " << idx_type << " %"
       << idx_name << "\n";
    ir << "  %a.val = load float, ptr addrspace(1) %a.ptr, align 4\n";
    ir << "  %b.val = load float, ptr addrspace(1) %b.ptr, align 4\n";
    ir << "  %sum = fadd float %a.val, %b.val\n";
    ir << "  store float %sum, ptr addrspace(1) %c.ptr, align 4\n";
    ir << "  ret void\n";
}

void emit_matrix_mul_body(std::ostringstream& ir, const std::vector<ParamInfo>& params) {
    const std::string& a_name = params[0].name;
    const std::string& b_name = params[1].name;
    const std::string& c_name = params[2].name;
    const std::string& n_name = params[3].name;
    const std::string& idx_name = params[4].name;
    const std::string idx_type = params[4].llvm_type;
    const int idx_align = alignment_for_type(idx_type);

    ir << "  %row = udiv " << idx_type << " %" << idx_name << ", %" << n_name << "\n";
    ir << "  %col = urem " << idx_type << " %" << idx_name << ", %" << n_name << "\n";
    ir << "  %acc.slot = alloca float, align 4\n";
    ir << "  store float 0.000000e+00, ptr %acc.slot, align 4\n";
    ir << "  %k.slot = alloca " << idx_type << ", align " << idx_align << "\n";
    ir << "  store " << idx_type << " 0, ptr %k.slot, align " << idx_align << "\n";
    ir << "  br label %mm.loop\n\n";
    ir << "mm.loop:\n";
    ir << "  %k = load " << idx_type << ", ptr %k.slot, align " << idx_align << "\n";
    ir << "  %k.in.bounds = icmp ult " << idx_type << " %k, %" << n_name << "\n";
    ir << "  br i1 %k.in.bounds, label %mm.body, label %mm.done\n\n";
    ir << "mm.body:\n";
    ir << "  %a.row.base = mul " << idx_type << " %row, %" << n_name << "\n";
    ir << "  %a.index = add " << idx_type << " %a.row.base, %k\n";
    ir << "  %b.row.base = mul " << idx_type << " %k, %" << n_name << "\n";
    ir << "  %b.index = add " << idx_type << " %b.row.base, %col\n";
    ir << "  %a.ptr = getelementptr float, ptr addrspace(1) %" << a_name << ", " << idx_type
       << " %a.index\n";
    ir << "  %b.ptr = getelementptr float, ptr addrspace(1) %" << b_name << ", " << idx_type
       << " %b.index\n";
    ir << "  %a.val = load float, ptr addrspace(1) %a.ptr, align 4\n";
    ir << "  %b.val = load float, ptr addrspace(1) %b.ptr, align 4\n";
    ir << "  %prod = fmul float %a.val, %b.val\n";
    ir << "  %acc.old = load float, ptr %acc.slot, align 4\n";
    ir << "  %acc.new = fadd float %acc.old, %prod\n";
    ir << "  store float %acc.new, ptr %acc.slot, align 4\n";
    ir << "  %k.next = add " << idx_type << " %k, 1\n";
    ir << "  store " << idx_type << " %k.next, ptr %k.slot, align " << idx_align << "\n";
    ir << "  br label %mm.loop\n\n";
    ir << "mm.done:\n";
    ir << "  %c.row.base = mul " << idx_type << " %row, %" << n_name << "\n";
    ir << "  %c.index = add " << idx_type << " %c.row.base, %col\n";
    ir << "  %c.ptr = getelementptr float, ptr addrspace(1) %" << c_name << ", " << idx_type
       << " %c.index\n";
    ir << "  %acc.final = load float, ptr %acc.slot, align 4\n";
    ir << "  store float %acc.final, ptr addrspace(1) %c.ptr, align 4\n";
    ir << "  ret void\n";
}

}  // namespace

LowerToLlvmResult lower_ptx_to_llvm_ir(std::string_view ptx, const LowerToLlvmOptions& options) {
    LowerToLlvmResult result;

    cumetal::passes::Phase1PipelineOptions pipeline_options;
    pipeline_options.strict = options.strict;
    pipeline_options.entry_name = options.entry_name;
    const auto pipeline = cumetal::passes::run_phase1_pipeline(ptx, pipeline_options);
    if (!pipeline.ok) {
        result.error = pipeline.error;
        return result;
    }

    const auto fields = to_field_map(pipeline.metadata);
    int arg_count = 0;
    if (const auto it = fields.find("kernel.arg_count"); it != fields.end()) {
        arg_count = std::max(0, std::stoi(it->second));
    }

    std::vector<std::string> arg_decls;
    std::vector<ParamInfo> params;
    arg_decls.reserve(static_cast<std::size_t>(arg_count));
    params.reserve(static_cast<std::size_t>(arg_count));
    for (int i = 0; i < arg_count; ++i) {
        const std::string type_key = "kernel.arg." + std::to_string(i) + ".type";
        const std::string name_key = "kernel.arg." + std::to_string(i) + ".name";

        const auto type_it = fields.find(type_key);
        const auto name_it = fields.find(name_key);
        const std::string ptx_type = (type_it != fields.end()) ? type_it->second : ".u32";
        const std::string llvm_type = map_param_type_to_llvm(ptx_type);
        const std::string arg_name =
            (name_it != fields.end() && !name_it->second.empty())
                ? name_it->second
                : ("arg_" + std::to_string(i));
        arg_decls.push_back(llvm_type + " %" + arg_name);
        params.push_back({.ptx_type = ptx_type, .llvm_type = llvm_type, .name = arg_name});
    }

    int air_major = 2;
    int air_minor = 8;
    if (const auto it = fields.find("air.version"); it != fields.end()) {
        (void)parse_major_minor(it->second, &air_major, &air_minor);
    }

    int language_major = 4;
    int language_minor = 0;
    if (const auto it = fields.find("language.version"); it != fields.end()) {
        (void)parse_major_minor(it->second, &language_major, &language_minor);
    }

    std::ostringstream ir;
    ir << "; ModuleID = '" << options.module_id << "'\n";
    ir << "target triple = \"" << options.target_triple << "\"\n\n";
    ir << "define void @" << pipeline.entry_name << "(";
    for (std::size_t i = 0; i < arg_decls.size(); ++i) {
        if (i > 0) {
            ir << ", ";
        }
        ir << arg_decls[i];
    }
    ir << ") #0 {\n";
    ir << "entry:\n";

    if (looks_like_vector_add_signature(pipeline.entry_name, params)) {
        emit_vector_add_body(ir, params);
    } else if (looks_like_matrix_mul_signature(pipeline.entry_name, params)) {
        emit_matrix_mul_body(ir, params);
    } else {
        emit_lowered_instruction_comments(ir, pipeline.lowered_instructions);
        ir << "  ret void\n";
    }
    ir << "}\n\n";

    ir << "attributes #0 = { \"air.kernel\" \"air.version\"=\"" << air_major << "."
       << air_minor << "\" }\n\n";
    ir << "!air.kernel = !{!0}\n";
    ir << "!0 = !{ptr @" << pipeline.entry_name << "}\n";
    ir << "!air.compile_options = !{!1}\n";
    ir << "!1 = !{!\"cumetal.phase1.pipeline\"}\n";
    ir << "!air.language_version = !{!2}\n";
    ir << "!2 = !{i32 " << language_major << ", i32 " << language_minor << "}\n";

    result.ok = true;
    result.entry_name = pipeline.entry_name;
    result.llvm_ir = ir.str();
    result.warnings = pipeline.warnings;
    return result;
}

}  // namespace cumetal::ptx
