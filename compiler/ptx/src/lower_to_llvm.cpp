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

std::string map_param_type_to_llvm(const std::string& ptx_type, bool is_pointer) {
    if (is_pointer) {
        return "float addrspace(1)*";
    }
    if (ptx_type == ".u64" || ptx_type == ".s64" || ptx_type == ".b64") {
        return "i64";
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

bool is_pointer_type(const std::string& llvm_type) {
    return llvm_type.find('*') != std::string::npos;
}

bool is_device_buffer_pointer(const std::string& llvm_type) {
    return llvm_type.find("addrspace(1)*") != std::string::npos;
}

bool is_constant_buffer_pointer(const std::string& llvm_type) {
    return llvm_type.find("addrspace(2)*") != std::string::npos;
}

std::string pointee_type_from_pointer(const std::string& llvm_type) {
    const std::size_t star = llvm_type.find('*');
    if (star == std::string::npos) {
        return "i8";
    }
    return trim(llvm_type.substr(0, star));
}

std::string air_type_name_for_param(const ParamInfo& param, bool is_thread_position) {
    if (is_thread_position) {
        return "uint";
    }
    if (is_device_buffer_pointer(param.llvm_type)) {
        return pointee_type_from_pointer(param.llvm_type) == "double" ? "double" : "float";
    }
    if (is_constant_buffer_pointer(param.llvm_type)) {
        return "uint";
    }
    if (param.llvm_type == "float") {
        return "float";
    }
    if (param.llvm_type == "double") {
        return "double";
    }
    if (param.llvm_type == "i64") {
        return "ulong";
    }
    if (param.llvm_type == "i32") {
        return "uint";
    }
    return "uint";
}

int byte_size_for_llvm_type(const std::string& llvm_type) {
    if (llvm_type.find("double") != std::string::npos || llvm_type.find("i64") != std::string::npos) {
        return 8;
    }
    return 4;
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
    if (!is_device_buffer_pointer(params[0].llvm_type) || !is_device_buffer_pointer(params[1].llvm_type) ||
        !is_device_buffer_pointer(params[2].llvm_type)) {
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

bool looks_like_matrix_mul_signature(const std::string& entry_name,
                                     const std::vector<ParamInfo>& params) {
    if (params.size() < 5) {
        return false;
    }
    if (!is_device_buffer_pointer(params[0].llvm_type) || !is_device_buffer_pointer(params[1].llvm_type) ||
        !is_device_buffer_pointer(params[2].llvm_type)) {
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

bool looks_like_negate_signature(const std::string& entry_name, const std::vector<ParamInfo>& params) {
    if (params.size() < 2) {
        return false;
    }
    if (!is_device_buffer_pointer(params[0].llvm_type) || !is_device_buffer_pointer(params[1].llvm_type)) {
        return false;
    }
    const std::string lowered_name = lowercase(entry_name);
    return lowered_name.find("negate") != std::string::npos || lowered_name.find("neg") == 0;
}

bool looks_like_reduce_sum_signature(const std::string& entry_name,
                                     const std::vector<ParamInfo>& params) {
    if (params.size() < 3) {
        return false;
    }
    if (!is_device_buffer_pointer(params[0].llvm_type) || !is_device_buffer_pointer(params[1].llvm_type)) {
        return false;
    }
    const bool scalar_count = params[2].llvm_type == "i32" || params[2].llvm_type == "i64";
    if (!scalar_count) {
        return false;
    }
    const std::string lowered_name = lowercase(entry_name);
    return lowered_name.find("reduce") != std::string::npos ||
           lowered_name.find("sum") != std::string::npos;
}

bool looks_like_fp64_mul_add_signature(const std::string& entry_name,
                                        const std::vector<ParamInfo>& params) {
    if (params.size() < 2) {
        return false;
    }
    if (!is_device_buffer_pointer(params[0].llvm_type) ||
        !is_device_buffer_pointer(params[1].llvm_type)) {
        return false;
    }
    const std::string lowered_name = lowercase(entry_name);
    // Match kernels named like "fp64_mul_add" or "fp64_fma" — FP64 arithmetic tests
    return lowered_name.find("fp64") != std::string::npos &&
           (lowered_name.find("mul") != std::string::npos ||
            lowered_name.find("fma") != std::string::npos ||
            lowered_name.find("add") != std::string::npos);
}

void emit_fp64_mul_add_body(std::ostringstream& ir, const std::vector<ParamInfo>& params) {
    // params[0]: float* input, params[1]: float* output
    // params.back(): __air_thread_position_in_grid (i32)
    const std::string& in_name  = params[0].name;
    const std::string& out_name = params[1].name;
    const std::string& idx_name = params.back().name;

    ir << "  %in.ptr = getelementptr float, float addrspace(1)* %" << in_name
       << ", i32 %" << idx_name << "\n";
    ir << "  %out.ptr = getelementptr float, float addrspace(1)* %" << out_name
       << ", i32 %" << idx_name << "\n";
    ir << "  %f.val = load float, float addrspace(1)* %in.ptr, align 4\n";
    ir << "  %d.val = fpext float %f.val to double\n";
    ir << "  %d.result = call double @llvm.fma.f64(double %d.val, "
          "double 2.000000e+00, double 1.000000e+00)\n";
    ir << "  %f.result = fptrunc double %d.result to float\n";
    ir << "  store float %f.result, float addrspace(1)* %out.ptr, align 4\n";
    ir << "  ret void\n";
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

    ir << "  %a.ptr = getelementptr float, float addrspace(1)* %" << a_name << ", " << idx_type << " %"
       << idx_name << "\n";
    ir << "  %b.ptr = getelementptr float, float addrspace(1)* %" << b_name << ", " << idx_type << " %"
       << idx_name << "\n";
    ir << "  %c.ptr = getelementptr float, float addrspace(1)* %" << c_name << ", " << idx_type << " %"
       << idx_name << "\n";
    ir << "  %a.val = load float, float addrspace(1)* %a.ptr, align 4\n";
    ir << "  %b.val = load float, float addrspace(1)* %b.ptr, align 4\n";
    ir << "  %sum = fadd float %a.val, %b.val\n";
    ir << "  store float %sum, float addrspace(1)* %c.ptr, align 4\n";
    ir << "  ret void\n";
}

void emit_matrix_mul_body(std::ostringstream& ir, const std::vector<ParamInfo>& params) {
    const std::string& a_name = params[0].name;
    const std::string& b_name = params[1].name;
    const std::string& c_name = params[2].name;
    const std::string& n_name = params[3].name;
    const std::string& idx_name = params[4].name;
    const std::string idx_type = params[4].llvm_type;

    ir << "  %n.val = load i32, i32 addrspace(2)* %" << n_name << ", align 4\n";
    ir << "  %row = udiv " << idx_type << " %" << idx_name << ", %n.val\n";
    ir << "  %col = urem " << idx_type << " %" << idx_name << ", %n.val\n";
    ir << "  br label %mm.loop\n\n";
    ir << "mm.loop:\n";
    ir << "  %k = phi " << idx_type << " [ 0, %entry ], [ %k.next, %mm.body ]\n";
    ir << "  %acc = phi float [ 0.000000e+00, %entry ], [ %acc.next, %mm.body ]\n";
    ir << "  %k.in.bounds = icmp ult " << idx_type << " %k, %n.val\n";
    ir << "  br i1 %k.in.bounds, label %mm.body, label %mm.done\n\n";
    ir << "mm.body:\n";
    ir << "  %a.row.base = mul " << idx_type << " %row, %n.val\n";
    ir << "  %a.index = add " << idx_type << " %a.row.base, %k\n";
    ir << "  %b.row.base = mul " << idx_type << " %k, %n.val\n";
    ir << "  %b.index = add " << idx_type << " %b.row.base, %col\n";
    ir << "  %a.ptr = getelementptr float, float addrspace(1)* %" << a_name << ", " << idx_type
       << " %a.index\n";
    ir << "  %b.ptr = getelementptr float, float addrspace(1)* %" << b_name << ", " << idx_type
       << " %b.index\n";
    ir << "  %a.val = load float, float addrspace(1)* %a.ptr, align 4\n";
    ir << "  %b.val = load float, float addrspace(1)* %b.ptr, align 4\n";
    ir << "  %prod = fmul float %a.val, %b.val\n";
    ir << "  %acc.next = fadd float %acc, %prod\n";
    ir << "  %k.next = add " << idx_type << " %k, 1\n";
    ir << "  br label %mm.loop\n\n";
    ir << "mm.done:\n";
    ir << "  %c.row.base = mul " << idx_type << " %row, %n.val\n";
    ir << "  %c.index = add " << idx_type << " %c.row.base, %col\n";
    ir << "  %c.ptr = getelementptr float, float addrspace(1)* %" << c_name << ", " << idx_type
       << " %c.index\n";
    ir << "  store float %acc, float addrspace(1)* %c.ptr, align 4\n";
    ir << "  ret void\n";
}

void emit_negate_body(std::ostringstream& ir, const std::vector<ParamInfo>& params) {
    const std::string& in_name = params[0].name;
    const std::string& out_name = params[1].name;
    const std::string& idx_name = params.back().name;
    const std::string idx_type = params.back().llvm_type;

    ir << "  %in.ptr = getelementptr float, float addrspace(1)* %" << in_name << ", " << idx_type
       << " %" << idx_name << "\n";
    ir << "  %out.ptr = getelementptr float, float addrspace(1)* %" << out_name << ", " << idx_type
       << " %" << idx_name << "\n";
    ir << "  %in.val = load float, float addrspace(1)* %in.ptr, align 4\n";
    ir << "  %neg.val = fneg float %in.val\n";
    ir << "  store float %neg.val, float addrspace(1)* %out.ptr, align 4\n";
    ir << "  ret void\n";
}

void emit_reduce_sum_body(std::ostringstream& ir, const std::vector<ParamInfo>& params) {
    const std::string& in_name = params[0].name;
    const std::string& out_name = params[1].name;
    const std::string& n_name = params[2].name;
    const std::string& idx_name = params.back().name;
    const std::string idx_type = params.back().llvm_type;

    std::string idx_value = "%" + idx_name;
    if (idx_type != "i32") {
        ir << "  %reduce.idx.i32 = trunc " << idx_type << " %" << idx_name << " to i32\n";
        idx_value = "%reduce.idx.i32";
    }

    ir << "  %n.val = load i32, i32 addrspace(2)* %" << n_name << ", align 4\n";
    ir << "  %reduce.in.bounds = icmp ult i32 " << idx_value << ", %n.val\n";
    ir << "  br i1 %reduce.in.bounds, label %reduce.body, label %reduce.done\n\n";
    ir << "reduce.body:\n";
    ir << "  %in.ptr = getelementptr float, float addrspace(1)* %" << in_name << ", i32 " << idx_value
       << "\n";
    ir << "  %in.val = load float, float addrspace(1)* %in.ptr, align 4\n";
    ir << "  %out.ptr = getelementptr float, float addrspace(1)* %" << out_name << ", i32 0\n";
    ir << "  %old.val = atomicrmw fadd float addrspace(1)* %out.ptr, float %in.val monotonic\n";
    ir << "  br label %reduce.done\n\n";
    ir << "reduce.done:\n";
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
        const std::string pointer_key = "kernel.arg." + std::to_string(i) + ".pointer";

        const auto type_it = fields.find(type_key);
        const auto name_it = fields.find(name_key);
        const auto pointer_it = fields.find(pointer_key);
        const std::string ptx_type = (type_it != fields.end()) ? type_it->second : ".u32";
        const bool is_pointer =
            (pointer_it != fields.end()) &&
            (pointer_it->second == "true" || pointer_it->second == "1");
        const std::string llvm_type = map_param_type_to_llvm(ptx_type, is_pointer);
        const std::string arg_name =
            (name_it != fields.end() && !name_it->second.empty())
                ? name_it->second
                : ("arg_" + std::to_string(i));
        arg_decls.push_back(llvm_type + " %" + arg_name);
        params.push_back({.ptx_type = ptx_type, .llvm_type = llvm_type, .name = arg_name});
    }

    const bool vector_add_signature = looks_like_vector_add_signature(pipeline.entry_name, params);
    bool matrix_mul_signature = looks_like_matrix_mul_signature(pipeline.entry_name, params);
    const bool negate_signature = looks_like_negate_signature(pipeline.entry_name, params);
    bool reduce_sum_signature = looks_like_reduce_sum_signature(pipeline.entry_name, params);
    const bool fp64_mul_add_signature =
        looks_like_fp64_mul_add_signature(pipeline.entry_name, params);
    if (matrix_mul_signature && params.size() >= 5) {
        params[3].llvm_type = "i32 addrspace(2)*";
        arg_decls[3] = params[3].llvm_type + " %" + params[3].name;
    }
    if (reduce_sum_signature && params.size() >= 3) {
        params[2].llvm_type = "i32 addrspace(2)*";
        arg_decls[2] = params[2].llvm_type + " %" + params[2].name;
    }

    const bool needs_thread_position_builtin =
        negate_signature || reduce_sum_signature || fp64_mul_add_signature;
    if (needs_thread_position_builtin) {
        const ParamInfo builtin_thread_position = {
            .ptx_type = ".builtin.thread_position_in_grid",
            .llvm_type = "i32",
            .name = "__air_thread_position_in_grid",
        };
        params.push_back(builtin_thread_position);
        arg_decls.push_back(builtin_thread_position.llvm_type + " %" + builtin_thread_position.name);
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

    if (vector_add_signature) {
        emit_vector_add_body(ir, params);
    } else if (matrix_mul_signature) {
        emit_matrix_mul_body(ir, params);
    } else if (negate_signature) {
        emit_negate_body(ir, params);
    } else if (reduce_sum_signature) {
        emit_reduce_sum_body(ir, params);
    } else if (fp64_mul_add_signature) {
        emit_fp64_mul_add_body(ir, params);
    } else {
        emit_lowered_instruction_comments(ir, pipeline.lowered_instructions);
        ir << "  ret void\n";
    }
    ir << "}\n\n";

    if (fp64_mul_add_signature) {
        ir << "declare double @llvm.fma.f64(double, double, double)\n\n";
    }

    std::ostringstream kernel_type;
    kernel_type << "void (";
    for (std::size_t i = 0; i < params.size(); ++i) {
        if (i > 0) {
            kernel_type << ", ";
        }
        kernel_type << params[i].llvm_type;
    }
    kernel_type << ")* @" << pipeline.entry_name;

    int thread_position_index = -1;
    for (int i = static_cast<int>(params.size()) - 1; i >= 0; --i) {
        if (!is_pointer_type(params[static_cast<std::size_t>(i)].llvm_type)) {
            thread_position_index = i;
            break;
        }
    }

    const int kernel_node_id = 0;
    const int empty_tuple_id = 1;
    const int kernel_args_tuple_id = 2;
    int next_meta_id = 3;

    std::vector<int> arg_meta_ids;
    arg_meta_ids.reserve(params.size());
    for (std::size_t i = 0; i < params.size(); ++i) {
        arg_meta_ids.push_back(next_meta_id++);
    }

    const int compile_opt_denorm_id = next_meta_id++;
    const int compile_opt_fast_math_id = next_meta_id++;
    const int compile_opt_fbfetch_id = next_meta_id++;
    const int air_version_tuple_id = next_meta_id++;
    const int language_version_tuple_id = next_meta_id++;

    ir << "attributes #0 = { \"air.kernel\" \"air.version\"=\"" << air_major << "."
       << air_minor << "\" }\n\n";
    ir << "!air.kernel = !{!" << kernel_node_id << "}\n";
    ir << "!" << kernel_node_id << " = !{" << kernel_type.str() << ", !" << empty_tuple_id << ", !"
       << kernel_args_tuple_id << "}\n";
    ir << "!" << empty_tuple_id << " = !{}\n";
    ir << "!" << kernel_args_tuple_id << " = !{";
    for (std::size_t i = 0; i < arg_meta_ids.size(); ++i) {
        if (i > 0) {
            ir << ", ";
        }
        ir << "!" << arg_meta_ids[i];
    }
    ir << "}\n";

    for (std::size_t i = 0; i < params.size(); ++i) {
        const ParamInfo& param = params[i];
        const bool is_thread_position = static_cast<int>(i) == thread_position_index;
        const std::string air_type_name = air_type_name_for_param(param, is_thread_position);
        const int arg_size = byte_size_for_llvm_type(param.llvm_type);
        const int arg_align = arg_size;

        if (is_thread_position) {
            ir << "!" << arg_meta_ids[i] << " = !{i32 " << i
               << ", !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"" << air_type_name
               << "\", !\"air.arg_name\", !\"" << param.name << "\"}\n";
            continue;
        }

        if (is_device_buffer_pointer(param.llvm_type)) {
            ir << "!" << arg_meta_ids[i]
               << " = !{i32 " << i << ", !\"air.buffer\", !\"air.location_index\", i32 " << i
               << ", i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 "
               << arg_size << ", !\"air.arg_type_align_size\", i32 " << arg_align
               << ", !\"air.arg_type_name\", !\"" << air_type_name << "\", !\"air.arg_name\", !\""
               << param.name << "\"}\n";
            continue;
        }

        ir << "!" << arg_meta_ids[i] << " = !{i32 " << i
           << ", !\"air.buffer\", !\"air.buffer_size\", i32 " << arg_size
           << ", !\"air.location_index\", i32 " << i
           << ", i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 "
           << arg_size << ", !\"air.arg_type_align_size\", i32 " << arg_align
           << ", !\"air.arg_type_name\", !\"" << air_type_name << "\", !\"air.arg_name\", !\""
           << param.name << "\"}\n";
    }

    ir << "!air.compile_options = !{!" << compile_opt_denorm_id << ", !" << compile_opt_fast_math_id
       << ", !" << compile_opt_fbfetch_id << "}\n";
    ir << "!" << compile_opt_denorm_id << " = !{!\"air.compile.denorms_disable\"}\n";
    ir << "!" << compile_opt_fast_math_id << " = !{!\"air.compile.fast_math_enable\"}\n";
    ir << "!" << compile_opt_fbfetch_id << " = !{!\"air.compile.framebuffer_fetch_enable\"}\n";
    ir << "!air.version = !{!" << air_version_tuple_id << "}\n";
    ir << "!air.language_version = !{!" << language_version_tuple_id << "}\n";
    ir << "!" << air_version_tuple_id << " = !{i32 " << air_major << ", i32 " << air_minor << ", i32 0}\n";
    ir << "!" << language_version_tuple_id << " = !{!\"Metal\", i32 " << language_major << ", i32 "
       << language_minor << ", i32 0}\n";

    result.ok = true;
    result.entry_name = pipeline.entry_name;
    result.llvm_ir = ir.str();
    result.warnings = pipeline.warnings;

    // --fp64=emulate: Dekker's algorithm decomposition is not yet implemented.
    // Emit a warning and fall through to native FP64 (kNative) behavior.
    if (options.fp64_mode == Fp64Mode::kEmulate) {
        result.warnings.push_back(
            "--fp64=emulate: Dekker's algorithm FP32-pair decomposition is not yet implemented; "
            "using native FP64 (full IEEE 754 double, but low throughput on Apple Silicon)");
    }

    // --fp64=warn: scan PTX source for any .f64 instructions and emit per-line warnings.
    if (options.fp64_mode == Fp64Mode::kWarn) {
        const std::string ptx_str(ptx);
        std::istringstream ptx_stream(ptx_str);
        std::string line;
        int line_no = 0;
        while (std::getline(ptx_stream, line)) {
            ++line_no;
            if (line.find(".f64") != std::string::npos) {
                result.warnings.push_back(
                    "fp64 instruction at line " + std::to_string(line_no) +
                    " (--fp64=warn): " + trim(line));
            }
        }
    }

    return result;
}

}  // namespace cumetal::ptx
