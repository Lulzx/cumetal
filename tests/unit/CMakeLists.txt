add_executable(cumetal_allocation_table_test
    allocation_table_test.cpp
    ${CMAKE_SOURCE_DIR}/runtime/rt/allocation_table.cpp
)

add_executable(cumetal_module_cache_test
    module_cache_test.cpp
    ${CMAKE_SOURCE_DIR}/runtime/cache/module_cache.cpp
)

add_executable(cumetal_library_conflict_test
    library_conflict_test.cpp
    ${CMAKE_SOURCE_DIR}/runtime/error/library_conflict.cpp
)

add_executable(cumetal_metallib_parser_test
    metallib_parser_test.cpp
)

add_executable(cumetal_ptx_parser_test
    ptx_parser_test.cpp
)

add_executable(cumetal_intrinsic_lower_test
    intrinsic_lower_test.cpp
)

add_executable(cumetal_addrspace_pass_test
    addrspace_pass_test.cpp
)

add_executable(cumetal_metadata_pass_test
    metadata_pass_test.cpp
)

add_executable(cumetal_phase1_pipeline_test
    phase1_pipeline_test.cpp
)

target_include_directories(cumetal_allocation_table_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/runtime/api
        ${CMAKE_SOURCE_DIR}/runtime/rt
        ${CMAKE_SOURCE_DIR}/runtime/metal_backend
)

target_include_directories(cumetal_module_cache_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/runtime/cache
)

target_include_directories(cumetal_library_conflict_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/runtime/error
)

target_link_libraries(cumetal_metallib_parser_test
    PRIVATE
        cumetal_common
)

target_link_libraries(cumetal_ptx_parser_test
    PRIVATE
        cumetal_ptx
)

target_link_libraries(cumetal_intrinsic_lower_test
    PRIVATE
        cumetal_passes
)

target_link_libraries(cumetal_addrspace_pass_test
    PRIVATE
        cumetal_passes
)

target_link_libraries(cumetal_metadata_pass_test
    PRIVATE
        cumetal_passes
)

target_link_libraries(cumetal_phase1_pipeline_test
    PRIVATE
        cumetal_passes
)

target_compile_options(cumetal_allocation_table_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_module_cache_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_library_conflict_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_metallib_parser_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_ptx_parser_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_intrinsic_lower_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_addrspace_pass_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_metadata_pass_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_phase1_pipeline_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_test(
    NAME unit_allocation_table
    COMMAND cumetal_allocation_table_test
)

add_test(
    NAME unit_module_cache
    COMMAND cumetal_module_cache_test
)

add_test(
    NAME unit_library_conflict
    COMMAND cumetal_library_conflict_test
)

add_test(
    NAME unit_metallib_parser
    COMMAND cumetal_metallib_parser_test
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.experimental.metallib
)

add_test(
    NAME unit_ptx_parser
    COMMAND cumetal_ptx_parser_test
)

add_test(
    NAME unit_intrinsic_lower
    COMMAND cumetal_intrinsic_lower_test
)

add_test(
    NAME unit_addrspace_pass
    COMMAND cumetal_addrspace_pass_test
)

add_test(
    NAME unit_metadata_pass
    COMMAND cumetal_metadata_pass_test
)

add_test(
    NAME unit_phase1_pipeline
    COMMAND cumetal_phase1_pipeline_test
)

add_test(
    NAME unit_install_uninstall_scripts
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_install_uninstall_test.sh
            ${CMAKE_SOURCE_DIR}/install/install.sh
            ${CMAKE_SOURCE_DIR}/install/uninstall.sh
            ${CMAKE_BINARY_DIR}
)
