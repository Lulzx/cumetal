if(NOT APPLE)
    message(STATUS "Skipping functional tests: Apple platform required")
    return()
endif()

add_executable(cumetal_runtime_vector_add_test
    runtime_vector_add_test.cpp
)

add_executable(cumetal_runtime_matrix_mul_test
    runtime_matrix_mul_test.cpp
)

add_executable(cumetal_runtime_stream_vector_add_test
    runtime_stream_vector_add_test.cpp
)

add_executable(cumetal_runtime_null_stream_sync_test
    runtime_null_stream_sync_test.cpp
)

add_executable(cumetal_runtime_stream_per_thread_test
    runtime_stream_per_thread_test.cpp
)

add_executable(cumetal_runtime_async_memops_test
    runtime_async_memops_test.cpp
)

add_executable(cumetal_runtime_event_test
    runtime_event_test.cpp
)

add_executable(cumetal_runtime_stream_wait_event_test
    runtime_stream_wait_event_test.cpp
)

add_executable(cumetal_runtime_stream_query_test
    runtime_stream_query_test.cpp
)

add_executable(cumetal_runtime_memcpy_kind_test
    runtime_memcpy_kind_test.cpp
)

add_executable(cumetal_runtime_symbol_memcpy_test
    runtime_symbol_memcpy_test.cpp
)

add_executable(cumetal_runtime_mem_get_info_test
    runtime_mem_get_info_test.cpp
)

add_executable(cumetal_runtime_host_alloc_test
    runtime_host_alloc_test.cpp
)

add_executable(cumetal_runtime_host_pointer_api_test
    runtime_host_pointer_api_test.cpp
)

add_executable(cumetal_runtime_stream_flags_test
    runtime_stream_flags_test.cpp
)

add_executable(cumetal_runtime_stream_callback_test
    runtime_stream_callback_test.cpp
)

add_executable(cumetal_runtime_device_api_test
    runtime_device_api_test.cpp
)

add_executable(cumetal_runtime_device_properties_test
    runtime_device_properties_test.cpp
)

add_executable(cumetal_runtime_device_attribute_test
    runtime_device_attribute_test.cpp
)

add_executable(cumetal_runtime_device_reset_test
    runtime_device_reset_test.cpp
)

add_executable(cumetal_runtime_device_flags_test
    runtime_device_flags_test.cpp
)

add_executable(cumetal_runtime_error_api_test
    runtime_error_api_test.cpp
)

add_executable(cumetal_runtime_profiler_api_test
    runtime_profiler_api_test.cpp
)

add_executable(cumetal_curand_uniform_test
    curand_uniform_test.cpp
)

add_executable(cumetal_cublas_api_test
    cublas_api_test.cpp
)

add_executable(cumetal_driver_vector_add_test
    driver_vector_add_test.cpp
)

add_executable(cumetal_driver_null_stream_sync_test
    driver_null_stream_sync_test.cpp
)

add_executable(cumetal_driver_matrix_mul_test
    driver_matrix_mul_test.cpp
)

add_executable(cumetal_driver_device_api_test
    driver_device_api_test.cpp
)

add_executable(cumetal_driver_error_api_test
    driver_error_api_test.cpp
)

add_executable(cumetal_driver_profiler_api_test
    driver_profiler_api_test.cpp
)

add_executable(cumetal_driver_device_query_test
    driver_device_query_test.cpp
)

add_executable(cumetal_driver_device_attribute_test
    driver_device_attribute_test.cpp
)

add_executable(cumetal_driver_stream_flags_test
    driver_stream_flags_test.cpp
)

add_executable(cumetal_driver_stream_per_thread_test
    driver_stream_per_thread_test.cpp
)

add_executable(cumetal_driver_stream_callback_test
    driver_stream_callback_test.cpp
)

add_executable(cumetal_driver_context_switch_test
    driver_context_switch_test.cpp
)

add_executable(cumetal_driver_context_requirements_test
    driver_context_requirements_test.cpp
)

add_executable(cumetal_driver_async_memcpy_test
    driver_async_memcpy_test.cpp
)

add_executable(cumetal_driver_memset_test
    driver_memset_test.cpp
)

add_executable(cumetal_driver_mem_get_info_test
    driver_mem_get_info_test.cpp
)

add_executable(cumetal_driver_mem_alloc_managed_test
    driver_mem_alloc_managed_test.cpp
)

add_executable(cumetal_driver_host_alloc_test
    driver_host_alloc_test.cpp
)

add_executable(cumetal_driver_host_pointer_api_test
    driver_host_pointer_api_test.cpp
)

add_executable(cumetal_driver_module_load_data_test
    driver_module_load_data_test.cpp
)

add_executable(cumetal_driver_module_load_data_ptx_test
    driver_module_load_data_ptx_test.cpp
)

add_executable(cumetal_driver_launch_extra_test
    driver_launch_extra_test.cpp
)

add_executable(cumetal_driver_launch_extra_scalar_test
    driver_launch_extra_scalar_test.cpp
)

add_executable(cumetal_driver_stream_wait_event_test
    driver_stream_wait_event_test.cpp
)

add_executable(cumetal_runtime_axpy_offset_test
    runtime_axpy_offset_test.cpp
)

add_executable(cumetal_runtime_atomic_test
    runtime_atomic_test.cpp
)

add_executable(cumetal_runtime_registration_path_test
    runtime_registration_path_test.cpp
)

target_link_libraries(cumetal_runtime_vector_add_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_matrix_mul_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_stream_vector_add_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_null_stream_sync_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_stream_per_thread_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_async_memops_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_event_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_stream_wait_event_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_stream_query_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_memcpy_kind_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_symbol_memcpy_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_mem_get_info_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_host_alloc_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_host_pointer_api_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_stream_flags_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_stream_callback_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_device_api_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_device_properties_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_device_attribute_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_device_reset_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_device_flags_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_error_api_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_profiler_api_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_curand_uniform_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_cublas_api_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_vector_add_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_null_stream_sync_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_matrix_mul_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_device_api_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_error_api_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_profiler_api_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_device_query_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_device_attribute_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_stream_flags_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_stream_per_thread_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_stream_callback_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_context_switch_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_context_requirements_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_async_memcpy_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_memset_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_mem_get_info_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_mem_alloc_managed_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_host_alloc_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_host_pointer_api_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_module_load_data_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_module_load_data_ptx_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_launch_extra_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_launch_extra_scalar_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_stream_wait_event_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_axpy_offset_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_atomic_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_registration_path_test
    PRIVATE
        cumetal_runtime
)

target_compile_options(cumetal_runtime_vector_add_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_matrix_mul_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_stream_vector_add_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_null_stream_sync_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_stream_per_thread_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_async_memops_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_event_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_stream_wait_event_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_stream_query_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_memcpy_kind_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_symbol_memcpy_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_mem_get_info_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_host_alloc_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_host_pointer_api_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_stream_flags_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_stream_callback_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_device_api_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_device_properties_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_device_attribute_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_device_reset_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_device_flags_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_error_api_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_profiler_api_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_curand_uniform_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_cublas_api_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_vector_add_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_null_stream_sync_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_matrix_mul_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_device_api_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_error_api_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_profiler_api_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_device_query_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_device_attribute_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_stream_flags_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_stream_per_thread_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_stream_callback_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_context_switch_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_context_requirements_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_async_memcpy_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_memset_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_mem_get_info_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_mem_alloc_managed_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_host_alloc_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_host_pointer_api_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_module_load_data_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_module_load_data_ptx_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_launch_extra_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_launch_extra_scalar_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_stream_wait_event_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_axpy_offset_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_atomic_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_registration_path_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_test(
    NAME functional_runtime_vector_add
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_vector_add_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_vector_add PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_vector_add_heap_alloc
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_vector_add_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(
    functional_runtime_vector_add_heap_alloc
    PROPERTIES
        SKIP_RETURN_CODE 77
        ENVIRONMENT "CUMETAL_MTLHEAP_ALLOC=1;CUMETAL_MTLHEAP_CHUNK_BYTES=1048576"
)

add_test(
    NAME functional_runtime_vector_add_cu
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_cumetalc_cu_runtime_vector_add.sh
            $<TARGET_FILE:cumetalc>
            $<TARGET_FILE:cumetal_runtime_vector_add_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/vector_add.cu
            ${CMAKE_CURRENT_BINARY_DIR}/vector_add.from_cu.xcrun.metallib
)

set_tests_properties(functional_runtime_vector_add_cu PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_sample_vector_add
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_samples_vector_add.sh
            $<TARGET_FILE:cumetalc>
            ${CMAKE_SOURCE_DIR}/samples/vectorAdd/vectorAdd.cu
            ${CMAKE_SOURCE_DIR}/samples/vectorAdd/vectorAdd.cpp
            ${CMAKE_SOURCE_DIR}/runtime/api
            $<TARGET_FILE_DIR:cumetal_runtime>
            ${CMAKE_CURRENT_BINARY_DIR}/sample_vector_add.from_cu.xcrun.metallib
            ${CMAKE_CURRENT_BINARY_DIR}/sample_vector_add
)

set_tests_properties(functional_sample_vector_add PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_matrix_mul
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_matrix_mul_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/matrix_mul.metal
            ${CMAKE_CURRENT_BINARY_DIR}/matrix_mul.metallib
)

set_tests_properties(functional_runtime_matrix_mul PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_stream_vector_add
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_stream_vector_add_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_stream_vector_add PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_null_stream_sync
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_null_stream_sync_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_null_stream_sync PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_stream_per_thread
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_stream_per_thread_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_stream_per_thread PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_async_memops
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_async_memops_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_async_memops PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_event
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_event_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_event PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_stream_wait_event
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_stream_wait_event_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_stream_wait_event PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_stream_query
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_stream_query_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_stream_query PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_memcpy_kind
    COMMAND cumetal_runtime_memcpy_kind_test
)

add_test(
    NAME functional_runtime_symbol_memcpy
    COMMAND cumetal_runtime_symbol_memcpy_test
)

add_test(
    NAME functional_runtime_mem_get_info
    COMMAND cumetal_runtime_mem_get_info_test
)

add_test(
    NAME functional_runtime_host_alloc
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_host_alloc_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_host_alloc PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_host_pointer_api
    COMMAND cumetal_runtime_host_pointer_api_test
)

add_test(
    NAME functional_runtime_stream_flags
    COMMAND cumetal_runtime_stream_flags_test
)

add_test(
    NAME functional_runtime_stream_callback
    COMMAND cumetal_runtime_stream_callback_test
)

add_test(
    NAME functional_runtime_device_api
    COMMAND cumetal_runtime_device_api_test
)

add_test(
    NAME functional_runtime_device_properties
    COMMAND cumetal_runtime_device_properties_test
)

add_test(
    NAME functional_runtime_device_attribute
    COMMAND cumetal_runtime_device_attribute_test
)

add_test(
    NAME functional_runtime_device_reset
    COMMAND cumetal_runtime_device_reset_test
)

add_test(
    NAME functional_runtime_device_flags
    COMMAND cumetal_runtime_device_flags_test
)

add_test(
    NAME functional_runtime_error_api
    COMMAND cumetal_runtime_error_api_test
)

add_test(
    NAME functional_runtime_profiler_api
    COMMAND cumetal_runtime_profiler_api_test
)

add_test(
    NAME functional_curand_uniform
    COMMAND cumetal_curand_uniform_test
)

add_test(
    NAME functional_cublas_api
    COMMAND cumetal_cublas_api_test
)

add_test(
    NAME functional_driver_vector_add
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_driver_vector_add_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_driver_vector_add PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_null_stream_sync
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_driver_null_stream_sync_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_driver_null_stream_sync PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_matrix_mul
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_driver_matrix_mul_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/matrix_mul.metal
            ${CMAKE_CURRENT_BINARY_DIR}/driver_matrix_mul.metallib
)

set_tests_properties(functional_driver_matrix_mul PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_device_api
    COMMAND cumetal_driver_device_api_test
)

add_test(
    NAME functional_driver_error_api
    COMMAND cumetal_driver_error_api_test
)

add_test(
    NAME functional_driver_profiler_api
    COMMAND cumetal_driver_profiler_api_test
)

add_test(
    NAME functional_driver_device_query
    COMMAND cumetal_driver_device_query_test
)

add_test(
    NAME functional_driver_device_attribute
    COMMAND cumetal_driver_device_attribute_test
)

add_test(
    NAME functional_driver_stream_flags
    COMMAND cumetal_driver_stream_flags_test
)

add_test(
    NAME functional_driver_stream_per_thread
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_driver_stream_per_thread_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_driver_stream_per_thread PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_stream_callback
    COMMAND cumetal_driver_stream_callback_test
)

add_test(
    NAME functional_driver_context_switch
    COMMAND cumetal_driver_context_switch_test
)

add_test(
    NAME functional_driver_context_requirements
    COMMAND cumetal_driver_context_requirements_test
)

add_test(
    NAME functional_driver_async_memcpy
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_driver_async_memcpy_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_driver_async_memcpy PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_memset
    COMMAND cumetal_driver_memset_test
)

add_test(
    NAME functional_driver_mem_get_info
    COMMAND cumetal_driver_mem_get_info_test
)

add_test(
    NAME functional_driver_mem_alloc_managed
    COMMAND cumetal_driver_mem_alloc_managed_test
)

add_test(
    NAME functional_driver_host_alloc
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_driver_host_alloc_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_driver_host_alloc PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_host_pointer_api
    COMMAND cumetal_driver_host_pointer_api_test
)

add_test(
    NAME functional_driver_module_load_data
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_driver_module_load_data_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_driver_module_load_data PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_module_load_data_ptx
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_driver_module_load_data_ptx.sh
            $<TARGET_FILE:cumetal_driver_module_load_data_ptx_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/vector_add.ptx
)

set_tests_properties(functional_driver_module_load_data_ptx PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_launch_extra
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_driver_launch_extra_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_driver_launch_extra PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_launch_extra_scalar
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_driver_launch_extra_scalar_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/matrix_mul_scalar.metal
            ${CMAKE_CURRENT_BINARY_DIR}/matrix_mul_scalar.metallib
)

set_tests_properties(functional_driver_launch_extra_scalar PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_stream_wait_event
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_driver_stream_wait_event_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_driver_stream_wait_event PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_axpy_offset
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_axpy_offset_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/axpy.metal
            ${CMAKE_CURRENT_BINARY_DIR}/axpy.metallib
)

set_tests_properties(functional_runtime_axpy_offset PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_atomic
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_atomic_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/atomic_inc.metal
            ${CMAKE_CURRENT_BINARY_DIR}/atomic_inc.metallib
)

set_tests_properties(functional_runtime_atomic PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_registration_path
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_registration_path_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_registration_path PROPERTIES SKIP_RETURN_CODE 77)
