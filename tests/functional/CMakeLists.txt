if(NOT APPLE)
    message(STATUS "Skipping functional tests: Apple platform required")
    return()
endif()

add_executable(cumetal_runtime_vector_add_test
    runtime_vector_add_test.cpp
)

add_executable(cumetal_runtime_stream_vector_add_test
    runtime_stream_vector_add_test.cpp
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

add_executable(cumetal_runtime_mem_get_info_test
    runtime_mem_get_info_test.cpp
)

add_executable(cumetal_runtime_host_alloc_test
    runtime_host_alloc_test.cpp
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

add_executable(cumetal_driver_vector_add_test
    driver_vector_add_test.cpp
)

add_executable(cumetal_driver_device_api_test
    driver_device_api_test.cpp
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

add_executable(cumetal_driver_stream_callback_test
    driver_stream_callback_test.cpp
)

add_executable(cumetal_driver_context_switch_test
    driver_context_switch_test.cpp
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

add_executable(cumetal_driver_module_load_data_test
    driver_module_load_data_test.cpp
)

add_executable(cumetal_driver_launch_extra_test
    driver_launch_extra_test.cpp
)

add_executable(cumetal_driver_stream_wait_event_test
    driver_stream_wait_event_test.cpp
)

add_executable(cumetal_runtime_axpy_offset_test
    runtime_axpy_offset_test.cpp
)

target_link_libraries(cumetal_runtime_vector_add_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_stream_vector_add_test
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

target_link_libraries(cumetal_runtime_mem_get_info_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_host_alloc_test
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

target_link_libraries(cumetal_driver_vector_add_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_device_api_test
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

target_link_libraries(cumetal_driver_stream_callback_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_context_switch_test
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

target_link_libraries(cumetal_driver_module_load_data_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_launch_extra_test
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

target_compile_options(cumetal_runtime_vector_add_test
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

target_compile_options(cumetal_driver_vector_add_test
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

target_compile_options(cumetal_driver_module_load_data_test
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

add_test(
    NAME functional_runtime_vector_add
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_vector_add_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_vector_add PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_stream_vector_add
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_stream_vector_add_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_stream_vector_add PROPERTIES SKIP_RETURN_CODE 77)

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
    NAME functional_driver_vector_add
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_driver_vector_add_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_driver_vector_add PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_device_api
    COMMAND cumetal_driver_device_api_test
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
    NAME functional_driver_stream_callback
    COMMAND cumetal_driver_stream_callback_test
)

add_test(
    NAME functional_driver_context_switch
    COMMAND cumetal_driver_context_switch_test
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
    NAME functional_driver_module_load_data
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_driver_module_load_data_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_driver_module_load_data PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_launch_extra
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_driver_launch_extra_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_driver_launch_extra PROPERTIES SKIP_RETURN_CODE 77)

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
