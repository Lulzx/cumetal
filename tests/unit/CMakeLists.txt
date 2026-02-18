add_executable(cumetal_allocation_table_test
    allocation_table_test.cpp
    ${CMAKE_SOURCE_DIR}/runtime/rt/allocation_table.cpp
)

target_include_directories(cumetal_allocation_table_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/runtime/api
        ${CMAKE_SOURCE_DIR}/runtime/rt
        ${CMAKE_SOURCE_DIR}/runtime/metal_backend
)

target_compile_options(cumetal_allocation_table_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_test(
    NAME unit_allocation_table
    COMMAND cumetal_allocation_table_test
)
