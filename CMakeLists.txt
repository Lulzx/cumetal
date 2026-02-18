cmake_minimum_required(VERSION 3.28)

project(cumetal VERSION 0.1.0 LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

option(CUMETAL_BUILD_TESTS "Build test targets" ON)

add_library(cumetal_common STATIC
    compiler/common/src/metallib.cpp
)

target_include_directories(cumetal_common
    PUBLIC
        compiler/common/include
)

target_compile_options(cumetal_common
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_library(cumetal_runtime SHARED
    runtime/cache/module_cache.cpp
    runtime/driver/cuda_driver.cpp
    runtime/rt/allocation_table.cpp
    runtime/rt/cuda_runtime.cpp
    runtime/metal_backend/metal_backend.mm
)

set_target_properties(cumetal_runtime
    PROPERTIES
        OUTPUT_NAME cumetal
)

set_source_files_properties(
    runtime/metal_backend/metal_backend.mm
    PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
)

target_include_directories(cumetal_runtime
    PUBLIC
        runtime/api
        runtime/cache
        runtime/rt
        runtime/metal_backend
)

target_link_libraries(cumetal_runtime
    PRIVATE
        "-framework Foundation"
        "-framework Metal"
)

target_compile_options(cumetal_runtime
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_library(cumetal_air_validate STATIC
    compiler/air_validate/src/validator.cpp
)

target_include_directories(cumetal_air_validate
    PUBLIC
        compiler/air_validate/include
)

target_link_libraries(cumetal_air_validate
    PUBLIC
        cumetal_common
)

target_compile_options(cumetal_air_validate
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_executable(air_validate
    compiler/air_validate/src/main.cpp
)

target_link_libraries(air_validate
    PRIVATE
        cumetal_air_validate
)

add_library(cumetal_air_emitter STATIC
    compiler/air_emitter/src/emitter.cpp
)

target_include_directories(cumetal_air_emitter
    PUBLIC
        compiler/air_emitter/include
)

target_link_libraries(cumetal_air_emitter
    PUBLIC
        cumetal_air_validate
)

target_compile_options(cumetal_air_emitter
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_executable(cumetal-air-emitter
    compiler/air_emitter/src/main.cpp
)

target_link_libraries(cumetal-air-emitter
    PRIVATE
        cumetal_air_emitter
)

add_executable(cumetalc
    compiler/cumetalc/main.cpp
)

target_link_libraries(cumetalc
    PRIVATE
        cumetal_air_emitter
)

target_compile_options(cumetalc
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_executable(air_inspect
    tools/air_inspect/main.cpp
)

target_link_libraries(air_inspect
    PRIVATE
        cumetal_common
)

target_compile_options(air_inspect
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

if(CUMETAL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/air_abi)
    add_subdirectory(tests/functional)
    add_subdirectory(tests/unit)
endif()

install(TARGETS
    cumetal_runtime
    air_inspect
    air_validate
    cumetal-air-emitter
    cumetalc
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    runtime/api/cuda.h
    runtime/api/cuda_runtime.h
    DESTINATION include
)
