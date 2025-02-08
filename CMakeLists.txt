cmake_minimum_required(VERSION 3.12)
project(onnx_yolo VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(OpenCV REQUIRED)

# Import ONNXRuntime
set(ONNXRuntime_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime)
set(ONNXRuntime_INCLUDE_DIRS "${ONNXRuntime_ROOT}/include")

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(ONNXRuntime_LIBS "${ONNXRuntime_ROOT}/lib/onnxruntime.lib")
    set(ONNXRuntime_DLL "${ONNXRuntime_ROOT}/lib/onnxruntime.dll")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(ONNXRuntime_LIBS "${ONNXRuntime_ROOT}/lib/libonnxruntime.so")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")  # macOS
    set(ONNXRuntime_LIBS "${ONNXRuntime_ROOT}/lib/libonnxruntime.dylib")
else()
    message(FATAL_ERROR "Unsupported OS: ${CMAKE_SYSTEM_NAME}")
endif()

# Add sources
file(GLOB_RECURSE SOURCES src/*.cpp)

# Create library
add_library(onnx_yolo SHARED ${SOURCES})

target_include_directories(onnx_yolo PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>  # For installation
  ${OpenCV_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${ONNXRuntime_INCLUDE_DIRS}>
)

# Link dependencies
target_link_libraries(onnx_yolo PRIVATE
  ${OpenCV_LIBS}
  ${ONNXRuntime_LIBS}
)

# RPATH settings
set_target_properties(onnx_yolo PROPERTIES
    CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE
    CMAKE_INSTALL_RPATH "\$ORIGIN/../lib"
)

# Installation rules
install(TARGETS onnx_yolo
    EXPORT onnx_yolo_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT onnx_yolo_targets
    FILE onnx_yoloConfig.cmake
    DESTINATION lib/cmake/onnx_yolo
)

# Copy ONNXRuntime DLL for Windows users
if(WIN32)
    install(FILES ${ONNXRuntime_DLL} DESTINATION bin)
endif()

add_subdirectory(examples)
