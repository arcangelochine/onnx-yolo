# Example executable
add_executable(example detection.cpp)

# Include directories
target_include_directories(example PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${OpenCV_INCLUDE_DIRS}
)

# Link against the onnx_yolo library and ONNX Runtime
target_link_libraries(example PRIVATE
  onnx_yolo
  ${ONNXRuntime_LIBS}
  ${OpenCV_LIBS}
)


