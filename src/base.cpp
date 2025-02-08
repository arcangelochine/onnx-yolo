#include <onnx_yolo/base.hpp>
#include <stdexcept>

namespace onnx_yolo {

YoloBase::YoloBase(const std::string &model_path)
    : runtime_(std::make_unique<Runtime>(model_path)) {}

} // namespace onnx_yolo
