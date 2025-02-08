#pragma once

#include "base.hpp"

namespace onnx_yolo {

class Yolo11 final : public YoloBase {
public:
  explicit Yolo11(const std::string &model_path);
  ~Yolo11() override = default;

  cv::Mat preprocess(const cv::Mat &image) override;

  std::vector<Ort::Value> infer(cv::Mat &input_blob) override;

  std::vector<Detection>
  postprocess(const std::vector<Ort::Value> &output_tensors,
              float confidence_threshold, float iou_threshold) override;
};

} // namespace onnx_yolo