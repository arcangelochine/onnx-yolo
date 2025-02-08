#pragma once

#include "runtime.hpp"

#include <memory>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace onnx_yolo {

struct Detection {
  int class_id;
  float score;
  cv::Rect bbox;
};

class YoloBase {
public:
  explicit YoloBase(const std::string &model_path);
  virtual ~YoloBase() = default;

  virtual cv::Mat preprocess(const cv::Mat &image) = 0;

  virtual std::vector<Ort::Value> infer(cv::Mat &input_blob) = 0;

  virtual std::vector<Detection>
  postprocess(const std::vector<Ort::Value> &output_tensors,
              float confidence_threshold, float iou_threshold) = 0;

  auto input_size() const -> const cv::Size & {
    return runtime_->input_size();
  };

protected:
  std::unique_ptr<Runtime> runtime_;
};

} // namespace onnx_yolo
