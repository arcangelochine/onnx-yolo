#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

namespace onnx_yolo {

class Runtime {
public:
  explicit Runtime(const std::string &model_path);
  ~Runtime() = default;

  const Ort::AllocatedStringPtr input_name(const size_t &index) const;
  const Ort::AllocatedStringPtr output_name(const size_t &index) const;
  std::vector<Ort::Value> run_inference(cv::Mat &tensor_blob);

  // getters
  auto input_size() const -> const cv::Size & { return input_size_; }

private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::AllocatorWithDefaultOptions allocator_;
  Ort::MemoryInfo memory_info_;
  Ort::Session session_;
  Ort::RunOptions run_options_;

  size_t input_count_;
  size_t output_count_;
  std::vector<int64_t> input_shape_;
  cv::Size input_size_;

  std::vector<const char *> input_names_;
  std::vector<const char *> output_names_;

  void config_session();
  void load_model(const std::string &model_path);
  void load_inputs_outputs();
};

} // namespace onnx_yolo
