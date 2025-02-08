#include <onnx_yolo/runtime.hpp>

#include <iostream>

#if defined(HAS_CUDA)
#include <cuda_provider_factory.h>
#elif defined(HAS_COREML)
#include <coreml_provider_factory.h>
#else
#include <cpu_provider_factory.h>
#endif

namespace onnx_yolo
{

  Runtime::Runtime(const std::string &model_path)
      : env_(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime"), memory_info_(nullptr),
        session_(nullptr)
  {
    config_session();

    memory_info_ = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    load_model(model_path);
    load_inputs_outputs();
  }

  const Ort::AllocatedStringPtr Runtime::input_name(const size_t &index) const
  {
    return session_.GetInputNameAllocated(index, allocator_);
  }

  const Ort::AllocatedStringPtr Runtime::output_name(const size_t &index) const
  {
    return session_.GetOutputNameAllocated(index, allocator_);
  }

  std::vector<Ort::Value> Runtime::run_inference(cv::Mat &tensor_blob)
  {
    const size_t tensor_size =
        input_shape_[1] * input_shape_[2] * input_shape_[3];

    auto tensor = Ort::Value::CreateTensor<float>(
        memory_info_, tensor_blob.ptr<float>(), tensor_size, input_shape_.data(),
        input_shape_.size());

    return session_.Run(run_options_, input_names_.data(), &tensor,
                        input_names_.size(), output_names_.data(),
                        output_names_.size());
  }

  void Runtime::config_session()
  {
    session_options_.SetIntraOpNumThreads(0);
    session_options_.DisableCpuMemArena();
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    bool cuda_available = false;

#if defined(HAS_CUDA)
    cuda_available =
        nullptr == OrtSessionOptionsAppendExecutionProvider_CUDA(session_options_, 0);
#elif defined(HAS_COREML)
    bool core_ml_available =
        nullptr ==
        OrtSessionOptionsAppendExecutionProvider_CoreML(session_options_, 0);
#else
    std::cout << "Falling back to CPU Execution Provider" << std::endl;
#endif
  }

  void Runtime::load_model(const std::string &model_path)
  {
    try
    {
      session_ = Ort::Session(env_, model_path.c_str(), session_options_);

      input_count_ = session_.GetInputCount();
      output_count_ = session_.GetOutputCount();
      input_shape_ = session_.GetInputTypeInfo(0)
                         .GetTensorTypeAndShapeInfo()
                         .GetShape(); // <batch, c, h, w>
      input_size_ = cv::Size(input_shape_[2], input_shape_[3]);
    }
    catch (const Ort::Exception &e)
    {
      std::cerr << e.what() << std::endl;
      exit(1);
    }
  }

  void Runtime::load_inputs_outputs()
  {
    input_names_.reserve(input_count_);
    output_names_.reserve(output_count_);

    for (size_t i = 0; i < input_count_; ++i)
    {
      auto name = input_name(i);
      input_names_.push_back(name.release());
    }

    for (size_t i = 0; i < output_count_; ++i)
    {
      auto name = output_name(i);
      output_names_.push_back(name.release());
    }
  }

} // onnx_yolo