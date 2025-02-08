#include <chrono>
#include <iostream>
#include <onnx_yolo/yolo11.hpp>
#include <opencv2/opencv.hpp>
#include <string>

int main(int argc, char const *argv[]) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <model_path> <image_path>"
              << std::endl;
    return 1;
  }

  std::string model_path = argv[1];
  std::string image_path = argv[2];

  // Load image
  cv::Mat image = cv::imread(image_path);

  if (image.empty()) {
    std::cerr << "Error: Could not load image at " << image_path << std::endl;
    return 1;
  }

  // Create YOLO object
  onnx_yolo::Yolo11 yolo11(model_path);

  // Measure time for preprocessing
  auto start_time = std::chrono::high_resolution_clock::now();

  auto input_tensor_values = yolo11.preprocess(image);

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<int64_t, std::milli> preprocess_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                            start_time);
  std::cout << "Preprocessing time: " << preprocess_duration.count() << " ms"
            << std::endl;

  // Measure time for inference
  start_time = std::chrono::high_resolution_clock::now();

  const auto output_tensors = yolo11.infer(input_tensor_values);

  end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<int64_t, std::milli> infer_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                            start_time);
  std::cout << "Inference time: " << infer_duration.count() << " ms"
            << std::endl;

  // Measure time for postprocessing
  start_time = std::chrono::high_resolution_clock::now();
  float conf_threshold = 0.5f;
  float iou_threshold = 0.4f;
  auto detections =
      yolo11.postprocess(output_tensors, conf_threshold, iou_threshold);

  // Calculate scaling factors based on resized image
  const float x_scale = image.cols / (float)yolo11.input_size().width;
  const float y_scale = image.rows / (float)yolo11.input_size().height;

  // Draw detections
  for (auto &det : detections) {
    // Scale detection
    det.bbox.x = static_cast<int>(det.bbox.x * x_scale);
    det.bbox.y = static_cast<int>(det.bbox.y * y_scale);
    det.bbox.width = static_cast<int>(det.bbox.width * x_scale);
    det.bbox.height = static_cast<int>(det.bbox.height * y_scale);

    cv::rectangle(image, det.bbox, cv::Scalar(0, 0, 200), 2);
    std::string label = "[" + std::to_string(det.class_id) + "] " +
                        std::to_string(det.score * 100.0f).substr(0, 4) + "%";

    // Calculate the size of the text
    int baseline = 0;
    cv::Size text_size =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

    // Set padding for the rectangle around the text
    int padding = 5; // Padding around the text

    // Define the rectangle coordinates
    cv::Rect text_rect(det.bbox.x, det.bbox.y - text_size.height - 10,
                       text_size.width + 2 * padding,
                       text_size.height + 2 * padding);

    // Draw the background rectangle (highlighting)
    cv::rectangle(image, text_rect, cv::Scalar(0, 0, 0),
                  -1); // Black background

    // Draw the text on top of the rectangle
    cv::putText(image, label, {det.bbox.x, det.bbox.y - 10},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  }
  end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<int64_t, std::milli> postprocess_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                            start_time);
  std::cout << "Postprocessing time: " << postprocess_duration.count() << " ms"
            << std::endl;

  // Measure total time
  int64_t total_duration_ms = preprocess_duration.count() +
                              infer_duration.count() +
                              postprocess_duration.count();

  std::cout << "Total time (preprocess + inference + postprocess): "
            << total_duration_ms << " ms" << std::endl;

  // Display image
  cv::resize(image, image, cv::Size(), 2, 2);
  cv::imshow("Detections", image);
  cv::waitKey(0);

  return 0;
}