#include <chrono>
#include <onnx_yolo/yolo11.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace onnx_yolo {

Yolo11::Yolo11(const std::string &model_path) : YoloBase(model_path) {}

cv::Mat Yolo11::preprocess(const cv::Mat &image) {
  cv::Mat input_image;

  // Resize image according to model input size
  cv::resize(image, input_image, runtime_->input_size(), 0, 0,
             cv::INTER_LINEAR);

  // Convert the image to float and normalize to [0, 1]
  cv::normalize(input_image, input_image, 0, 1, cv::NORM_MINMAX, CV_32F);

  // Generate blob for inference
  cv::Mat input_blob;
  cv::dnn::blobFromImage(input_image, input_blob, 1.0, runtime_->input_size(),
                         cv::Scalar(), true, false);

  return input_blob;
}

std::vector<Ort::Value> Yolo11::infer(cv::Mat &input_blob) {
  // make the blob continous for efficiency
  if (!input_blob.isContinuous())
    input_blob = input_blob.clone();

  return runtime_->run_inference(input_blob);
}

std::vector<Detection>
Yolo11::postprocess(const std::vector<Ort::Value> &output_tensors,
                    float confidence_threshold, float iou_threshold) {
  std::vector<Detection> detections;
  const float *output_data = output_tensors[0].GetTensorData<float>();
  const auto &output_shape =
      output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

  // transpose the matrix to simplify postprocessing
  cv::Mat output_matrix(output_shape[1], output_shape[2], CV_32F,
                        (void *)output_data);
  cv::transpose(output_matrix, output_matrix);

  const int num_classes = output_matrix.cols - 4;

  // find detections
  for (int i = 0; i < output_matrix.rows; ++i) {
    // find best class
    float best_score = -1;
    int best_class = -1;

    for (int j = 0; j < num_classes; ++j) {
      const float class_score = output_matrix.at<float>(i, j + 4);
      if (class_score > best_score) {
        best_score = class_score;
        best_class = j;
      }
    }

    // apply confidence threshold and retrieve bbox data
    if (best_score >= confidence_threshold) {
      const float cx = output_matrix.at<float>(i, 0);
      const float cy = output_matrix.at<float>(i, 1);
      const float w = output_matrix.at<float>(i, 2);
      const float h = output_matrix.at<float>(i, 3);

      Detection detection;
      detection.class_id = best_class;
      detection.score = best_score;

      // shift bounding box to top left corner
      detection.bbox.x = static_cast<int>(cx - w / 2);
      detection.bbox.y = static_cast<int>(cy - h / 2);
      detection.bbox.width = static_cast<int>(w);
      detection.bbox.height = static_cast<int>(h);

      detections.push_back(detection);
    }
  }

  // apply nms
  std::vector<cv::Rect> bboxes;
  std::vector<float> scores;
  for (const auto &det : detections) {
    bboxes.push_back(det.bbox);
    scores.push_back(det.score);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(bboxes, scores, confidence_threshold, iou_threshold,
                    indices);

  std::vector<Detection> filtered_detections;
  for (int idx : indices) {
    filtered_detections.push_back(detections[idx]);
  }

  return filtered_detections;
}

} // namespace onnx_yolo