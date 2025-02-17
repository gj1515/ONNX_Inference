#include "demo.h"
#include <iostream>
#include <filesystem>
#include <string>


bool process_file(pose::Demo& pose_detector, const std::string& input_path, const std::string& output_path, bool is_video) {

    if (is_video) {
        return pose_detector.ProcessVideo(input_path, output_path);
    }
    else {
        cv::Mat image = cv::imread(input_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << input_path << std::endl;
            return false;
        }

        cv::Mat output_image;
        if (!pose_detector.ProcessImage(image, output_image)) {
            std::cerr << "Failed to process image: " << input_path << std::endl;
            return false;
        }

        return cv::imwrite(output_path, output_image);
    }
}

int main() {
    // Set parameters
    const std::string model_path = "D:/Dev/Project/Pose/weights/2d/coco/vitpose/trained/192x256/export_0113_checkpoint_epoch_77.onnx";
    // "D:/Dev/Project/Pose/weights/2d/coco/vitpose/trained/192x256/export_0113_checkpoint_epoch_77.onnx", "D:/Dev/Project/Pose/weights/2d/coco/HigherHRNet/export_model_best_512x512.onnx"
    const std::string input_path = "D:/Dev/Dataset/inputs/images/single/";
    // "D:/Dev/Dataset/inputs/images/single/";, "D:/Dev/Dataset/inputs/videos/single/";
    const std::string output_path = "D:/test/";
    const std::string model_name = "vitpose";

    // Initialize pose detector
    pose::Demo pose_detector;

    if (!pose_detector.LoadModel(model_path, model_name)) {
        std::cerr << "Failed to load model from: " << model_path << std::endl;
        return 1;
    }
    std::cout << "Model loaded successfully" << std::endl;

    try {
        namespace fs = std::filesystem;

        fs::create_directories(output_path);

        for (const auto& entry : fs::directory_iterator(input_path)) {
            if (entry.is_regular_file()) {
                std::string filepath = entry.path().u8string();
                std::string filename = entry.path().filename().string();
                std::string ext = entry.path().extension().string();

                if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" &&
                    ext != ".mp4" && ext != ".avi") {
                    continue;
                }

                bool is_video_file = (ext == ".mp4" || ext == ".avi");
                std::string output_file = (fs::path(output_path) / filename).string();

                std::cout << "Processing: " << filename << std::endl;

                if (!process_file(pose_detector, filepath, output_file, is_video_file)) {
                    std::cerr << "Failed to process: " << filename << std::endl;
                }
            }
        }

        std::cout << "Processing completed successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}