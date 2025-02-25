#pragma once

#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
#include <vector>
#include <memory>
#include <string>
#include <Eigen/Dense>
#include <cmath>
#include <torch/torch.h>
#include <map>


namespace pose {
    struct ColorStyle {
        std::vector<cv::Scalar> colors;
        std::vector<std::pair<int, int>> link_pairs;
        std::vector<cv::Scalar> point_colors;
    };

    struct PredictionResult {
        std::vector<std::vector<std::vector<float>>> coords;
        std::vector<std::vector<std::vector<float>>> maxvals;
    };

    class Demo {
    public:
        Demo();
        ~Demo();

        bool LoadModel(const std::string& model_path, const std::string& model_type);
        bool ProcessImage(const cv::Mat& image, cv::Mat& output_image);
        bool ProcessVideo(const std::string& video_path, const std::string& output_path);

    private:
        std::string model_name;

        static constexpr int Input_width = 512;
        static constexpr int Input_height = 512;

        ColorStyle color_style_;

        std::vector<std::string> input_name_strings_;
        std::vector<std::string> output_name_strings_;

        //*************************************************************************
        // initialize  enviroment...one enviroment per process
        // enviroment maintains thread pools and other state info
        Ort::Env env_{ ORT_LOGGING_LEVEL_WARNING, "demo" };
        std::unique_ptr<Ort::Session> session_;

        // CPU
        Ort::MemoryInfo memory_info_{
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
        };

        // CUDA
        //Ort::MemoryInfo memory_info_{
        //    Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault)
        //};

        std::vector<const char*> input_names_;
        std::vector<const char*> output_names_;
        std::vector<std::vector<int64_t>> input_shapes_;
        std::vector<std::vector<int64_t>> output_shapes_;

        // Processing methods
        cv::Mat PreprocessImage(const cv::Mat& image, float& scale_x, float& scale_y, int& left_pad, int& top_pad);

        void PostprocessOutputs_HigherHRNet(
            const std::vector<OrtValue*>& outputs,
            const cv::Size& original_size,
            std::vector<std::vector<cv::Point2f>>& keypoints,
            std::vector<float>& scores,
            float scale_x,
            float scale_y,
            int left_pad,
            int top_pad);

        void PostprocessOutputs_ViTPose(
            const std::vector<OrtValue*>& outputs,
            const cv::Size& original_size,
            std::vector<std::vector<cv::Point2f>>& keypoints,
            std::vector<float>& scores,
            float scale_x,
            float scale_y,
            int left_pad,
            int top_pad);

        void DrawPoses(cv::Mat& image, const std::vector<std::vector<cv::Point2f>>& keypoints, const std::vector<float>& scores);

        void InitializeColorStyle();

        bool Resize(const cv::Mat& input, cv::Mat& output, float& scale_x, float& scale_y);

        void NormalizeImage(cv::Mat& image);

        bool PadImage(const cv::Mat& input, cv::Mat& output, float aspect_ratio, int& left_pad, int& top_pad);

        std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<std::vector<float>>>>
            get_final_preds_no_transform(const std::vector<cv::Mat>& heatmaps);

        PredictionResult get_max_preds(const std::vector<cv::Mat>& heatmaps);

        std::vector<cv::Mat> gaussian_blur(std::vector<cv::Mat>& heatmaps, int kernel);

        std::vector<float> taylor(const cv::Mat& heatmap, const std::vector<float>& coord);

        cv::Mat convert_tensor_to_mat(float* tensor_data, const std::vector<int64_t>& dims);

        std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>

        get_multi_stage_outputs_onnx(
            const std::vector<OrtValue*>& outputs,
            const std::vector<int64_t>* size_projected = nullptr);

        torch::Tensor convert_ort_to_torch(OrtValue* ort_tensor);

        std::pair<torch::Tensor, std::vector<torch::Tensor>>
            aggregate_results_onnx(
                std::vector<torch::Tensor>& tags_list,
                const std::vector<torch::Tensor>& heatmaps,
                const std::vector<torch::Tensor>& tags);
    };
}