#pragma once

#include <torch/torch.h>
#include <vector>
#include <map>

namespace pose {


    class Params {
    public:
        Params(void* cfg);

        int num_joints;
        int max_num_people;
        float detection_threshold;
        float tag_threshold;
        bool use_detection_val;
        bool ignore_too_much;
        std::vector<int> joint_order;
    };

    torch::Tensor py_max_match(const torch::Tensor& scores);

    torch::Tensor match_by_tag(const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& inp, Params& params);

    class HeatmapParser {
    public:
        HeatmapParser(void* cfg);

        Params params;
        bool tag_per_joint;
        torch::nn::MaxPool2d pool{ nullptr };

        // NMS function
        torch::Tensor nms(torch::Tensor det);

        // Match function
        std::vector<torch::Tensor> match(const torch::Tensor& tag_k, const torch::Tensor& loc_k, const torch::Tensor& val_k);

        // Top-k function
        std::map<std::string, torch::Tensor> top_k(torch::Tensor det, torch::Tensor tag);

        // Adjust function
        std::vector<torch::Tensor> adjust(std::vector<torch::Tensor> ans, torch::Tensor det);

        // Refine function
        torch::Tensor refine(torch::Tensor det, torch::Tensor tag, torch::Tensor keypoints);

        // Parse function
        std::pair<std::vector<torch::Tensor>, std::vector<float>> parse(torch::Tensor det, torch::Tensor tag, bool adjust = true, bool refine = true);
    };
}