#include "matrix.h"
#include "matrix_base.h"
#include "munkres.h"
#include "utils.h"
#include "group.h"
#include <vector>
#include <map>
#include <functional>
#include <torch/torch.h>

using namespace pose;

Params::Params(void* cfg) {
    num_joints = 17;
    max_num_people = 30;

    detection_threshold = 0.1;
    tag_threshold = 1.0;
    use_detection_val = true;
    ignore_too_much = false;

    joint_order = {
        0, 1, 2, 3, 4, 5, 6, 11, 12, 7, 8, 9, 10, 13, 14, 15, 16
    };
}

// C++ implementation of py_max_match function
torch::Tensor pose::py_max_match(const torch::Tensor& scores) {
    // Create a Munkres object
    munkres_cpp::Matrix<double> cost_matrix(scores.size(0), scores.size(1));

    // Convert torch tensor to 2D vector for munkres
    std::vector<std::vector<double>> scores_vec;
    for (int i = 0; i < scores.size(0); i++) {
        std::vector<double> row;
        for (int j = 0; j < scores.size(1); j++) {
            row.push_back(scores[i][j].item<double>());
        }
        scores_vec.push_back(row);
    }

    // Solve the assignment problem
    munkres_cpp::Munkres<double, munkres_cpp::Matrix> solver(cost_matrix);

    // Extract the results
    std::vector<std::pair<int, int>> tmp_vec;
    for (size_t i = 0; i < scores_vec.size(); ++i) {
        for (size_t j = 0; j < scores_vec[i].size(); ++j) {
            // In the munkres implementation, 0 indicates an assignment
            if (scores_vec[i][j] == 0) {
                tmp_vec.push_back(std::make_pair(i, j));
            }
        }
    }

    // Convert the result to torch tensor
    auto options = torch::TensorOptions().dtype(torch::kInt32);
    torch::Tensor tmp = torch::zeros({ static_cast<int64_t>(tmp_vec.size()), 2 }, options);
    for (size_t i = 0; i < tmp_vec.size(); ++i) {
        tmp[i][0] = tmp_vec[i].first;
        tmp[i][1] = tmp_vec[i].second;
    }

    return tmp;
}

// C++ implementation of match_by_tag function
torch::Tensor pose::match_by_tag(const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& inp, Params& params) {
    auto tag_k = std::get<0>(inp);
    auto loc_k = std::get<1>(inp);
    auto val_k = std::get<2>(inp);

    // Create default tensor with zeros
    auto default_ = torch::zeros({ params.num_joints, 3 + tag_k.size(2) });

    std::map<float, torch::Tensor> joint_dict;
    std::map<float, std::vector<torch::Tensor>> tag_dict;

    for (int i = 0; i < params.num_joints; i++) {
        int idx = params.joint_order[i];

        auto tags = tag_k[idx];

        // Concatenate loc_k, val_k and tags
        auto joints = torch::cat({
            loc_k[idx],
            val_k[idx].index({torch::indexing::Slice(), torch::indexing::None}),
            tags
            }, 1);

        // Create mask for detection threshold
        auto mask = joints.index({ torch::indexing::Slice(), 2 }) > params.detection_threshold;

        tags = tags.index({ mask });
        joints = joints.index({ mask });

        if (joints.size(0) == 0) {
            continue;
        }

        if (i == 0 || joint_dict.empty()) {
            // First joint or empty dict
            for (int j = 0; j < tags.size(0); j++) {
                auto tag = tags[j];
                auto joint = joints[j];

                float key = tag[0].item<float>();

                auto joint_entry = default_.clone();
                joint_entry[idx] = joint;
                joint_dict[key] = joint_entry;

                std::vector<torch::Tensor> tag_list;
                tag_list.push_back(tag);
                tag_dict[key] = tag_list;
            }
        }
        else {
            // Get keys of joint_dict up to max_num_people
            std::vector<float> grouped_keys;
            for (auto& pair : joint_dict) {
                if (grouped_keys.size() >= params.max_num_people) {
                    break;
                }
                grouped_keys.push_back(pair.first);
            }

            // Calculate mean tags for each key
            std::vector<torch::Tensor> grouped_tags;
            for (auto& key : grouped_keys) {
                auto tag_list = tag_dict[key];
                auto tag_stack = torch::stack(tag_list);
                auto mean_tag = torch::mean(tag_stack, 0);
                grouped_tags.push_back(mean_tag);
            }

            if (params.ignore_too_much && grouped_keys.size() == params.max_num_people) {
                continue;
            }

            // Create a tensor from grouped_tags
            auto grouped_tags_tensor = torch::stack(grouped_tags);

            // Calculate difference between joints and grouped_tags
            auto diff = joints.index({ torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice(3, torch::indexing::None) }) -
                grouped_tags_tensor.index({ torch::indexing::None, torch::indexing::Slice(), torch::indexing::Slice() });

            // Calculate Euclidean distance
            auto diff_normed = torch::norm(diff, 2, 2);
            auto diff_saved = diff_normed.clone();

            if (params.use_detection_val) {
                diff_normed = torch::round(diff_normed) * 100 - joints.index({ torch::indexing::Slice(), torch::indexing::Slice(2, 3) });
            }

            int num_added = diff.size(0);
            int num_grouped = diff.size(1);

            // Handle case where there are more points to add than groups
            if (num_added > num_grouped) {
                auto padding = torch::ones({ num_added, num_added - num_grouped }) * 1e10;
                diff_normed = torch::cat({ diff_normed, padding }, 1);
            }

            // Convert to CPU for munkres
            auto pairs = pose::py_max_match(diff_normed);

            for (int j = 0; j < pairs.size(0); j++) {
                int row = pairs[j][0].item<int>();
                int col = pairs[j][1].item<int>();

                if (row < num_added && col < num_grouped &&
                    diff_saved[row][col].item<float>() < params.tag_threshold) {
                    // Match to existing group
                    float key = grouped_keys[col];
                    joint_dict[key][idx] = joints[row];
                    tag_dict[key].push_back(tags[row]);
                }
                else {
                    // Create new group
                    float key = tags[row][0].item<float>();
                    auto joint_entry = default_.clone();
                    joint_entry[idx] = joints[row];
                    joint_dict[key] = joint_entry;

                    std::vector<torch::Tensor> tag_list;
                    tag_list.push_back(tags[row]);
                    tag_dict[key] = tag_list;
                }
            }
        }
    }

    // Convert joint_dict to tensor
    std::vector<torch::Tensor> ans_vec;
    for (auto& pair : joint_dict) {
        ans_vec.push_back(pair.second);
    }

    torch::Tensor ans;
    if (!ans_vec.empty()) {
        ans = torch::stack(ans_vec);
    }
    else {
        ans = torch::zeros({ 0, params.num_joints, 3 + tag_k.size(2) });
    }

    return ans;
}


HeatmapParser::HeatmapParser(void* cfg) : params(cfg) {
    tag_per_joint = true;
    pool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(5).stride(1).padding(2));
}

torch::Tensor HeatmapParser::nms(torch::Tensor det) {
    auto maxm = pool->forward(det);
    maxm = torch::eq(maxm, det).to(torch::kFloat32);
    det = det * maxm;
    return det;
}


std::vector<torch::Tensor>HeatmapParser::match(const torch::Tensor& tag_k, const torch::Tensor& loc_k, const torch::Tensor& val_k) {
    // Define a lambda function that will be mapped over the zipped inputs
    auto match_lambda = [this](const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& x) {
        return match_by_tag(x, this->params);
        };

    // Create vector to hold results
    std::vector<torch::Tensor> results;

    // Process each element in the input vectors
    for (int i = 0; i < tag_k.size(0); i++) {
        auto tuple = std::make_tuple(
            tag_k[i],
            loc_k[i],
            val_k[i]
        );
        results.push_back(match_lambda(tuple));
    }

    return results;
}

// Top-k function
std::map<std::string, torch::Tensor> HeatmapParser::top_k(torch::Tensor det, torch::Tensor tag) {
    // Apply NMS
    det = this->nms(det);

    // Get dimensions
    int64_t num_images = det.size(0);
    int64_t num_joints = det.size(1);
    int64_t h = det.size(2);
    int64_t w = det.size(3);

    // Reshape detection to [num_images, num_joints, h*w]
    det = det.view({ num_images, num_joints, -1 });

    // Get top-k values and indices
    auto topk_result = det.topk(30, 2);
    auto val_k = std::get<0>(topk_result);
    auto ind = std::get<1>(topk_result);

    // Reshape tag to [tag.size(0), tag.size(1), w*h, tag.size(3)]
    tag = tag.view({ tag.size(0), tag.size(1), w * h, -1 });

    // Stack tag_k along dimension 3
    std::vector<torch::Tensor> tag_k_parts;
    for (int64_t i = 0; i < tag.size(3); i++) {
        auto tag_part = tag.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), i });
        auto gathered = torch::gather(tag_part, 2, ind);
        tag_k_parts.push_back(gathered);
    }
    auto tag_k = torch::stack(tag_k_parts, 3);

    // Calculate x and y from ind
    auto x = ind % w;
    auto y = (ind / w).to(torch::kLong);

    // Stack x and y to form ind_k
    auto ind_k = torch::stack({ x, y }, 3);

    // Create result map
    std::map<std::string, torch::Tensor> ans;
    ans["tag_k"] = tag_k.cpu();
    ans["loc_k"] = ind_k.cpu();
    ans["val_k"] = val_k.cpu();

    return ans;
}

// Adjust function
std::vector<torch::Tensor> HeatmapParser::adjust(std::vector<torch::Tensor> ans, torch::Tensor det) {
    for (int batch_id = 0; batch_id < ans.size(); batch_id++) {
        torch::Tensor& people = ans[batch_id];

        for (int people_id = 0; people_id < people.size(0); people_id++) {
            for (int joint_id = 0; joint_id < people.size(1); joint_id++) {
                // If joint exists (confidence > 0)
                if (people[people_id][joint_id][2].item<float>() > 0) {
                    // Get coordinates
                    float y = people[people_id][joint_id][0].item<float>();
                    float x = people[people_id][joint_id][1].item<float>();

                    int xx = static_cast<int>(x);
                    int yy = static_cast<int>(y);

                    auto tmp = det[batch_id][joint_id];

                    // Compare adjacent values for y adjustment
                    if (yy + 1 < tmp.size(1) && yy - 1 >= 0) {
                        if (tmp[xx][std::min(yy + 1, static_cast<int>(tmp.size(1) - 1))].item<float>() >
                            tmp[xx][std::max(yy - 1, 0)].item<float>()) {
                            y += 0.25;
                        }
                        else {
                            y -= 0.25;
                        }
                    }

                    // Compare adjacent values for x adjustment
                    if (xx + 1 < tmp.size(0) && xx - 1 >= 0) {
                        if (tmp[std::min(xx + 1, static_cast<int>(tmp.size(0) - 1))][yy].item<float>() >
                            tmp[std::max(0, xx - 1)][yy].item<float>()) {
                            x += 0.25;
                        }
                        else {
                            x -= 0.25;
                        }
                    }

                    // Update coordinates with offset
                    people[people_id][joint_id][0] = y + 0.5;
                    people[people_id][joint_id][1] = x + 0.5;
                }
            }
        }
    }

    return ans;
}

// Refine function
torch::Tensor HeatmapParser::refine(torch::Tensor det, torch::Tensor tag, torch::Tensor keypoints) {
    // Check if tag is 3D, add an extra dimension if necessary
    if (tag.dim() == 3) {
        // Reshape tag to [17, 128, 128, 1]
        tag = tag.unsqueeze(3);
    }

    // Collect tag values for detected keypoints
    std::vector<torch::Tensor> tags;
    for (int i = 0; i < keypoints.size(0); i++) {
        if (keypoints[i][2].item<float>() > 0) {
            // Get coordinates of detected keypoint
            int x = static_cast<int>(keypoints[i][0].item<float>());
            int y = static_cast<int>(keypoints[i][1].item<float>());

            // Save tag value
            tags.push_back(tag[i][y][x]);
        }
    }

    // If no keypoints detected, early return
    if (tags.empty()) {
        return keypoints;
    }

    // Calculate mean tag
    torch::Tensor tags_tensor = torch::stack(tags);
    torch::Tensor prev_tag = torch::mean(tags_tensor, 0);

    // Vector to store refined keypoints
    std::vector<std::tuple<float, float, float>> ans;

    for (int i = 0; i < keypoints.size(0); i++) {
        // Get detection scores for joint i
        torch::Tensor tmp = det[i];

        // Calculate distance between all tag values and mean tag
        torch::Tensor tt = torch::sqrt(torch::sum(torch::pow(tag[i] - prev_tag.reshape({ 1, 1, -1 }), 2), 2));

        // Adjust scores by tag distance
        torch::Tensor tmp2 = tmp - torch::round(tt);

        // Find maximum position
        auto max_result = torch::argmax(tmp2.flatten());
        int64_t max_index = max_result.item<int64_t>();

        // Convert flat index to 2D coordinates
        int y = max_index / tmp.size(1);
        int x = max_index % tmp.size(1);

        // Store original coordinates
        int xx = x;
        int yy = y;

        // Get detection score at maximum position
        float val = tmp[y][x].item<float>();

        // Add 0.5 offset
        x += 0.5;
        y += 0.5;

        // Add a quarter offset based on neighboring values
        if (xx + 1 < tmp.size(1) && xx - 1 >= 0) {
            if (tmp[yy][std::min(xx + 1, static_cast<int>(tmp.size(1) - 1))].item<float>() >
                tmp[yy][std::max(xx - 1, 0)].item<float>()) {
                x += 0.25;
            }
            else {
                x -= 0.25;
            }
        }

        if (yy + 1 < tmp.size(0) && yy - 1 >= 0) {
            if (tmp[std::min(yy + 1, static_cast<int>(tmp.size(0) - 1))][xx].item<float>() >
                tmp[std::max(0, yy - 1)][xx].item<float>()) {
                y += 0.25;
            }
            else {
                y -= 0.25;
            }
        }

        // Add to results
        ans.push_back(std::make_tuple(x, y, val));
    }

    // Convert results to tensor
    torch::Tensor ans_tensor = torch::zeros_like(keypoints);
    for (int i = 0; i < ans.size(); i++) {
        float x = std::get<0>(ans[i]);
        float y = std::get<1>(ans[i]);
        float val = std::get<2>(ans[i]);

        // Update keypoints if new detection is valid and original was not detected
        if (val > 0 && keypoints[i][2].item<float>() == 0) {
            ans_tensor[i][0] = x;
            ans_tensor[i][1] = y;
            ans_tensor[i][2] = val;
        }
        else {
            // Copy original keypoint
            ans_tensor[i] = keypoints[i];
        }
    }

    return ans_tensor;
}

// Parse function
std::pair<std::vector<torch::Tensor>, std::vector<float>> HeatmapParser::parse(torch::Tensor det, torch::Tensor tag, bool adjust, bool refine) {
    // Call match with the result of top_k
    auto top_k_result = this->top_k(det, tag);

    std::vector<torch::Tensor> ans = this->match(top_k_result["tag_k"], top_k_result["loc_k"], top_k_result["val_k"]);

    // Apply adjust if specified
    if (adjust) {
        ans = this->adjust(ans, det);
    }

    // Calculate scores as mean of confidence values
    std::vector<float> scores;
    for (int i = 0; i < ans[0].size(0); i++) {
        auto person = ans[0][i];
        auto mean_score = torch::mean(person.index({ torch::indexing::Slice(), 2 })).item<float>();
        scores.push_back(mean_score);
    }

    // Apply refine if specified
    if (refine) {
        auto& batch_ans = ans[0];

        // Convert tensor data to CPU for processing
        auto det_numpy = det[0].cpu();
        auto tag_numpy = tag[0].cpu();

        // Refine each detected person
        for (int i = 0; i < batch_ans.size(0); i++) {
            batch_ans[i] = this->refine(det_numpy, tag_numpy, batch_ans[i]);
        }

        // Replace with refined results
        ans = { batch_ans };
    }

    return std::make_pair(ans, scores);
}