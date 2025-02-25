#include "demo.h"
#include "matrix.h"
#include "matrix_base.h"
#include "munkres.h"
#include "utils.h"
#include "group.h"
#include <Util/Macrofunc.h>
#include <vector>
#include <map>
#include <functional>
#include <torch/torch.h>

using namespace pose;

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

const cv::Scalar RED(0, 0, 255);
const cv::Scalar BLUE(255, 0, 0);
const cv::Scalar GREEN(0, 255, 0);


Demo::Demo() {
    InitializeColorStyle();
}

Demo::~Demo() = default;

void Demo::InitializeColorStyle() {

    color_style_.colors = {
        RED, RED, BLUE, BLUE, GREEN, RED, BLUE, GREEN, RED, BLUE, RED, BLUE, GREEN, RED, BLUE, RED, BLUE, RED, BLUE
    };

    color_style_.link_pairs = {
        {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},
        {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8}, {7, 9},
        {8, 10}, {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, {0, 5}, {0, 6}
    };

    color_style_.point_colors = {
        GREEN, RED, BLUE, RED, BLUE, RED, BLUE, RED, BLUE, RED, BLUE, RED, BLUE, RED, BLUE, RED, BLUE
    };
}


bool Demo::LoadModel(const std::string& model_path, const std::string& model_type) {
    try {
        // initialize session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // CUDA
        try {
            OrtCUDAProviderOptionsV2* cuda_options = nullptr;
            Ort::ThrowOnError(g_ort->CreateCUDAProviderOptions(&cuda_options));
            std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(g_ort->ReleaseCUDAProviderOptions)>
                rel_cuda_options(cuda_options, g_ort->ReleaseCUDAProviderOptions);

            Ort::ThrowOnError(g_ort->SessionOptionsAppendExecutionProvider_CUDA_V2(
                static_cast<OrtSessionOptions*>(session_options),
                rel_cuda_options.get()));
        }
        catch (const Ort::Exception& e) {
            std::cout << "CUDA provider not available, falling back to CPU: " << e.what() << std::endl;
        }

        model_name = model_type;

        // Convert string to wstring
        // ONNX model load
        std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(env_, widestr.c_str(), session_options);

        //*************************************************************************
        // print model input layer (node names, types, shape etc.)
        Ort::AllocatorWithDefaultOptions allocator;

        size_t num_input_nodes = session_->GetInputCount();
        input_names_.resize(num_input_nodes);
        input_name_strings_.resize(num_input_nodes);
        input_shapes_.resize(num_input_nodes);

        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_name_strings_[i] = input_name.get();
            input_names_[i] = input_name_strings_[i].c_str();

            auto type_info = session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_shapes_[i] = tensor_info.GetShape();

            if (input_shapes_[i][0] == -1) {
                input_shapes_[i][0] = 1;
            }

            std::cout << "Input " << i << " shape: ";
            for (const auto& dim : input_shapes_[i]) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
        }

        //*************************************************************************
        // print model output layer (node names, types, shape etc.)
        size_t num_output_nodes = session_->GetOutputCount();
        output_names_.resize(num_output_nodes);
        output_name_strings_.resize(num_output_nodes);
        output_shapes_.resize(num_output_nodes);

        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_name_strings_[i] = output_name.get();
            output_names_[i] = output_name_strings_[i].c_str();

            auto type_info = session_->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            output_shapes_[i] = tensor_info.GetShape();
        }

        return true;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

bool Demo::PadImage(const cv::Mat& input, cv::Mat& output, float aspect_ratio, int& left_pad, int& top_pad) {
    if (input.empty()) {
        return false;
    }

    float current_aspect_ratio = static_cast<float>(input.cols) / static_cast<float>(input.rows);

    left_pad = 0;
    top_pad = 0;

    if (current_aspect_ratio < aspect_ratio) {
        // Pad horizontally
        int target_width = static_cast<int>(aspect_ratio * input.rows);
        int pad_width = target_width - input.cols;
        left_pad = pad_width / 2;
        int right_pad = pad_width - left_pad;

        cv::copyMakeBorder(input, output,
            0, 0,                    // top, bottom
            left_pad, right_pad,     // left, right
            cv::BORDER_CONSTANT, cv::Scalar(0));
    }
    else {
        // Pad vertically
        int target_height = static_cast<int>(input.cols / aspect_ratio);
        int pad_height = target_height - input.rows;
        top_pad = pad_height / 2;
        int bottom_pad = pad_height - top_pad;

        cv::copyMakeBorder(input, output,
            top_pad, bottom_pad,     // top, bottom
            0, 0,                    // left, right
            cv::BORDER_CONSTANT, cv::Scalar(0));
    }

    return true;
}

bool Demo::Resize(const cv::Mat& input, cv::Mat& output, float& scale_x, float& scale_y) {

    if (input.empty()) {
        std::cerr << "Empty input image" << std::endl;
        return false;
    }

    scale_x = (float)input.cols / Input_width;
    scale_y = (float)input.rows / Input_height;

    cv::resize(input, output, cv::Size(Input_width, Input_height));

    if (output.size() != cv::Size(Input_width, Input_height)) {
        std::cerr << "Failed to resize to " << Input_width << "x" << Input_height << std::endl;
        return false;
    }

    return true;
}

void Demo::NormalizeImage(cv::Mat& image) {
    image.convertTo(image, CV_32F, 1.0 / 255.0);

    std::vector<float> mean = { 0.485f, 0.456f, 0.406f };
    std::vector<float> std = { 0.229f, 0.224f, 0.225f };

    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);

    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }

    cv::merge(channels, image);
}

cv::Mat Demo::PreprocessImage(const cv::Mat& image, float& scale_x, float& scale_y, int& left_pad, int& top_pad) {
    cv::Mat padded, resized, normalized;

    if (!PadImage(image, padded, static_cast<float>(Input_width) / Input_height, left_pad, top_pad)) {
        throw std::runtime_error("Failed to pad image");
    }

    if (!Resize(padded, resized, scale_x, scale_y)) {
        throw std::runtime_error("Failed to resize and pad image");
    }

    resized.copyTo(normalized);
    NormalizeImage(normalized);

    return normalized;
}

void Demo::PostprocessOutputs_ViTPose(
    const std::vector<OrtValue*>& outputs,
    const cv::Size& original_size,
    std::vector<std::vector<cv::Point2f>>& keypoints,
    std::vector<float>& scores,
    float scale_x,
    float scale_y,
    int left_pad,
    int top_pad) {

    // Get heatmaps and tags from model 
    float* heatmap_data = nullptr;
    Ort::ThrowOnError(g_ort->GetTensorMutableData(outputs[0], (void**)&heatmap_data));

    // Get tensor info
    OrtTensorTypeAndShapeInfo* tensor_info;
    Ort::ThrowOnError(g_ort->GetTensorTypeAndShape(outputs[0], &tensor_info));

    // Get shape
    size_t dim_count;
    Ort::ThrowOnError(g_ort->GetDimensionsCount(tensor_info, &dim_count));
    std::vector<int64_t> dims(dim_count);
    Ort::ThrowOnError(g_ort->GetDimensions(tensor_info, dims.data(), dim_count));

    // Convert tensor data to vector of cv::Mat
    std::vector<cv::Mat> heatmaps;

    int batch_size = dims[0];
    int num_joints = dims[1];
    int height = dims[2];
    int width = dims[3];

    // Reshape heatmap data into separate cv::Mat for each batch
    for (int n = 0; n < batch_size; n++) {
        cv::Mat batch_heatmap(num_joints, height * width, CV_32F);
        for (int j = 0; j < num_joints; j++) {
            float* row_ptr = batch_heatmap.ptr<float>(j);
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int tensor_idx = n * (num_joints * height * width) +
                        j * (height * width) +
                        h * width + w;
                    row_ptr[h * width + w] = heatmap_data[tensor_idx];
                }
            }
        }
        heatmaps.push_back(batch_heatmap);
    }

    auto [pred_coords, pred_maxvals] = get_final_preds_no_transform(heatmaps);

    // Convert predictions to keypoints format
    keypoints.clear();
    scores.clear();

    for (size_t n = 0; n < pred_coords.size(); n++) {
        std::vector<cv::Point2f> current_person;
        float current_score = 0.0f;

        for (size_t j = 0; j < pred_coords[n].size(); j++) {
            float conf = pred_maxvals[n][j][0];
                
            float x = pred_coords[n][j][0] * 4 * scale_x - left_pad;
            float y = pred_coords[n][j][1] * 4 * scale_y - top_pad;
            current_person.push_back(cv::Point2f(x, y));
            current_score += conf;

        }

        if (!current_person.empty()) {
            keypoints.push_back(current_person);
            scores.push_back(current_score / pred_coords[n].size());
        }
    }

    g_ort->ReleaseTensorTypeAndShapeInfo(tensor_info);
}


void Demo::PostprocessOutputs_HigherHRNet(
    const std::vector<OrtValue*>& outputs,
    const cv::Size& original_size,
    std::vector<std::vector<cv::Point2f>>& keypoints,
    std::vector<float>& scores,
    float scale_x,
    float scale_y,
    int left_pad,
    int top_pad) {

    const int NUM_KEYPOINTS = 17; // COCO keypoints
    std::vector<int64_t> base_size = { Input_width, Input_height };
    std::vector<int64_t> center = { Input_width / 2, Input_height / 2 };
    std::vector<float> scale = { Input_width / 200.0, Input_height / 200.0 };
    std::vector<torch::Tensor> tags_list;

    auto [torch_outputs, heatmaps, tags] = get_multi_stage_outputs_onnx(outputs, &base_size);




    /*
    // Debug intermediate outputs
    std::cout << "\n=== Debug Intermediate Outputs ===" << std::endl;
    std::cout << "torch_outputs size: " << torch_outputs.size() << std::endl;
    for (size_t i = 0; i < torch_outputs.size(); i++) {
        std::cout << "torch_output " << i << " shape: ";
        for (const auto& dim : torch_outputs[i].sizes()) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        std::cout << "torch_output " << i << " min/max: "
            << torch_outputs[i].min().item<float>() << "/"
            << torch_outputs[i].max().item<float>() << std::endl;
    }

    std::cout << "\nheatmaps size: " << heatmaps.size() << std::endl;
    for (size_t i = 0; i < heatmaps.size(); i++) {
        std::cout << "heatmap " << i << " shape: ";
        for (const auto& dim : heatmaps[i].sizes()) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        std::cout << "heatmap " << i << " min/max: "
            << heatmaps[i].min().item<float>() << "/"
            << heatmaps[i].max().item<float>() << std::endl;
    }

    std::cout << "\ntags size: " << tags.size() << std::endl;
    for (size_t i = 0; i < tags.size(); i++) {
        std::cout << "tag " << i << " shape: ";
        for (const auto& dim : tags[i].sizes()) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        std::cout << "tag " << i << " min/max: "
            << tags[i].min().item<float>() << "/"
            << tags[i].max().item<float>() << std::endl;
    }
    */


    auto [final_heatmaps, final_tags_list] = aggregate_results_onnx(tags_list, heatmaps, tags);



    /*
    // Debug final outputs
    std::cout << "\n=== Debug Final Outputs ===" << std::endl;
    std::cout << "final_heatmaps shape: ";
    for (const auto& dim : final_heatmaps.sizes()) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
    std::cout << "final_heatmaps min/max: "
        << final_heatmaps.min().item<float>() << "/"
        << final_heatmaps.max().item<float>() << std::endl;

    std::cout << "\nfinal_tags_list size: " << final_tags_list.size() << std::endl;
    for (size_t i = 0; i < final_tags_list.size(); i++) {
        std::cout << "final_tag " << i << " shape: ";
        for (const auto& dim : final_tags_list[i].sizes()) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        std::cout << "final_tag " << i << " min/max: "
            << final_tags_list[i].min().item<float>() << "/"
            << final_tags_list[i].max().item<float>() << std::endl;
    }
    std::cout << "============================\n" << std::endl;
    */




    // Concatenate tags along dimension 4
    torch::Tensor tags_tensor = torch::cat(final_tags_list, 4);

    /*
    // Print the shape of the concatenated tensor
    std::cout << "\nAfter concatenation:" << std::endl;
    std::cout << "tags_tensor shape: ";
    for (const auto& dim : tags_tensor.sizes()) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
    */

    HeatmapParser parser(nullptr);
    auto [grouped, parsed_scores] = parser.parse(final_heatmaps, tags_tensor, true, true);

    /*
    // 디버깅 코드 추가
    std::cout << "\n=== Debug Parse Results ===\n";

    // grouped 정보 출력
    std::cout << "grouped size: " << grouped.size() << std::endl;
    if (!grouped.empty()) {
        auto& first_batch = grouped[0];
        std::cout << "First batch shape: [" << first_batch.size(0) << ", "
            << first_batch.size(1) << ", " << first_batch.size(2) << "]" << std::endl;
        std::cout << "Number of people detected: " << first_batch.size(0) << std::endl;

    }
    else {
        std::cout << "No people detected in grouped" << std::endl;
    }

    // parsed_scores 정보 출력
    std::cout << "\nparsed_scores size: " << parsed_scores.size() << std::endl;
    if (!parsed_scores.empty()) {
        std::cout << "Scores: ";
        for (size_t i = 0; i < std::min(parsed_scores.size(), static_cast<size_t>(5)); i++) {
            std::cout << parsed_scores[i] << " ";
        }
        if (parsed_scores.size() > 5) {
            std::cout << "... and " << (parsed_scores.size() - 5) << " more scores";
        }
        std::cout << std::endl;
    }

    std::cout << "=========================\n" << std::endl;
    */

    // Clear existing keypoints and scores
    keypoints.clear();
    scores.clear();

    // If no people detected, return
    if (grouped.empty() || grouped[0].size(0) == 0) {
        return;
    }

    // Process each person in the first batch
    auto& batch = grouped[0];
    int num_people = batch.size(0);
    int num_joints = batch.size(1);

    for (int person_idx = 0; person_idx < num_people; person_idx++) {
        std::vector<cv::Point2f> person_keypoints;
        float person_score = parsed_scores[person_idx];

        for (int joint_idx = 0; joint_idx < num_joints; joint_idx++) {
            // Get coordinates from the tensor
            float x = batch[person_idx][joint_idx][0].item<float>();
            float y = batch[person_idx][joint_idx][1].item<float>();
            float conf = batch[person_idx][joint_idx][2].item<float>();

            // Scale back to original image space
            float scaled_x = x  * scale_x - left_pad;
            float scaled_y = y  * scale_y - top_pad;

            // Only add keypoint if confidence is above 0
            if (conf > 0) {
                person_keypoints.push_back(cv::Point2f(scaled_x, scaled_y));
            }
            else {
                // Add an invalid point for consistency (will be skipped during drawing)
                person_keypoints.push_back(cv::Point2f(-1, -1));
            }
        }

        // Add this person if we have any valid keypoints
        if (!person_keypoints.empty()) {
            keypoints.push_back(person_keypoints);
            scores.push_back(person_score);
        }
    }
    

}

void Demo::DrawPoses(
    cv::Mat& image,
    const std::vector<std::vector<cv::Point2f>>& keypoints,
    const std::vector<float>& scores) {

    for (size_t person_idx = 0; person_idx < keypoints.size(); person_idx++) {
        const auto& person_keypoints = keypoints[person_idx];
        float score = scores[person_idx];

        // Draw skeleton lines
        for (size_t i = 0; i < color_style_.link_pairs.size(); i++) {
            const auto& pair = color_style_.link_pairs[i];

            if (pair.first < person_keypoints.size() &&
                pair.second < person_keypoints.size()) {

                const auto& pt1 = person_keypoints[pair.first];
                const auto& pt2 = person_keypoints[pair.second];

                // Only draw if both points are valid
                if (pt1.x >= 0 && pt1.y >= 0 && pt2.x >= 0 && pt2.y >= 0) {
                    cv::line(image, pt1, pt2, color_style_.colors[i], 2);
                }
            }
        }

        // Draw keypoints
        for (size_t i = 0; i < person_keypoints.size(); i++) {
            const auto& pt = person_keypoints[i];
            if (pt.x >= 0 && pt.y >= 0) {
                cv::circle(image, pt, 3, color_style_.point_colors[i], -1);
                cv::circle(image, pt, 3, cv::Scalar(0, 0, 0), 1);  // Black outline
            }
        }
    }
}

std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<std::vector<float>>>>
Demo::get_final_preds_no_transform(const std::vector<cv::Mat>& heatmaps) {
    PredictionResult pred_result = get_max_preds(heatmaps);
    std::vector<cv::Mat> heatmaps_processed = heatmaps;

    // post-processing
    heatmaps_processed = gaussian_blur(heatmaps_processed, 11);

    for (auto& heatmap : heatmaps_processed) {
        cv::max(heatmap, 1e-10, heatmap);
        cv::log(heatmap, heatmap);
    }

    // Apply taylor expansion
    for (size_t n = 0; n < pred_result.coords.size(); n++) {
        for (size_t p = 0; p < pred_result.coords[n].size(); p++) {
            std::vector<float> refined_coord = taylor(heatmaps_processed[n], pred_result.coords[n][p]);
            pred_result.coords[n][p] = refined_coord;
        }
    }

    return std::make_pair(pred_result.coords, pred_result.maxvals);
}

PredictionResult Demo::get_max_preds(const std::vector<cv::Mat>& heatmaps) {
    PredictionResult result;
    int batch_size = heatmaps.size();
    int num_joints = heatmaps[0].rows;
    int width = 48;

    result.coords.resize(batch_size, std::vector<std::vector<float>>(num_joints, std::vector<float>(2)));
    result.maxvals.resize(batch_size, std::vector<std::vector<float>>(num_joints, std::vector<float>(1)));

    for (int n = 0; n < batch_size; n++) {
        for (int p = 0; p < num_joints; p++) {
            cv::Point maxLoc;
            double maxVal;
            cv::minMaxLoc(heatmaps[n].row(p), nullptr, &maxVal, nullptr, &maxLoc);

            result.maxvals[n][p][0] = maxVal;
            result.coords[n][p][0] = maxLoc.x % width;
            result.coords[n][p][1] = maxLoc.x / width;

            if (maxVal <= 0) {
                result.coords[n][p][0] = 0.0f;
                result.coords[n][p][1] = 0.0f;
            }
        }
    }

    return result;
}

std::vector<cv::Mat> Demo::gaussian_blur(std::vector<cv::Mat>& heatmaps, int kernel) {
    int border = (kernel - 1) / 2;

    for (size_t i = 0; i < heatmaps.size(); i++) {
        double origin_max;
        cv::minMaxLoc(heatmaps[i], nullptr, &origin_max);

        cv::Mat padded;
        cv::copyMakeBorder(heatmaps[i], padded, border, border, border, border, cv::BORDER_CONSTANT, 0);
        cv::GaussianBlur(padded, padded, cv::Size(kernel, kernel), 0);

        cv::Mat cropped = padded(cv::Range(border, padded.rows - border),
            cv::Range(border, padded.cols - border));

        double current_max;
        cv::minMaxLoc(cropped, nullptr, &current_max);
        cropped *= origin_max / current_max;

        heatmaps[i] = cropped.clone();
    }

    return heatmaps;
}

std::vector<float> Demo::taylor(const cv::Mat& heatmap, const std::vector<float>& coord) {

    std::vector<float> result = coord;
    int px = static_cast<int>(coord[0]);
    int py = static_cast<int>(coord[1]);

    if (1 < px && px < heatmap.cols - 2 && 1 < py && py < heatmap.rows - 2) {
        Eigen::Matrix2f hessian;
        Eigen::Vector2f derivative;

        float dx = 0.5f * (heatmap.at<float>(py, px + 1) - heatmap.at<float>(py, px - 1));
        float dy = 0.5f * (heatmap.at<float>(py + 1, px) - heatmap.at<float>(py - 1, px));
        float dxx = 0.25f * (heatmap.at<float>(py, px + 2) - 2 * heatmap.at<float>(py, px) + heatmap.at<float>(py, px - 2));
        float dxy = 0.25f * (heatmap.at<float>(py + 1, px + 1) - heatmap.at<float>(py - 1, px + 1) - heatmap.at<float>(py + 1, px - 1) + heatmap.at<float>(py - 1, px - 1));
        float dyy = 0.25f * (heatmap.at<float>(py + 2, px) - 2 * heatmap.at<float>(py, px) + heatmap.at<float>(py - 2, px));

        derivative << dx, dy;
        hessian << dxx, dxy, dxy, dyy;

        if (std::abs(hessian.determinant()) > 1e-10) {
            Eigen::Vector2f offset = -hessian.inverse() * derivative;
            result[0] += offset(0);
            result[1] += offset(1);
        }
    }

    return result;
}

cv::Mat Demo::convert_tensor_to_mat(float* tensor_data, const std::vector<int64_t>& dims) {
    if (dims.size() != 4) {
        throw std::runtime_error("Expected 4D tensor");
    }

    int batch_size = dims[0];
    int channels = dims[1];
    int height = dims[2];
    int width = dims[3];

    cv::Mat result(height, width, CV_32FC(channels));
    memcpy(result.data, tensor_data, batch_size * channels * height * width * sizeof(float));

    return result;
}

bool Demo::ProcessVideo(
    const std::string& video_path,
    const std::string& output_path) {

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << video_path << std::endl;
        return false;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter writer;
    writer.open(output_path,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        cv::Size(width, height));

    if (!writer.isOpened()) {
        std::cerr << "Error creating video writer" << std::endl;
        return false;
    }

    cv::Mat frame;
    int frame_count = 0;
    double total_time = 0.0;

    while (cap.read(frame)) {

        auto start_time = std::chrono::high_resolution_clock::now();

        cv::Mat output_frame;
        if (!ProcessImage(frame, output_frame)) {
            std::cerr << "Error processing frame " << frame_count << std::endl;
            continue;
        }

        auto end_time = std::chrono::high_resolution_clock::now();

        writer.write(output_frame);
        frame_count++;

        std::chrono::duration<double> diff = end_time - start_time;
        total_time += diff.count();
    }

    double avg_fps = frame_count / total_time;
    std::cout << "Average FPS: " << avg_fps << std::endl;

    cap.release();
    writer.release();
    return true;
}

bool Demo::ProcessImage(const cv::Mat& image, cv::Mat& output_image) {
    try {
        if (!session_ || input_shapes_.empty()) {
            std::cerr << "Model not properly loaded" << std::endl;
            return false;
        }

        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

        float scale_x, scale_y = 1.0f;
        int left_pad, top_pad = 0;

        cv::Mat processed = PreprocessImage(rgb_image, scale_x, scale_y, left_pad, top_pad);

        size_t tensor_size = 1 * 3 * Input_height * Input_width;  // channels * height * width
        std::vector<float> input_tensor_values(tensor_size);

        // HWC to CHW conversion
        float* input_data = input_tensor_values.data();
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < Input_height; h++) {
                for (int w = 0; w < Input_width; w++) {
                    size_t tensor_idx = c * Input_height * Input_width + h * Input_width + w;
                    if (tensor_idx >= tensor_size) {
                        std::cerr << "Tensor index out of bounds" << std::endl;
                        return false;
                    }
                    input_data[tensor_idx] = processed.at<cv::Vec3f>(h, w)[c];
                }
            }
        }

        // Create input tensor
        std::vector<int64_t> input_shape = { 1, 3, Input_height, Input_width };
        OrtMemoryInfo* memory_info;

        // CPU
        Ort::ThrowOnError(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

        // CUDA
        //Ort::ThrowOnError(g_ort->CreateMemoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault, &memory_info));


        OrtValue* input_tensor = nullptr;
        Ort::ThrowOnError(g_ort->CreateTensorWithDataAsOrtValue(
            memory_info,
            input_tensor_values.data(),
            tensor_size * sizeof(float),
            input_shape.data(),
            input_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensor));

        std::vector<OrtValue*> outputs;
        if (model_name == "vitpose") {
            outputs.resize(1);  // ViTPose has 1 output
        }
        else if (model_name == "higherhrnet") {
            outputs.resize(2);  // HigherHRNet has 2 outputs
        }

        // Inference
        Ort::ThrowOnError(g_ort->Run(
            static_cast<OrtSession*>(*session_),
            nullptr,
            input_names_.data(),
            (const OrtValue* const*)&input_tensor,
            1,
            output_names_.data(),
            output_names_.size(),
            outputs.data()));

        // Process outputs
        std::vector<std::vector<cv::Point2f>> keypoints;
        std::vector<float> scores;
        if (model_name == "vitpose") {
            PostprocessOutputs_ViTPose(outputs, rgb_image.size(), keypoints, scores,
                scale_x, scale_y, left_pad, top_pad);
        }
        else if (model_name == "higherhrnet") {
            PostprocessOutputs_HigherHRNet(outputs, rgb_image.size(), keypoints, scores,
                scale_x, scale_y, left_pad, top_pad);
        }

        image.copyTo(output_image);
        DrawPoses(output_image, keypoints, scores);

        // Cleanup
        g_ort->ReleaseValue(input_tensor);
        for (auto& tensor : outputs) {
            g_ort->ReleaseValue(tensor);
        }
        g_ort->ReleaseMemoryInfo(memory_info);

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error processing image: " << e.what() << std::endl;
        return false;
    }
}

torch::Tensor Demo::convert_ort_to_torch(OrtValue* ort_tensor) {
    float* tensor_data = nullptr;
    g_ort->GetTensorMutableData(ort_tensor, (void**)&tensor_data);

    OrtTensorTypeAndShapeInfo* tensor_info;
    g_ort->GetTensorTypeAndShape(ort_tensor, &tensor_info);

    std::vector<int64_t> dims;
    size_t dim_count;
    g_ort->GetDimensionsCount(tensor_info, &dim_count);
    dims.resize(dim_count);
    g_ort->GetDimensions(tensor_info, dims.data(), dim_count);

    torch::Tensor torch_tensor = torch::from_blob(tensor_data, dims, torch::kFloat32);

    g_ort->ReleaseTensorTypeAndShapeInfo(tensor_info);
    return torch_tensor;
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
Demo::get_multi_stage_outputs_onnx(
    const std::vector<OrtValue*>& outputs,
    const std::vector<int64_t>* size_projected) {

    std::vector<torch::Tensor> torch_outputs;
    std::vector<torch::Tensor> heatmaps;
    std::vector<torch::Tensor> tags;

    torch::Tensor heatmaps_avg = torch::zeros({});
    int num_heatmaps = 0;

    // Convert and process original outputs
    for (size_t i = 0; i < outputs.size(); i++) {
        torch::Tensor output = convert_ort_to_torch(outputs[i]);

        if (outputs.size() > 1 && i != outputs.size() - 1) {
            auto target_tensor = convert_ort_to_torch(outputs[1]);
            output = torch::nn::functional::interpolate(
                output,
                torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{target_tensor.size(2), target_tensor.size(3)})
                .mode(torch::kBilinear)
                .align_corners(false));
        }


        if (i == 0) {
            heatmaps_avg = output.slice(1, 0, 17);
        }
        else {
            heatmaps_avg += output.slice(1, 0, 17);
        }
        num_heatmaps += 1;

        if (i == 0) {
            tags.push_back(output.slice(1, 17));
        }

        torch_outputs.push_back(output);
    }

    if (num_heatmaps > 0) {
        heatmaps.push_back(heatmaps_avg / num_heatmaps);
    }

    // Resize if size_projected is provided
    if (size_projected != nullptr) {
        for (auto& hm : heatmaps) {
            hm = torch::nn::functional::interpolate(
                hm,
                torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{ (*size_projected)[1], (*size_projected)[0] })
                .mode(torch::kBilinear)
                .align_corners(false)
            );
        }

        for (auto& tag : tags) {
            tag = torch::nn::functional::interpolate(
                tag,
                torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{ (*size_projected)[1], (*size_projected)[0] })
                .mode(torch::kBilinear)
                .align_corners(false)
            );
        }
    }

    return std::make_tuple(torch_outputs, heatmaps, tags);
}

std::pair<torch::Tensor, std::vector<torch::Tensor>>
Demo::aggregate_results_onnx(
    std::vector<torch::Tensor>& tags_list,
    const std::vector<torch::Tensor>& heatmaps,
    const std::vector<torch::Tensor>& tags) {

    // Add each tag tensor to tags_list with an extra dimension
    for (const auto& tms : tags) {
        tags_list.push_back(tms.unsqueeze(4));
    }

    // Average the heatmaps
    torch::Tensor heatmaps_avg;
    if (!heatmaps.empty()) {
        // If we have flip augmentation, average the heatmaps
        if (heatmaps.size() > 1) {
            heatmaps_avg = (heatmaps[0] + heatmaps[1]) / 2.0;
        }
        else {
            heatmaps_avg = heatmaps[0];
        }
    }

    return { heatmaps_avg, tags_list };
}