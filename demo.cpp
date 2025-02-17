#include "demo.h"
#include <Util/Macrofunc.h>

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

    keypoints.clear();
    scores.clear();

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

    // Get heatmaps and tags from model 
    float* heatmap_data = nullptr;
    float* tag_data = nullptr;
    Ort::ThrowOnError(g_ort->GetTensorMutableData(outputs[1], (void**)&heatmap_data));
    Ort::ThrowOnError(g_ort->GetTensorMutableData(outputs[1], (void**)&tag_data));

    // Get tensor info
    OrtTensorTypeAndShapeInfo* tensor_info;
    Ort::ThrowOnError(g_ort->GetTensorTypeAndShape(outputs[1], &tensor_info));

    // Get shape
    size_t dim_count;
    Ort::ThrowOnError(g_ort->GetDimensionsCount(tensor_info, &dim_count));
    std::vector<int64_t> dims(dim_count);
    Ort::ThrowOnError(g_ort->GetDimensions(tensor_info, dims.data(), dim_count));

    int batch_size = dims[0];
    int num_joints = dims[1];
    int height = dims[2];
    int width = dims[3];

    keypoints.clear();
    scores.clear();

    std::vector<cv::Point2f> current_person;
    current_person.reserve(NUM_KEYPOINTS);
    float current_score = 0.0f;

    // Find peaks in each joint heatmap
    for (int j = 0; j < NUM_KEYPOINTS; j++) {
        float max_val = -1;
        int max_x = 0, max_y = 0;

        // Find maximum value position in heatmap
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int idx = j * height * width + h * width + w;
                float val = heatmap_data[idx];
                if (val > max_val) {
                    max_val = val;
                    max_x = w;
                    max_y = h;
                }
            }
        }

        if (max_val > 0.3f) {  // confidence threshold
            current_person.push_back(cv::Point2f(
                max_x * 2 * scale_x - left_pad,
                max_y * 2 * scale_y - top_pad
            ));
            current_score += max_val;
        }
        else {
            current_person.push_back(cv::Point2f(-1, -1));  // invalid point
        }
    }

    if (current_person.size() > 0) {
        keypoints.push_back(current_person);
        scores.push_back(current_score / NUM_KEYPOINTS);
    }

    g_ort->ReleaseTensorTypeAndShapeInfo(tensor_info);
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
        // g_ort->CreateMemoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault, &memory_info);


        OrtValue* input_tensor = nullptr;
        Ort::ThrowOnError(g_ort->CreateTensorWithDataAsOrtValue(
            memory_info,
            input_tensor_values.data(),
            tensor_size * sizeof(float),
            input_shape.data(),
            input_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensor));

        // Inference
        std::vector<OrtValue*> outputs;
        if (model_name == "vitpose") {
            outputs.resize(1);  // ViTPose has 1 output
        }
        else if (model_name == "higherhrnet") {
            outputs.resize(2);  // HigherHRNet has 2 outputs
        }

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
        float dxy = 0.25f * (heatmap.at<float>(py + 1, px + 1) - heatmap.at<float>(py - 1, px + 1) -
            heatmap.at<float>(py + 1, px - 1) + heatmap.at<float>(py - 1, px - 1));
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