

cv::Mat image = cv::imread("img.png", CV_LOAD_IMAGE_COLOR);
cv::Mat image_float;
image.convertTo(image_float, CV_32FC3);
image_float = image_float / 255.0;
std::vector<float> input_array;
input_array.assign((float*)image_float.datastart, (float*)image_float.dataend);

    ...........

    InferenceModel<armnnCaffeParser::ICaffeParser, float> model(params);
    std::vector<float> output(model.GetOutputSize());
    model.Run(input_array, output);