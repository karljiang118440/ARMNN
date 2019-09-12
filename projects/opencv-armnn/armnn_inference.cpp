
//资料参考来源
   //  https://zhuanlan.zhihu.com/p/71772730 ，确实可以了解如何进行 opencv 使用
   
armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create(); 
armnn::INetworkPtr pose_reg_network{nullptr, [](armnn::INetwork *){}};
armnn::IOptimizedNetworkPtr pose_reg_optNet{nullptr, [](armnn::IOptimizedNetwork *){}};
armnn::InputTensors pose_reg_in_tensors;
armnn::OutputTensors pose_reg_ou_tensors;
armnn::IRuntimePtr runtime{nullptr, [](armnn::IRuntime *){}};
float yaw[1];
float pose_reg_input[64*64*3];


// loading tflite model
std::string pose_reg_modelPath = "/sdcard/Algo/pose.tflite";
pose_reg_network = parser->CreateNetworkFromBinaryFile(pose_reg_modelPath.c_str());

// binding input and output
armnnTfLiteParser::BindingPointInfo pose_reg_input_bind  = 
                              parser->GetNetworkInputBindingInfo(0, "input/ImageInput");
armnnTfLiteParser::BindingPointInfo pose_reg_output_bind = 
                              parser->GetNetworkOutputBindingInfo(0, "yaw/yangle");

// wrapping pose reg input and output
armnn::Tensor pose_reg_input_tensor(pose_reg_input_bind.second, pose_reg_input);
pose_reg_in_tensors.push_back(std::make_pair(pose_reg_input_bind.first, pose_reg_input_tensor));

armnn::Tensor pose_reg_output_tensor(pose_reg_output_bind.second, yaw);
pose_reg_ou_tensors.push_back(std::make_pair(pose_reg_output_bind.first, pose_reg_output_tensor));

// config runtime, fp16 accuracy 
armnn::IRuntime::CreationOptions runtimeOptions;
runtime = armnn::IRuntime::Create(runtimeOptions);
armnn::OptimizerOptions OptimizerOptions;
OptimizerOptions.m_ReduceFp32ToFp16 = true;
this->pose_reg_optNet = 
armnn::Optimize(*pose_reg_network, {armnn::Compute::GpuAcc},runtime->GetDeviceSpec(), OptimizerOptions);
runtime->LoadNetwork(this->pose_reg_identifier, std::move(this->pose_reg_optNet));

// load image
cv::Mat rgb_image = cv::imread("face.jpg", 1);
cv::resize(rgb_image, rgb_image, cv::Size(pose_reg_input_size, pose_reg_input_size));
rgb_image.convertTo(rgb_image, CV_32FC3);
rgb_image = (rgb_image - 127.5f) * 0.017f;

// preprocess image
int TOTAL   = 64 * 64 * 3;
float* data = (float*) rgb_image.data;
for (int i = 0; i < TOTAL; ++i) {
    pose_reg_input[i] = data[i];
}

// invoke graph forward inference
armnn::Status ret = runtime->EnqueueWorkload(
    this->pose_reg_identifier,
    this->pose_reg_in_tensors,
    this->pose_reg_ou_tensors
);
float result = yaw[0] * 180 / 3.14; 