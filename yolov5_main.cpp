
/**********************************************************************************
 *
 * Author Denis  zhangyuyang
 *
 **********************************************************************************/


#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>


std::vector<float> Letterbox(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size) {
    auto in_h = static_cast<float>(src.rows);
    auto in_w = static_cast<float>(src.cols);
    float out_h = out_size.height;
    float out_w = out_size.width;

    float scale = std::min(out_w / in_w, out_h / in_h);

    int mid_h = static_cast<int>(in_h * scale);
    int mid_w = static_cast<int>(in_w * scale);

    cv::resize(src, dst, cv::Size(mid_w, mid_h));

    int top = (static_cast<int>(out_h) - mid_h) / 2;
    int down = (static_cast<int>(out_h) - mid_h + 1) / 2;
    int left = (static_cast<int>(out_w) - mid_w) / 2;
    int right = (static_cast<int>(out_w) - mid_w + 1) / 2;

    cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    std::vector<float> pad_info{ static_cast<float>(left), static_cast<float>(top), scale };
    return pad_info;
}

std::vector<torch::Tensor> nms(torch::Tensor preds, float score_thresh = 0.15, float iou_thresh = 0.35)
{
    std::vector<torch::Tensor> output;

    for (size_t i = 0; i < preds.sizes()[0]; ++i)
    {
        //pred.sizes 25200, 117
        torch::Tensor pred = preds.select(0, i);
        pred = pred.to(at::kCPU);

        // Filter by scores
        torch::Tensor scores = pred.select(1, 4) * std::get<0>(torch::max(pred.slice(1, 5, pred.sizes()[1]-32), 1));
        pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));

        if (pred.sizes()[0] == 0) continue;

        // (center_x, center_y, w, h) to (left, top, right, bottom)
        pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
        pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
        pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
        pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

        // Computing scores and classes
        std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]-32), 1);
        pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
        pred.select(1, 5) = std::get<1>(max_tuple);
		//80+5+32 = 117 
        torch::Tensor  dets = pred.slice(1, 0, pred.sizes()[1]);

        torch::Tensor keep = torch::empty({ dets.sizes()[0] });
        torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
        std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
        torch::Tensor v = std::get<0>(indexes_tuple);
        torch::Tensor indexes = std::get<1>(indexes_tuple);
        int count = 0;

        while (indexes.sizes()[0] > 0)
        {
            keep[count] = (indexes[0].item().toInt());
            count += 1;

            // Computing overlaps
            torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1);
            for (size_t i = 0; i < indexes.sizes()[0] - 1; ++i)
            {
                lefts[i] = std::max(dets[indexes[0]][0].item().toFloat(), dets[indexes[i + 1]][0].item().toFloat());
                tops[i] = std::max(dets[indexes[0]][1].item().toFloat(), dets[indexes[i + 1]][1].item().toFloat());
                rights[i] = std::min(dets[indexes[0]][2].item().toFloat(), dets[indexes[i + 1]][2].item().toFloat());
                bottoms[i] = std::min(dets[indexes[0]][3].item().toFloat(), dets[indexes[i + 1]][3].item().toFloat());
                widths[i] = std::max(float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
                heights[i] = std::max(float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());
            }
            torch::Tensor overlaps = widths * heights;

            // FIlter by IOUs
            torch::Tensor ious = overlaps / (areas.select(0, indexes[0].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);
            indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);

        }
        keep = keep.toType(torch::kInt64);
        output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));

    }
    return output;
}



int main() {


    cv::Mat image = cv::imread("yolov5/bus.jpg");
    cv::Mat img_input;
    std::vector<float> pad_info = Letterbox(image, img_input, cv::Size(640, 640));
    //To do Restore to the original image before the change 
    //const float pad_w = pad_info[0];
    //const float pad_h = pad_info[1];
    //const float scale = pad_info[2];

    cv::Mat input;
    cv::cvtColor(img_input, input, cv::COLOR_BGR2RGB);
    torch::Tensor tensor_image = torch::from_blob(input.data, { 1, input.rows, input.cols, 3 }, torch::kByte);
    tensor_image = tensor_image.permute({ 0, 3, 1, 2 }); // Change shape to {1, 3, height, width}
    tensor_image = tensor_image.to(at::kFloat).div(255);
    tensor_image = tensor_image.to(at::kCUDA);

    std::vector<torch::jit::IValue> inputs = { tensor_image };

    torch::jit::script::Module model = torch::jit::load("yolov5/yolov5s-seg-gpu.torchscript");
    model.to(torch::kCUDA);
    model.eval();
    auto net_outputs = model.forward(inputs).toTuple();

    at::Tensor main_output = net_outputs->elements()[0].toTensor();
    at::Tensor mask_output = net_outputs->elements()[1].toTensor();

    at::Tensor cpu_main_output = main_output[0].to(torch::kCPU);
    at::Tensor cpu_mask_output = mask_output[0].to(torch::kCPU);
    cpu_mask_output = cpu_mask_output.reshape({ 32, -1 });
	int index1 = main_output.sizes()[2];
    int index2 = main_output.sizes()[2]- 32;
	//从目标检测的输出中获取协方差矩阵 后面会用于与mask做矩阵乘法 原始模型里 这两个值是117和32 
	//协方差矩阵的大小是1x32
    at::Tensor cpu_main_outputproposals = cpu_main_output.slice(1, index2, index1);

    std::vector<torch::Tensor> dets = nms(main_output, 0.5, 0.5);


    if (dets.size() > 0)
    {
        at::Tensor proposals = dets[0].slice(1, index2, index1);
        //mask  nx25600
        at::Tensor proposals_res = proposals.matmul(cpu_mask_output);
        //reshape nx160x160

        // Visualize result
        int total = 0;
        for (size_t i = 0; i < dets[0].sizes()[0]; ++i)
        {
            float left = dets[0][i][0].item().toFloat();
            float top = dets[0][i][1].item().toFloat();
            float right = dets[0][i][2].item().toFloat();
            float bottom = dets[0][i][3].item().toFloat();
            float score = dets[0][i][4].item().toFloat();
            int classID = dets[0][i][5].item().toInt();
			
            cv::Mat mask_res = cv::Mat::zeros(cv::Size(160, 160), CV_32FC1);
            std::memcpy((void*)mask_res.data, proposals_res[i].data_ptr(), sizeof(float) * 25600);
            cv::Mat dest, seg_mask;
            cv::exp(-mask_res, dest);
            dest = 1.0 / (1.0 + dest);
            resize(dest, seg_mask, cv::Size(640, 640), cv::INTER_NEAREST);
            cv::Rect mask_rect = cv::Rect{ cv::Point((int)right,(int)top),cv::Point((int)left,(int)bottom) };
            mask_rect &= cv::Rect(0, 0, 640, 640);
			
			// segment thresh 0.5
            seg_mask = seg_mask(mask_rect) > 0.5; 
            cv::Mat imshow_mask = img_input.clone();
            imshow_mask(mask_rect).setTo(cv::Scalar(0, 0, 255), seg_mask);
			
			//imshow
            cv::addWeighted(img_input, 0.5, imshow_mask, 0.5, 0, img_input);
            cv::rectangle(img_input, cv::Rect(left, top, (right - left), (bottom - top)), cv::Scalar(0, 255, 0), 2);//画框


            total = total + 1;
        }
    }

    cv::imshow("img_input", img_input);
    cv::waitKey(0);
    return 0;
}