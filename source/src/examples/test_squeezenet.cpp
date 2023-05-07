#include <string>
#include <chrono>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "amat.h"
#include "net.h"


using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = (int)cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
        std::greater<std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

void test_squeezenet()
{
    std::string image_path = "D:/06DL/frigate/data/cat.jpg";
    cv::Mat img = cv::imread(image_path);
    cv::Mat img_resize;
    cv::resize(img, img_resize, cv::Size(227, 227));
    /*cv::imshow("img", img_resize);
    cv::waitKey(0);*/

    {
        img_resize.convertTo(img_resize, CV_32FC3);
        const std::vector<float> mean_values = { 104.f, 117.f, 123.f };
        cv::Mat arr[3];
        cv::split(img_resize, arr);

        const int input_w = 227;
        const int input_h = 227;
        float* pfdata = new float[input_w * input_h * 3];
        std::vector<cv::Mat> input;
        for (int i = 0; i < 3; i++)
        {
            cv::Mat channel(input_h, input_w, CV_32FC1, pfdata + i * input_w * input_h);
            input.push_back(channel);
        }
        for (int i = 0; i < 3; i++)
        {
            input[i] = arr[i] - mean_values[i];
        }

        ACNN::Net squeezenet;
        squeezenet.opt.use_sgemm_convolution = true;
        squeezenet.load_param("D:/06DL/frigate/data/squeezenet_v1.1.param");
        squeezenet.load_model("D:/06DL/frigate/data/squeezenet_v1.1.bin");
        squeezenet.set_input_name("data");
        squeezenet.set_output_name("prob");

        ACNN::aMat input_data(input_w, input_h, 3, pfdata, sizeof(float));
        ACNN::aMat output_data;
        high_resolution_clock::time_point startTime = high_resolution_clock::now();
        for (int i = 0; i < 1; i++)
        {
            squeezenet.forward(input_data, output_data);
        }
        high_resolution_clock::time_point endTime = high_resolution_clock::now();
        milliseconds timeInterval = std::chrono::duration_cast<milliseconds>(endTime - startTime);
        std::cout << "squeezenet.forward cost time: " << timeInterval.count() << " ms" << std::endl;

        std::vector<float> cls_scores;
        cls_scores.resize(output_data.m_w);
        for (int j = 0; j < output_data.m_w; j++)
        {
            cls_scores[j] = output_data[j];
        }

        print_topk(cls_scores, 3);

        delete[] pfdata;
    }
}

void test_squeezenet_int8()
{
    std::string image_path = "D:/06DL/frigate/data/cat.jpg";
    cv::Mat img = cv::imread(image_path);
    cv::Mat img_resize;
    cv::resize(img, img_resize, cv::Size(227, 227));
    /*cv::imshow("img", img_resize);
    cv::waitKey(0);*/

    {
        img_resize.convertTo(img_resize, CV_32FC3);
        const std::vector<float> mean_values = { 104.f, 117.f, 123.f };
        cv::Mat arr[3];
        cv::split(img_resize, arr);

        const int input_w = 227;
        const int input_h = 227;
        float* pfdata = new float[input_w * input_h * 3];
        std::vector<cv::Mat> input;
        for (int i = 0; i < 3; i++)
        {
            cv::Mat channel(input_h, input_w, CV_32FC1, pfdata + i * input_w * input_h);
            input.push_back(channel);
        }
        for (int i = 0; i < 3; i++)
        {
            input[i] = arr[i] - mean_values[i];
        }

        ACNN::Net squeezenet;
        squeezenet.opt.use_sgemm_convolution = true;
        squeezenet.opt.use_int8_inference = true;
        squeezenet.load_param("D:/06DL/frigate/data/squeezenet_v1.1.int8.param");
        squeezenet.load_model("D:/06DL/frigate/data/squeezenet_v1.1.int8.bin");
        squeezenet.set_input_name("data");
        squeezenet.set_output_name("prob");

        ACNN::aMat input_data(input_w, input_h, 3, pfdata, sizeof(float));
        ACNN::aMat output_data;
        high_resolution_clock::time_point startTime = high_resolution_clock::now();
        for (int i = 0; i < 1; i++)
        {
            squeezenet.forward(input_data, output_data);
        }
        high_resolution_clock::time_point endTime = high_resolution_clock::now();
        milliseconds timeInterval = std::chrono::duration_cast<milliseconds>(endTime - startTime);
        std::cout << "squeezenet.forward cost time: " << timeInterval.count() << " ms" << std::endl;

        std::vector<float> cls_scores;
        cls_scores.resize(output_data.m_w);
        for (int j = 0; j < output_data.m_w; j++)
        {
            cls_scores[j] = output_data[j];
        }

        print_topk(cls_scores, 3);

        delete[] pfdata;
    }
}