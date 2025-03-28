#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

using namespace cv;
using namespace std;

__global__ void grayscaleKernel(unsigned char *input, unsigned char *output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];
        output[y * width + x] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

__global__ void colorDodgeKernel(unsigned char *gray, unsigned char *blurred, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        unsigned char g = gray[idx];
        unsigned char b = blurred[idx];
        output[idx] = (b == 255) ? 255 : min(255, (g * 255) / (255 - b + 1));
    }
}

Mat gaussianBlurCPU(const Mat &img) {
    Mat blurred;
    GaussianBlur(img, blurred, Size(21, 21), 0);
    return blurred;
}

Mat pencilSketchCPU(const Mat &gray, const Mat &blurred) {
    Mat sketch;
    divide(gray, 255 - blurred, sketch, 256);
    return sketch;
}

Mat pencilSketchCUDA(const Mat &input) {
    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();

    size_t imgSize = width * height * channels;
    size_t graySize = width * height;

    unsigned char *d_input, *d_gray, *d_blurred, *d_output;
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_gray, graySize);
    cudaMalloc(&d_blurred, graySize);
    cudaMalloc(&d_output, graySize);

    cudaMemcpy(d_input, input.data, imgSize, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    grayscaleKernel<<<grid, block>>>(d_input, d_gray, width, height, channels);
    cudaDeviceSynchronize();

    Mat gray(height, width, CV_8UC1);
    cudaMemcpy(gray.data, d_gray, graySize, cudaMemcpyDeviceToHost);

    Mat blurred = gaussianBlurCPU(gray);
    cudaMemcpy(d_blurred, blurred.data, graySize, cudaMemcpyHostToDevice);

    colorDodgeKernel<<<grid, block>>>(d_gray, d_blurred, d_output, width, height);
    cudaDeviceSynchronize();

    Mat sketch(height, width, CV_8UC1);
    cudaMemcpy(sketch.data, d_output, graySize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_gray);
    cudaFree(d_blurred);
    cudaFree(d_output);

    return sketch;
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error: Could not open webcam!" << endl;
        return -1;
    }

    ofstream csvFile("pencilsketch_timings.csv");
    csvFile << "Frame,CPU Time (s),GPU Time (s)" << endl;

    int frameCount = 0;
    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        double t1 = (double)getTickCount();
        Mat blurredCPU = gaussianBlurCPU(gray);
        Mat sketchCPU = pencilSketchCPU(gray, blurredCPU);
        double t2 = (double)getTickCount();
        double cpuTime = (t2 - t1) / getTickFrequency();
        cpuTime=cpuTime*1000;

        t1 = (double)getTickCount();
        Mat sketchGPU = pencilSketchCUDA(frame);
        t2 = (double)getTickCount();
        double gpuTime = (t2 - t1) / getTickFrequency();
        gpuTime=gpuTime*1000;


        csvFile << frameCount++ << "," << cpuTime << "," << gpuTime << endl;

        imshow("CPU Pencil Sketch", sketchCPU);
        imshow("GPU Pencil Sketch", sketchGPU);

        if (waitKey(1) == 27) break;
    }

    csvFile.close();
    cap.release();
    destroyAllWindows();
    return 0;
}
