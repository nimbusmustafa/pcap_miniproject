#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <fstream>

#define BLOCK_SIZE 16

using namespace cv;
using namespace std;
using namespace std::chrono;

__global__ void sepiaKernel(unsigned char *input, unsigned char *output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int index = (y * width + x) * channels;
    float r = input[index];
    float g = input[index + 1];
    float b = input[index + 2];
    output[index] = min(255.0f, (r * 0.393f + g * 0.769f + b * 0.189f));
    output[index + 1] = min(255.0f, (r * 0.349f + g * 0.686f + b * 0.168f));
    output[index + 2] = min(255.0f, (r * 0.272f + g * 0.534f + b * 0.131f));
}

__global__ void negativeKernel(unsigned char *input, unsigned char *output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int index = (y * width + x) * channels;
    output[index] = 255 - input[index];     
    output[index + 1] = 255 - input[index + 1];  
    output[index + 2] = 255 - input[index + 2];  
}

void applySepiaEffectCPU(Mat &image) {
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            Vec3b pixel = image.at<Vec3b>(y, x);
            float r = pixel[2], g = pixel[1], b = pixel[0];
            image.at<Vec3b>(y, x)[2] = min(255.0f, (r * 0.393f + g * 0.769f + b * 0.189f));
            image.at<Vec3b>(y, x)[1] = min(255.0f, (r * 0.349f + g * 0.686f + b * 0.168f));
            image.at<Vec3b>(y, x)[0] = min(255.0f, (r * 0.272f + g * 0.534f + b * 0.131f));
        }
    }
}

void applyNegativeEffectCPU(Mat &image) {
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            Vec3b pixel = image.at<Vec3b>(y, x);
            image.at<Vec3b>(y, x) = Vec3b(255 - pixel[0], 255 - pixel[1], 255 - pixel[2]);
        }
    }
}

float applyEffectGPU(Mat &image, void (*kernel)(unsigned char*, unsigned char*, int, int, int)) {
    int imgSize = image.rows * image.cols * 3;
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);
    cudaMemcpy(d_input, image.data, imgSize, cudaMemcpyHostToDevice);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((image.cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (image.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel<<<grid, block>>>(d_input, d_output, image.cols, image.rows, 3);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(image.data, d_output, imgSize, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    return milliseconds;
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error opening webcam!" << endl;
        return -1;
    }

    ofstream sepiaCSV("sepia_timings.csv");
    ofstream negativeCSV("negative_timings.csv");
    sepiaCSV << "Frame, CPU_Time(ms), GPU_Time(ms)\n";
    negativeCSV << "Frame, CPU_Time(ms), GPU_Time(ms)\n";

    int frameCount = 0;
    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        
        Mat imageSepiaCPU = frame.clone();
        auto startCPU = high_resolution_clock::now();
        applySepiaEffectCPU(imageSepiaCPU);
        auto stopCPU = high_resolution_clock::now();
        auto cpuTimeSepia = duration_cast<milliseconds>(stopCPU - startCPU).count();
        
        Mat imageSepiaGPU = frame.clone();
        float gpuTimeSepia = applyEffectGPU(imageSepiaGPU, sepiaKernel);
        
        sepiaCSV << frameCount << ", " << cpuTimeSepia << ", " << gpuTimeSepia << "\n";

        Mat imageNegativeCPU = frame.clone();
        startCPU = high_resolution_clock::now();
        applyNegativeEffectCPU(imageNegativeCPU);
        stopCPU = high_resolution_clock::now();
        auto cpuTimeNegative = duration_cast<milliseconds>(stopCPU - startCPU).count();
        
        Mat imageNegativeGPU = frame.clone();
        float gpuTimeNegative = applyEffectGPU(imageNegativeGPU, negativeKernel);
        
        negativeCSV << frameCount << ", " << cpuTimeNegative << ", " << gpuTimeNegative << "\n";

        imshow("Sepia CPU", imageSepiaCPU);
        imshow("Sepia GPU", imageSepiaGPU);
        imshow("Negative CPU", imageNegativeCPU);
        imshow("Negative GPU", imageNegativeGPU);

        frameCount++;
        if (waitKey(1) == 27) break;
    }

    sepiaCSV.close();
    negativeCSV.close();
    cap.release();
    destroyAllWindows();
    return 0;
}
