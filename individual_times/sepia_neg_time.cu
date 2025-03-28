#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

#define BLOCK_SIZE 16

using namespace cv;
using namespace std;
using namespace std::chrono;

// CUDA Kernel: Sepia Effect
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

// CUDA Kernel: Negative Effect
__global__ void negativeKernel(unsigned char *input, unsigned char *output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int index = (y * width + x) * channels;
    output[index] = 255 - input[index];     
    output[index + 1] = 255 - input[index + 1];  
    output[index + 2] = 255 - input[index + 2];  
}

// CPU Function for Sepia Effect
void applySepiaEffectCPU(Mat &image) {
    Mat result = image.clone();
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            Vec3b pixel = image.at<Vec3b>(y, x);
            float r = pixel[2], g = pixel[1], b = pixel[0];

            result.at<Vec3b>(y, x)[2] = min(255.0f, (r * 0.393f + g * 0.769f + b * 0.189f));
            result.at<Vec3b>(y, x)[1] = min(255.0f, (r * 0.349f + g * 0.686f + b * 0.168f));
            result.at<Vec3b>(y, x)[0] = min(255.0f, (r * 0.272f + g * 0.534f + b * 0.131f));
        }
    }
    image = result;
}

// CPU Function for Negative Effect
void applyNegativeEffectCPU(Mat &image) {
    Mat result = image.clone();
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            Vec3b pixel = image.at<Vec3b>(y, x);
            result.at<Vec3b>(y, x) = Vec3b(255 - pixel[0], 255 - pixel[1], 255 - pixel[2]);
        }
    }
    image = result;
}

// GPU Function for Sepia Effect
void applySepiaEffectGPU(Mat &image) {
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

    cudaEventRecord(start, 0);
    sepiaKernel<<<grid, block>>>(d_input, d_output, image.cols, image.rows, 3);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "CUDA Sepia Processing Time: " << milliseconds << " ms" << endl;

    cudaMemcpy(image.data, d_output, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

// GPU Function for Negative Effect
void applyNegativeEffectGPU(Mat &image) {
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

    cudaEventRecord(start, 0);
    negativeKernel<<<grid, block>>>(d_input, d_output, image.cols, image.rows, 3);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "CUDA Negative Processing Time: " << milliseconds << " ms" << endl;

    cudaMemcpy(image.data, d_output, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    Mat image = imread("input2.jpg");
    if (image.empty()) {
        cerr << "Error loading image!" << endl;
        return -1;
    }

    // CPU Timing
    Mat imageSepiaCPU = image.clone();
    auto startSepiaCPU = high_resolution_clock::now();
    applySepiaEffectCPU(imageSepiaCPU);
    auto stopSepiaCPU = high_resolution_clock::now();
    auto durationSepiaCPU = duration_cast<milliseconds>(stopSepiaCPU - startSepiaCPU);
    cout << "CPU Sepia Processing Time: " << durationSepiaCPU.count() << " ms" << endl;

    Mat imageNegativeCPU = image.clone();
    auto startNegativeCPU = high_resolution_clock::now();
    applyNegativeEffectCPU(imageNegativeCPU);
    auto stopNegativeCPU = high_resolution_clock::now();
    auto durationNegativeCPU = duration_cast<milliseconds>(stopNegativeCPU - startNegativeCPU);
    cout << "CPU Negative Processing Time: " << durationNegativeCPU.count() << " ms" << endl;

    // GPU Timing
    Mat imageSepiaGPU = image.clone();
    applySepiaEffectGPU(imageSepiaGPU);

    Mat imageNegativeGPU = image.clone();
    applyNegativeEffectGPU(imageNegativeGPU);

    // Save Results
    imwrite("img/output_sepia_cpu.jpg", imageSepiaCPU);
    imwrite("img/output_sepia_gpu.jpg", imageSepiaGPU);
    imwrite("img/output_negative_cpu.jpg", imageNegativeCPU);
    imwrite("img/output_negative_gpu.jpg", imageNegativeGPU);

    return 0;
}
