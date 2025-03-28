#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

using namespace cv;
using namespace std;

// CUDA Kernel for Grayscale Conversion
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

// CUDA Kernel for Color Dodge Blend
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

// Gaussian Blur (CPU)
Mat gaussianBlurCPU(const Mat &img) {
    Mat blurred;
    GaussianBlur(img, blurred, Size(21, 21), 0);
    return blurred;
}

// Pencil Sketch (CPU)
Mat pencilSketchCPU(const Mat &gray, const Mat &blurred) {
    Mat sketch;
    divide(gray, 255 - blurred, sketch, 256);
    return sketch;
}

// CUDA Pencil Sketch Function
Mat pencilSketchCUDA(const Mat &input) {
    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();

    size_t imgSize = width * height * channels;
    size_t graySize = width * height;

    // Allocate memory on GPU
    unsigned char *d_input, *d_gray, *d_blurred, *d_output;
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_gray, graySize);
    cudaMalloc(&d_blurred, graySize);
    cudaMalloc(&d_output, graySize);

    // Copy image to GPU
    cudaMemcpy(d_input, input.data, imgSize, cudaMemcpyHostToDevice);

    // Launch Grayscale Kernel
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    grayscaleKernel<<<grid, block>>>(d_input, d_gray, width, height, channels);
    cudaDeviceSynchronize();

    // Copy grayscale image to host for Gaussian Blur (since OpenCV GaussianBlur is more optimized)
    Mat gray(height, width, CV_8UC1);
    cudaMemcpy(gray.data, d_gray, graySize, cudaMemcpyDeviceToHost);

    // Apply Gaussian Blur (CPU)
    Mat blurred = gaussianBlurCPU(gray);

    // Copy blurred image back to GPU
    cudaMemcpy(d_blurred, blurred.data, graySize, cudaMemcpyHostToDevice);

    // Launch Color Dodge Kernel
    colorDodgeKernel<<<grid, block>>>(d_gray, d_blurred, d_output, width, height);
    cudaDeviceSynchronize();

    // Copy result back to host
    Mat sketch(height, width, CV_8UC1);
    cudaMemcpy(sketch.data, d_output, graySize, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_gray);
    cudaFree(d_blurred);
    cudaFree(d_output);

    return sketch;
}

int main() {
    Mat input = imread("input2.jpg");

    if (input.empty()) {
        cout << "Error: Could not load image!" << endl;
        return -1;
    }

    // Convert to Grayscale (CPU)
    Mat gray;
    cvtColor(input, gray, COLOR_BGR2GRAY);

    // Measure CPU Time
    double t1 = (double)getTickCount();
    Mat blurredCPU = gaussianBlurCPU(gray);
    Mat sketchCPU = pencilSketchCPU(gray, blurredCPU);
    double t2 = (double)getTickCount();
    double cpuTime = (t2 - t1) / getTickFrequency();
    cout << "CPU Time: " << cpuTime << " seconds" << endl;

    // Measure GPU Time
    t1 = (double)getTickCount();
    Mat sketchGPU = pencilSketchCUDA(input);
    t2 = (double)getTickCount();
    double gpuTime = (t2 - t1) / getTickFrequency();
    cout << "GPU Time: " << gpuTime << " seconds" << endl;

    // Save output images
    imwrite("img/pencil_sketch_cpu.jpg", sketchCPU);
    imwrite("img/pencil_sketch_gpu.jpg", sketchGPU);

    return 0;
}
