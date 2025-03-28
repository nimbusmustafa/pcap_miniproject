#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

#define BLOCK_SIZE 16

using namespace cv;
using namespace std;
using namespace std::chrono;

// CUDA Kernel: Color Quantization
__global__ void colorQuantizationKernel(unsigned char *input, unsigned char *output, int width, int height, int levels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        for (int i = 0; i < 3; i++) {
            output[idx + i] = (input[idx + i] / levels) * levels;
        }
    }
}

// CPU Function for Cartoon Effect
void applyCartoonEffectCPU(Mat &image) {
    Mat smoothImage;
    bilateralFilter(image, smoothImage, 9, 75, 75);

    Mat grayImage;
    cvtColor(smoothImage, grayImage, COLOR_BGR2GRAY);

    Mat edges;
    Sobel(grayImage, edges, CV_8U, 1, 1);
    threshold(edges, edges, 50, 255, THRESH_BINARY_INV);

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            Vec3b &pixel = image.at<Vec3b>(y, x);
            for (int i = 0; i < 3; i++) {
                pixel[i] = (pixel[i] / 64) * 64;
            }
        }
    }

    Mat cartoonImage;
    bitwise_and(image, image, cartoonImage, edges);
    image = cartoonImage;
}

// GPU Function for Cartoon Effect
void applyCartoonEffectGPU(Mat &image) {
    Mat smoothImage;
    bilateralFilter(image, smoothImage, 9, 75, 75);

    Mat grayImage;
    cvtColor(smoothImage, grayImage, COLOR_BGR2GRAY);

    Mat edges;
    Sobel(grayImage, edges, CV_8U, 1, 1);
    threshold(edges, edges, 50, 255, THRESH_BINARY_INV);

    int imgSize = image.rows * image.cols * 3;
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);
    cudaMemcpy(d_input, smoothImage.data, imgSize, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((image.cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (image.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    colorQuantizationKernel<<<grid, block>>>(d_input, d_output, image.cols, image.rows, 64);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "CUDA Processing Time: " << milliseconds << " ms" << std::endl;

    Mat quantizedImage(image.rows, image.cols, CV_8UC3);
    cudaMemcpy(quantizedImage.data, d_output, imgSize, cudaMemcpyDeviceToHost);

    Mat cartoonImage;
    bitwise_and(quantizedImage, quantizedImage, cartoonImage, edges);

    cudaFree(d_input);
    cudaFree(d_output);

    image = cartoonImage;
}

int main() {
    Mat image = imread("input2.jpg");
    if (image.empty()) {
        cerr << "Error loading image!" << endl;
        return -1;
    }

    // CPU Timing
    Mat imageCPU = image.clone();
    auto startCPU = high_resolution_clock::now();
    applyCartoonEffectCPU(imageCPU);
    auto stopCPU = high_resolution_clock::now();
    auto durationCPU = duration_cast<milliseconds>(stopCPU - startCPU);
    cout << "CPU Processing Time: " << durationCPU.count() << " ms" << endl;

    // GPU Timing
    Mat imageGPU = image.clone();
    applyCartoonEffectGPU(imageGPU);

    // Save Results
    imwrite("img/output_cartoon_cpu.jpg", imageCPU);
    imwrite("img/output_cartoon_gpu.jpg", imageGPU);

    return 0;
}
