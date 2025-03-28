#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <fstream>

#define BLOCK_SIZE 16

using namespace cv;
using namespace std;
using namespace std::chrono;

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

    Mat quantizedImage(image.rows, image.cols, CV_8UC3);
    cudaMemcpy(quantizedImage.data, d_output, imgSize, cudaMemcpyDeviceToHost);

    Mat cartoonImage;
    bitwise_and(quantizedImage, quantizedImage, cartoonImage, edges);

    cudaFree(d_input);
    cudaFree(d_output);

    image = cartoonImage;
}

int main() {
    VideoCapture cap(0); // Open the default webcam
    if (!cap.isOpened()) {
        cerr << "Error: Unable to access webcam!" << endl;
        return -1;
    }

    ofstream csvFile("cartoon_timings.csv");
    csvFile << "Frame,CPU_Time(ms),GPU_Time(ms)\n";

    int frameCount = 0;

    while (true) {
        Mat frame;
        cap >> frame; // Capture frame
        if (frame.empty()) break;

        Mat imageCPU = frame.clone();
        Mat imageGPU = frame.clone();

        // Measure CPU Processing Time
        auto startCPU = high_resolution_clock::now();
        applyCartoonEffectCPU(imageCPU);
        auto stopCPU = high_resolution_clock::now();
        auto durationCPU = duration_cast<milliseconds>(stopCPU - startCPU).count();

        // Measure GPU Processing Time
        auto startGPU = high_resolution_clock::now();
        applyCartoonEffectGPU(imageGPU);
        auto stopGPU = high_resolution_clock::now();
        auto durationGPU = duration_cast<milliseconds>(stopGPU - startGPU).count();

        // Save frame times to CSV
        csvFile << frameCount << "," << durationCPU << "," << durationGPU << "\n";
        frameCount++;

        // Display both frames
        imshow("Cartoon Effect - CPU", imageCPU);
        imshow("Cartoon Effect - GPU", imageGPU);

        // Break loop on 'q' key press
        if (waitKey(1) == 'q') break;
    }

    csvFile.close();
    cap.release();
    destroyAllWindows();

    return 0;
}
