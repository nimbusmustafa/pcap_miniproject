#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <fstream>

#define BLOCK_SIZE 16
#define RADIUS 5  // Neighborhood size for oil painting effect
#define INTENSITY_LEVELS 256

using namespace cv;
using namespace std;
using namespace std::chrono;

// CUDA Kernel: Oil Painting Effect
__global__ void oilPaintingKernel(unsigned char *input, unsigned char *output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int hist[INTENSITY_LEVELS] = {0};
    int avgColor[INTENSITY_LEVELS][3] = {0};

    for (int dy = -RADIUS; dy <= RADIUS; dy++) {
        for (int dx = -RADIUS; dx <= RADIUS; dx++) {
            int nx = min(max(x + dx, 0), width - 1);
            int ny = min(max(y + dy, 0), height - 1);
            int index = (ny * width + nx) * channels;

            int intensity = (input[index] + input[index + 1] + input[index + 2]) / 3;
            hist[intensity]++;
            for (int c = 0; c < 3; c++) {
                avgColor[intensity][c] += input[index + c];
            }
        }
    }

    int maxIntensity = 0, maxCount = 0;
    for (int i = 0; i < INTENSITY_LEVELS; i++) {
        if (hist[i] > maxCount) {
            maxCount = hist[i];
            maxIntensity = i;
        }
    }

    int pixelIdx = (y * width + x) * channels;
    for (int c = 0; c < 3; c++) {
        output[pixelIdx + c] = avgColor[maxIntensity][c] / maxCount;
    }
}

// CPU Function for Oil Painting Effect
void applyOilPaintingEffectCPU(Mat &image) {
    Mat result = image.clone();

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            int hist[INTENSITY_LEVELS] = {0};
            int avgColor[INTENSITY_LEVELS][3] = {0};

            for (int dy = -RADIUS; dy <= RADIUS; dy++) {
                for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                    int nx = min(max(x + dx, 0), image.cols - 1);
                    int ny = min(max(y + dy, 0), image.rows - 1);

                    Vec3b pixel = image.at<Vec3b>(ny, nx);
                    int intensity = (pixel[0] + pixel[1] + pixel[2]) / 3;
                    hist[intensity]++;
                    for (int c = 0; c < 3; c++) {
                        avgColor[intensity][c] += pixel[c];
                    }
                }
            }

            int maxIntensity = 0, maxCount = 0;
            for (int i = 0; i < INTENSITY_LEVELS; i++) {
                if (hist[i] > maxCount) {
                    maxCount = hist[i];
                    maxIntensity = i;
                }
            }

            for (int c = 0; c < 3; c++) {
                result.at<Vec3b>(y, x)[c] = avgColor[maxIntensity][c] / maxCount;
            }
        }
    }

    image = result;
}

// GPU Function for Oil Painting Effect
float applyOilPaintingEffectGPU(Mat &image) {
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
    oilPaintingKernel<<<grid, block>>>(d_input, d_output, image.cols, image.rows, 3);
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
        cerr << "Error: Could not open webcam!" << endl;
        return -1;
    }

    ofstream file("oilpainting_timings.csv");
    file << "Frame,CPU_Time(ms),GPU_Time(ms)\n";

    int frameCount = 0;
    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        Mat imageCPU = frame.clone();
        auto startCPU = high_resolution_clock::now();
        applyOilPaintingEffectCPU(imageCPU);
        auto stopCPU = high_resolution_clock::now();
        auto durationCPU = duration_cast<milliseconds>(stopCPU - startCPU);
        float cpuTime = durationCPU.count();

        Mat imageGPU = frame.clone();
        float gpuTime = applyOilPaintingEffectGPU(imageGPU);

        file << frameCount << "," << cpuTime << "," << gpuTime << "\n";

        imshow("Original", frame);
        imshow("CPU Oil Painting", imageCPU);
        imshow("GPU Oil Painting", imageGPU);

        frameCount++;

        if (waitKey(1) == 27) break;  // Press 'ESC' to exit
    }

    file.close();
    cap.release();
    destroyAllWindows();

    return 0;
}
