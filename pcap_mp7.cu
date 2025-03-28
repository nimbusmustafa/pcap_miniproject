#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>
#include <iostream>

#define BLOCK_SIZE 16
#define MAX_INTENSITY_BINS 256

using namespace cv;
using namespace cv::xphoto;

// CUDA Kernel: Color Quantization (Reduce Colors)
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

// CUDA Kernel: Grayscale Conversion
__global__ void grayscaleKernel(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];
        unsigned char gray = (r * 0.299 + g * 0.587 + b * 0.114);
        output[y * width + x] = gray;
    }
}

// CUDA Kernel: Inversion
__global__ void invertKernel(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = 255 - input[idx];
    }
}

// CUDA Kernel: Gaussian Blur (Simplified, not optimized)
__global__ void gaussianBlurKernel(unsigned char *input, unsigned char *output, int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float sum = 0.0f;
        int count = 0;
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    sum += input[ny * width + nx];
                    count++;
                }
            }
        }
        output[idx] = (unsigned char)(sum / count);
    }
}

// CUDA Kernel: Color Dodge Blending
__global__ void colorDodgeKernel(unsigned char *gray, unsigned char *blur, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float result = gray[idx] / (1.0f - (blur[idx] / 255.0f));
        output[idx] = (unsigned char)min(result, 255.0f);
    }
}

// CUDA Kernel: Oil Painting Effect
__global__ void oilPaintingKernel(unsigned char *input, unsigned char *output, int width, int height, int radius, int intensityLevels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        
        // Initialize intensity bins
        int intensityCount[MAX_INTENSITY_BINS] = {0};
        int averageR[MAX_INTENSITY_BINS] = {0};
        int averageG[MAX_INTENSITY_BINS] = {0};
        int averageB[MAX_INTENSITY_BINS] = {0};
        
        // Calculate intensity for neighboring pixels
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int nIdx = (ny * width + nx) * 3;
                    unsigned char r = input[nIdx];
                    unsigned char g = input[nIdx + 1];
                    unsigned char b = input[nIdx + 2];
                    int curIntensity = (int)((double)((r + g + b) / 3) * intensityLevels) / 255;
                    if (curIntensity >= 0 && curIntensity < MAX_INTENSITY_BINS) {
                        intensityCount[curIntensity]++;
                        averageR[curIntensity] += r;
                        averageG[curIntensity] += g;
                        averageB[curIntensity] += b;
                    }
                }
            }
        }
        
        // Find the most populated intensity bin
        int maxCount = 0;
        int maxIndex = 0;
        for (int i = 0; i < MAX_INTENSITY_BINS; i++) {
            if (intensityCount[i] > maxCount) {
                maxCount = intensityCount[i];
                maxIndex = i;
            }
        }
        
        // Calculate final color
        if (maxCount > 0) {
            unsigned char finalR = (unsigned char)(averageR[maxIndex] / maxCount);
            unsigned char finalG = (unsigned char)(averageG[maxIndex] / maxCount);
            unsigned char finalB = (unsigned char)(averageB[maxIndex] / maxCount);
            output[idx] = finalR;
            output[idx + 1] = finalG;
            output[idx + 2] = finalB;
        } else {
            // Handle case where no neighbors are found
            output[idx] = input[idx];
            output[idx + 1] = input[idx + 1];
            output[idx + 2] = input[idx + 2];
        }
    }
}

// CUDA Kernel: Sepia Effect
__global__ void sepiaKernel(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];
        unsigned char newR = (unsigned char)(0.393 * r + 0.769 * g + 0.189 * b);
        unsigned char newG = (unsigned char)(0.349 * r + 0.686 * g + 0.168 * b);
        unsigned char newB = (unsigned char)(0.272 * r + 0.534 * g + 0.131 * b);
        output[idx] = newR;
        output[idx + 1] = newG;
        output[idx + 2] = newB;
    }
}

// CUDA Kernel: Negative Effect
__global__ void negativeKernel(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        output[idx] = 255 - input[idx];
        output[idx + 1] = 255 - input[idx + 1];
        output[idx + 2] = 255 - input[idx + 2];
    }
}

void applyCartoonEffect(Mat &image) {
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
    colorQuantizationKernel<<<grid, block>>>(d_input, d_output, image.cols, image.rows, 64);
    
    Mat quantizedImage(image.rows, image.cols, CV_8UC3);
    cudaMemcpy(quantizedImage.data, d_output, imgSize, cudaMemcpyDeviceToHost);
    
    Mat cartoonImage;
    bitwise_and(quantizedImage, quantizedImage, cartoonImage, edges);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    image = cartoonImage; // Update the original image with the cartoon effect
}

void applyOilPaintingEffect(Mat &image) {
    int imgSize = image.rows * image.cols * 3;
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);
    cudaMemcpy(d_input, image.data, imgSize, cudaMemcpyHostToDevice);
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((image.cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (image.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    oilPaintingKernel<<<grid, block>>>(d_input, d_output, image.cols, image.rows, 5, 20);
    
    cudaMemcpy(image.data, d_output, imgSize, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

void applyManualPencilSketchEffect(Mat &image) {
    int imgSize = image.rows * image.cols * 3;
    unsigned char *d_input;
    cudaMalloc(&d_input, imgSize);
    cudaMemcpy(d_input, image.data, imgSize, cudaMemcpyHostToDevice);
    
    // Grayscale Conversion
    unsigned char *d_gray;
    cudaMalloc(&d_gray, image.rows * image.cols);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((image.cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (image.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    grayscaleKernel<<<grid, block>>>(d_input, d_gray, image.cols, image.rows);
    
    // Inversion
    unsigned char *d_inv;
    cudaMalloc(&d_inv, image.rows * image.cols);
    invertKernel<<<grid, block>>>(d_gray, d_inv, image.cols, image.rows);
    
    // Gaussian Blur
    unsigned char *d_blur;
    cudaMalloc(&d_blur, image.rows * image.cols);
    gaussianBlurKernel<<<grid, block>>>(d_inv, d_blur, image.cols, image.rows, 10);
    
    // Color Dodge Blending
    unsigned char *d_sketch;
    cudaMalloc(&d_sketch, image.rows * image.cols);
    colorDodgeKernel<<<grid, block>>>(d_gray, d_blur, d_sketch, image.cols, image.rows);
    
    // Create a new grayscale Mat to hold the sketch
    Mat sketch(image.rows, image.cols, CV_8UC1);
    cudaMemcpy(sketch.data, d_sketch, image.rows * image.cols, cudaMemcpyDeviceToHost);
    
    // Convert the grayscale sketch to BGR
    cvtColor(sketch, image, COLOR_GRAY2BGR); // Update the original image
    
    cudaFree(d_input);
    cudaFree(d_gray);
    cudaFree(d_inv);
    cudaFree(d_blur);
    cudaFree(d_sketch);
}

void applySepiaEffect(Mat &image) {
    int imgSize = image.rows * image.cols * 3;
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);
    cudaMemcpy(d_input, image.data, imgSize, cudaMemcpyHostToDevice);
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((image.cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (image.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    sepiaKernel<<<grid, block>>>(d_input, d_output, image.cols, image.rows);
    
    cudaMemcpy(image.data, d_output, imgSize, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

void applyNegativeEffect(Mat &image) {
    int imgSize = image.rows * image.cols * 3;
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);
    cudaMemcpy(d_input, image.data, imgSize, cudaMemcpyHostToDevice);
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((image.cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (image.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    negativeKernel<<<grid, block>>>(d_input, d_output, image.cols, image.rows);
    
    cudaMemcpy(image.data, d_output, imgSize, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    VideoCapture cap(0); // Open default webcam
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return -1;
    }

    int choice = 0; // Default: No effect

    while (true) {
        Mat frame;
        cap >> frame; // Capture frame

        if (frame.empty()) {
            std::cerr << "Failed to capture frame." << std::endl;
            continue; // Skip processing this iteration
        }


        switch (choice) {
            case 0: { 
                imshow("Webcam", frame); // Display original feed
                break;
            }
            case 1: { 
                Mat tempFrame = frame.clone();
                applyCartoonEffect(tempFrame);
                imshow("Webcam", tempFrame);
                break;
            }
            case 2: { 
                applyOilPaintingEffect(frame);
                imshow("Webcam", frame);
                break;
            }
            case 3: { 
                applyManualPencilSketchEffect(frame);
                imshow("Webcam", frame);
                break;
            }
            case 4: { 
                applySepiaEffect(frame);
                imshow("Webcam", frame);
                break;
            }
            case 5: { 
                applyNegativeEffect(frame);
                imshow("Webcam", frame);
                break;
            }
            default:
                imshow("Webcam", frame); // Display original feed
        }

        char c = (char)waitKey(10);
        if (c == '0') choice = 0; // Default feed
        else if (c == '1') choice = 1; // Cartoon effect
        else if (c == '2') choice = 2; // Oil painting effect
        else if (c == '3') choice = 3; // Pencil sketch effect
        else if (c == '4') choice = 4; // Sepia effect
        else if (c == '5') choice = 5; // Negative effect
        else if (c == 27) break; // ESC to exit
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
