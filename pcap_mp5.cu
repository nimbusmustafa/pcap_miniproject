#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>
#include <iostream>


#define BLOCK_SIZE 16  

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

void applyCartoonEffect(Mat &image) {
    Mat smoothImage;
    bilateralFilter(image, smoothImage, 9, 75, 75);
    
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    
    Mat edges;
    Sobel(grayImage, edges, CV_8U, 1, 1);
    threshold(edges, edges, 50, 255, THRESH_BINARY_INV);
    
    int imgSize = image.rows * image.cols * 3;
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);
    cudaMemcpy(d_input, image.data, imgSize, cudaMemcpyHostToDevice);
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((image.cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (image.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    colorQuantizationKernel<<<grid, block>>>(d_input, d_output, image.cols, image.rows, 64);
    
    Mat quantizedImage(image.rows, image.cols, CV_8UC3);
    cudaMemcpy(quantizedImage.data, d_output, imgSize, cudaMemcpyDeviceToHost);
    
    Mat cartoonImage;
    bitwise_and(quantizedImage, quantizedImage, cartoonImage, edges);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    imshow("Cartoon Effect", cartoonImage);
    waitKey(0);
}

void applyOilPaintingEffect(Mat &image) {
    Mat oilPaintedImage;
    oilPainting(image, oilPaintedImage, 7, 1);
    imshow("Oil Painting Effect", oilPaintedImage);
    waitKey(0);
}

void applyManualPencilSketchEffect(Mat &image) {
    Mat gray, inv, blur, sketch;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    bitwise_not(gray, inv);
    GaussianBlur(inv, blur, Size(21, 21), 0);
    divide(gray, 255 - blur, sketch, 256);
    
    imshow("Pencil Sketch Effect", sketch);
    waitKey(0);
}


int main() {
    Mat image = imread("input2.jpg");
    if (image.empty()) {
        std::cerr << "Image not found!" << std::endl;
        return -1;
    }
    
    int choice;
    std::cout << "Choose Effect:\n1. Cartoon\n2. Oil Painting\n3. Pencil Sketch\n";
    std::cin >> choice;
    
    switch (choice) {
        case 1: applyCartoonEffect(image); break;
        case 2: applyOilPaintingEffect(image); break;
        case 3: applyManualPencilSketchEffect(image); break;
        default: std::cout << "Invalid choice!" << std::endl;
    }
    return 0;
}
