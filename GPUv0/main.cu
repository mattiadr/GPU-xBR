#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

#include "img_utils.h"
#include "xBR_utils.h"


__global__ void expand_pixel_kernel(unsigned int rows, unsigned int cols, PixelRGB *inputRGB, PixelYUV *inputYUV, PixelRGB *output, unsigned int scaleFactor) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col >= cols || row >= rows)
		return;

	expand_pixel(rows, cols, inputRGB, inputYUV, output, scaleFactor, row, col);
}

void expand_frame(unsigned int rows, unsigned int cols, PixelRGB *d_input, PixelYUV *d_yuv_data, PixelRGB *d_output, unsigned int scaleFactor) {
	dim3 threadsPerBlock = dim3(min(1024, rows * cols));
	dim3 blocks = dim3(ceil(rows * cols / (float)threadsPerBlock.x));

	// printf("rgb_to_yuv_kernel: (%d, %d) blocks with (%d, %d) threads...\n", blocks.x, blocks.y, threadsPerBlock.x, threadsPerBlock.y);
	rgb_to_yuv_kernel<<<blocks, threadsPerBlock>>>(rows * cols, d_input, d_yuv_data);
	cudaError_t err = cudaDeviceSynchronize();

	if (err != cudaSuccess){
		printf("Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	threadsPerBlock = dim3(min(cols, 32), min(rows, 32));
	blocks = dim3(ceil(cols / (float)threadsPerBlock.x), ceil(rows / (float)threadsPerBlock.y));

	// printf("expand_pixel_kernel: (%d, %d) blocks with (%d, %d) threads...\n", blocks.x, blocks.y, threadsPerBlock.x, threadsPerBlock.y);
	expand_pixel_kernel<<<blocks, threadsPerBlock>>>(rows, cols, d_input, d_yuv_data, d_output, scaleFactor);
	// std::cout << cudaGetLastError() << std::endl;

	err = cudaDeviceSynchronize();

	if (err != cudaSuccess){
		printf("Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

}

void expand_image(std::string input_path, std::string output_path, unsigned int scaleFactor) {
	cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
	PixelRGB *rgb_data = (PixelRGB *) img.data;
	PixelRGB *output = (PixelRGB *) malloc(img.rows * img.cols * scaleFactor * scaleFactor * sizeof (PixelRGB));

	PixelRGB *d_rgb_data, *d_output;
	PixelYUV *d_yuv_data;
	cudaMalloc((void**)&d_rgb_data, (img.rows * img.cols) * sizeof(PixelRGB));
	cudaMalloc((void**)&d_output, (img.rows * img.cols * scaleFactor * scaleFactor) * sizeof(PixelRGB));
	cudaMalloc((void**)&d_yuv_data, (img.rows * img.cols) * sizeof(PixelYUV));

	cudaMemcpy(d_rgb_data, rgb_data, (img.rows * img.cols) * sizeof(PixelRGB), cudaMemcpyHostToDevice);

	expand_frame(img.rows, img.cols, d_rgb_data, d_yuv_data, d_output, scaleFactor);

	cudaMemcpy(output, d_output, (img.rows * img.cols * scaleFactor * scaleFactor) * sizeof(PixelRGB), cudaMemcpyDeviceToHost);

	cv::Mat img_out(img.rows * scaleFactor, img.cols * scaleFactor, CV_8UC3, (void *) output);
	cv::imwrite(output_path, img_out);
}

void expand_video(std::string input_path, std::string output_path, unsigned int scaleFactor) {
	cv::VideoCapture video(input_path);

	if (!video.isOpened()) {
		std::cerr << "Error opening video file" << std::endl;
		return;
	}

	double fps = video.get(cv::CAP_PROP_FPS);
	double frame_width = video.get(cv::CAP_PROP_FRAME_WIDTH);
	double frame_heigth = video.get(cv::CAP_PROP_FRAME_HEIGHT);
	int fourcc = static_cast<int>(video.get(cv::CAP_PROP_FOURCC));

	PixelRGB *rgb_data;
	PixelRGB *output = (PixelRGB *) malloc(frame_heigth * frame_width * scaleFactor * scaleFactor * sizeof (PixelRGB));
	cv::VideoWriter out_video(output_path, fourcc, fps, cv::Size(frame_width * scaleFactor, frame_heigth * scaleFactor));

	// allocate memory on device
	PixelRGB *d_rgb_data, *d_output;
	PixelYUV *d_yuv_data;
	cudaMalloc((void**)&d_rgb_data, (frame_heigth * frame_width) * sizeof(PixelRGB));
	cudaMalloc((void**)&d_output, (frame_heigth * frame_width * scaleFactor * scaleFactor) * sizeof(PixelRGB));
	cudaMalloc((void**)&d_yuv_data, (frame_heigth * frame_width) * sizeof(PixelYUV));

	// process frames
	cv::Mat frame;
	while (1) {
		video >> frame;
		if (frame.empty())
			break;
		rgb_data = (PixelRGB *) frame.data;

		// copy data to device
		cudaMemcpy(d_rgb_data, rgb_data, (frame_heigth * frame_width) * sizeof(PixelRGB), cudaMemcpyHostToDevice);

		expand_frame(frame_heigth, frame_width, d_rgb_data, d_yuv_data, d_output, scaleFactor);

		// copy data from device
		cudaMemcpy(output, d_output, (frame_heigth * frame_width * scaleFactor * scaleFactor) * sizeof(PixelRGB), cudaMemcpyDeviceToHost);

		// save frame to video
		cv::Mat frame_out = cv::Mat(frame_heigth * scaleFactor, frame_width * scaleFactor, CV_8UC3, (uchar*) output);
		out_video << frame_out;
	}
	video.release();
	out_video.release();
}

int main(int argc, char const *argv[]) {
	if (argc < 5) {
		std::cout << "USAGE - " << argv[0] << ": scaleFactor type inputFile outputFile" << std::endl;
		std::cout << "TYPES: (i)mage, (v)ideo" << std::endl;
		return 0;
	}

	int scaleFactor = atoi(argv[1]);
	std::string type = argv[2];
	std::string input_path = argv[3];
	std::string output_path = argv[4];
	
	if (type == "i" || type == "image") {
		expand_image(input_path, output_path, scaleFactor);
		return 0;
	} else if (type == "v" || type == "video") {
		expand_video(input_path, output_path, scaleFactor);
		return 0;
	}

	std::cerr << "Type not recognized" << std::endl;

	return 1;
}
