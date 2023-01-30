#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <cmath>

#include "img_utils.h"
#include "xBR_utils.h"

#include "video_capture.c"

#define BLOCK_DIM 32
#define TILE_DIM 28
#define OFFSET 2

__global__ void expand_pixel_kernel(const unsigned int rows, const unsigned int cols, const PixelRGB *input, PixelRGB *output, const unsigned int scaleFactor) {
	const int t_row = threadIdx.y;
	const int t_col = threadIdx.x;

	const int o_row = blockIdx.y * TILE_DIM + t_row;
	const int o_col = blockIdx.x * TILE_DIM + t_col;

	const int i_row = o_row - OFFSET;
	const int i_col = o_col - OFFSET;

	// declare shared memory
	__shared__ PixelRGB sh_inputRGB[BLOCK_DIM * BLOCK_DIM];
	__shared__ PixelYUV sh_inputYUV[BLOCK_DIM * BLOCK_DIM];

	// copy data in shared memory
	PixelRGB p;
	if (i_row >= 0 && i_row < rows && i_col >= 0 && i_col < cols) {
		p = input[i_row * cols + i_col];
	} else if (i_row < 0 && i_col < 0) {
		// top left corner
		p = input[0 * cols + 0];
	} else if (i_row < 0 && i_col >= cols) {
		// top right corner
		p = input[0 * cols + (cols - 1)];
	} else if (i_row >= rows && i_col < 0) {
		// bottom left corner
		p = input[(rows - 1) * cols + 0];
	} else if (i_row >= rows && i_col >= cols) {
		// bottom right corner
		p = input[(rows - 1) * cols + (cols - 1)];
	} else if (i_row < 0) {
		// top edge
		p = input[0 * cols + i_col];
	} else if (i_row >= rows) {
		// bottom edge
		p = input[(rows - 1) * cols + i_col];
	} else if (i_col < 0) {
		// left edge
		p = input[i_row * cols + 0];
	} else if (i_col >= cols) {
		// right edge
		p = input[i_row * cols + (cols - 1)];
	} else {
		// this should never happen
		assert(false);
	}

	__syncthreads();

	sh_inputRGB[t_row * blockDim.x + t_col] = p;
	sh_inputYUV[t_row * blockDim.x + t_col] = rgb_to_yuv(p);

	__syncthreads();

	if (t_row >= TILE_DIM || t_col >= TILE_DIM || o_row >= rows || o_col >= cols) {
		return;
	}

	expand_pixel_tiling(cols, BLOCK_DIM, sh_inputRGB, sh_inputYUV, output, scaleFactor, t_row + OFFSET, t_col + OFFSET, o_row, o_col);
}

void expand_frame(unsigned int rows, unsigned int cols, PixelRGB3 *d_input_3, PixelRGB3 *d_output_3, PixelRGB *d_input, PixelRGB *d_output, unsigned int scaleFactor) {
	// convert frame from 3 to 4 channels
	dim3 threadsPerBlock_convert(min(1024, rows * cols));
	dim3 blocks_convert(ceil(rows * cols / (float) threadsPerBlock_convert.x));

	rgb3_to_rgb_kernel<<<blocks_convert, threadsPerBlock_convert>>>(rows * cols, d_input_3, d_input);

	cudaError_t err = cudaDeviceSynchronize();

	if (err != cudaSuccess){
		printf("Failed to launch kernel (error code %d) %s!\n", cudaGetLastError(), cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// expand pixels
	dim3 threadsPerBlock_expand(BLOCK_DIM, BLOCK_DIM);
	dim3 blocks_expand(ceil(cols / (float) TILE_DIM), ceil(rows / (float) TILE_DIM));

	expand_pixel_kernel<<<blocks_expand, threadsPerBlock_expand>>>(rows, cols, d_input, d_output, scaleFactor);

	err = cudaDeviceSynchronize();

	if (err != cudaSuccess){
		printf("Failed to launch kernel (error code %d) %s!\n", cudaGetLastError(), cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// convert frame from 4 to 3 channels
	blocks_convert.x *= scaleFactor * scaleFactor;
	rgb_to_rgb3_kernel<<<blocks_convert, threadsPerBlock_convert>>>(rows * cols * scaleFactor * scaleFactor, d_output, d_output_3);

	err = cudaDeviceSynchronize();

	if (err != cudaSuccess){
		printf("Failed to launch kernel (error code %d) %s!\n", cudaGetLastError(), cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void expand_image(std::string input_path, std::string output_path, unsigned int scaleFactor) {
	cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);

	PixelRGB3 *output = (PixelRGB3 *) malloc(img.rows * img.cols * scaleFactor * scaleFactor * sizeof (PixelRGB3));

	// allocate memory on device
	PixelRGB3 *d_input_3, *d_output_3;
	PixelRGB *d_input, *d_output;

	cudaMalloc((void **) &d_input_3, (img.rows * img.cols) * sizeof(PixelRGB3));
	cudaMalloc((void **) &d_output_3, (img.rows * img.cols * scaleFactor * scaleFactor) * sizeof(PixelRGB3));
	cudaMalloc((void **) &d_input, (img.rows * img.cols) * sizeof(PixelRGB));
	cudaMalloc((void **) &d_output, (img.rows * img.cols * scaleFactor * scaleFactor) * sizeof(PixelRGB));

	// copy data to device
	cudaMemcpy(d_input_3, img.data, (img.rows * img.cols) * sizeof(PixelRGB3), cudaMemcpyHostToDevice);

	expand_frame(img.rows, img.cols, d_input_3, d_output_3, d_input, d_output, scaleFactor);

	// copy data from device
	cudaMemcpy(output, d_output_3, (img.rows * img.cols * scaleFactor * scaleFactor) * sizeof(PixelRGB3), cudaMemcpyDeviceToHost);

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

	PixelRGB *output = (PixelRGB *) malloc(frame_heigth * frame_width * scaleFactor * scaleFactor * sizeof (PixelRGB));
	cv::VideoWriter out_video(output_path, fourcc, fps, cv::Size(frame_width * scaleFactor, frame_heigth * scaleFactor));

	// allocate memory on device
	PixelRGB3 *d_input_3, *d_output_3;
	PixelRGB *d_input, *d_output;

	cudaMalloc((void **) &d_input_3, (frame_heigth * frame_width) * sizeof(PixelRGB3));
	cudaMalloc((void **) &d_output_3, (frame_heigth * frame_width * scaleFactor * scaleFactor) * sizeof(PixelRGB3));
	cudaMalloc((void **) &d_input, (frame_heigth * frame_width) * sizeof(PixelRGB));
	cudaMalloc((void **) &d_output, (frame_heigth * frame_width * scaleFactor * scaleFactor) * sizeof(PixelRGB));

	// process frames
	cv::Mat frame;
	while (1) {
		video >> frame;
		if (frame.empty())
			break;

		// copy data to device
		cudaMemcpy(d_input_3, frame.data, (frame_heigth * frame_width) * sizeof(PixelRGB3), cudaMemcpyHostToDevice);

		expand_frame(frame_heigth, frame_width, d_input_3, d_output_3, d_input, d_output, scaleFactor);

		// copy data from device
		cudaMemcpy(output, d_output_3, (frame_heigth * frame_width * scaleFactor * scaleFactor) * sizeof(PixelRGB3), cudaMemcpyDeviceToHost);

		cv::Mat frame_out = cv::Mat(frame_heigth * scaleFactor, frame_width * scaleFactor, CV_8UC3, (void *) output);

		// save frame to video
		out_video << frame_out;
	}
	video.release();
	out_video.release();
}

int main(int argc, char const *argv[]) {
	if (argc < 3) {
		std::cout << "USAGE - " << argv[0] << ": scaleFactor type [inputFile] [outputFile]" << std::endl;
		std::cout << "TYPES: (i)mage, (v)ideo, (c)apture" << std::endl;
		return 0;
	}

	int scaleFactor = atoi(argv[1]);
	std::string type = argv[2];
	
	if (type == "i" || type == "image") {
		if (argc < 5) {
			std::cerr << "Missing file" << std::endl;
			return 1;
		}
		std::string input_path = argv[3];
		std::string output_path = argv[4];
		expand_image(input_path, output_path, scaleFactor);
		return 0;
	} else if (type == "v" || type == "video") {
		if (argc < 5) {
			std::cerr << "Missing file" << std::endl;
			return 1;
		}
		std::string input_path = argv[3];
		std::string output_path = argv[4];
		expand_video(input_path, output_path, scaleFactor);
		return 0;
	} else if (type == "c" || type == "capture") {
		expand_capture(scaleFactor);
	}

	std::cerr << "Type not recognized" << std::endl;

	return 1;
}
