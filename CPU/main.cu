#include <opencv2/opencv.hpp>
#include <iostream>

#include "img_utils.h"
#include "xBR_utils.h"

int main(int argc, char const *argv[]) {
	std::string image_path = argv[2];
	cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);

	int scaleFactor = atoi(argv[1]);

	PixelRGB *rgb_data = (PixelRGB *) img.data;
	PixelYUV *yuv_data = (PixelYUV *) malloc(img.rows * img.cols * sizeof (PixelYUV));
	PixelRGB *output = (PixelRGB *) malloc(img.rows * img.cols * scaleFactor * scaleFactor * sizeof (PixelRGB));

	rgb_to_yuv(img.rows * img.cols, rgb_data, yuv_data);

	for (int r = 0; r < img.rows; r++) {
		for (int c = 0; c < img.cols; c++) {
			expand_pixel(img.rows, img.cols, rgb_data, yuv_data, output, scaleFactor, r, c);
		}
	}

	cv::Mat img_out(img.rows * scaleFactor, img.cols * scaleFactor, CV_8UC3, (void *) output);
	cv::imwrite(argv[3], img_out);

	return 0;
}

