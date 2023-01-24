#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

#include "img_utils.h"
#include "xBR_utils.h"


void expand_frame(unsigned int rows, unsigned int cols, PixelRGB *input, PixelRGB *output, unsigned int scaleFactor) {
	PixelYUV *yuv_data = (PixelYUV *) malloc(rows * cols * sizeof (PixelYUV));
	rgb_to_yuv(rows * cols, input, yuv_data);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			expand_pixel(rows, cols, input, yuv_data, output, scaleFactor, r, c);
		}
	}
}

void expand_image(std::string input_path, std::string output_path, unsigned int scaleFactor) {
	cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
	PixelRGB *rgb_data = (PixelRGB *) img.data;
	PixelRGB *output = (PixelRGB *) malloc(img.rows * img.cols * scaleFactor * scaleFactor * sizeof (PixelRGB));

	expand_frame(img.rows, img.cols, rgb_data, output, scaleFactor);

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

	cv::Mat frame;
	while (1) {
		video >> frame;
		if (frame.empty())
			break;
		rgb_data = (PixelRGB *) frame.data;

		expand_frame(frame_heigth, frame_width, rgb_data, output, scaleFactor);

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
