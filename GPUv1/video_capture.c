#include <X11/X.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>

#include "img_utils.h"
#include "xBR_utils.h"

#define BLOCK_DIM 32
#define TILE_DIM 28

__global__ void expand_pixel_kernel(const unsigned int rows, const unsigned int cols, const PixelRGB *input, PixelRGB *output, const unsigned int scaleFactor);

void expand_screenshot_frame(unsigned int rows, unsigned int cols, PixelRGB *d_input, PixelRGB *d_output, unsigned int scaleFactor) {
	// expand pixels
	dim3 threadsPerBlock_expand(BLOCK_DIM, BLOCK_DIM);
	dim3 blocks_expand(ceil(cols / (float) TILE_DIM), ceil(rows / (float) TILE_DIM));

	expand_pixel_kernel<<<blocks_expand, threadsPerBlock_expand>>>(rows, cols, d_input, d_output, scaleFactor);

	cudaError_t err = cudaDeviceSynchronize();

	if (err != cudaSuccess){
		printf("Failed to launch kernel (error code %d) %s!\n", cudaGetLastError(), cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void expand_capture(unsigned int scaleFactor) {
	// capture window
	Display *disp = XOpenDisplay(NULL);
	Window root = RootWindow(disp, XScreenNumberOfScreen(ScreenOfDisplay(disp, DefaultScreen(disp))));

	Cursor curs = XCreateFontCursor(disp, XC_crosshair);
	if ((XGrabPointer(disp, root, False, ButtonPressMask | ButtonReleaseMask, GrabModeSync, GrabModeAsync, root, curs, CurrentTime) != GrabSuccess)) {
		fprintf(stdout, "Couldn't grab pointer.\n");
		exit(1);
	}

	Window win = None;

	XEvent ev;
	int buttons = 0;
	while (win == None || buttons != 0) {
		XAllowEvents(disp, SyncPointer, CurrentTime);
		XWindowEvent(disp, root, ButtonPressMask | ButtonReleaseMask, &ev);
		switch (ev.type) {
			case ButtonPress:
				if (win == None) {
					win = ev.xbutton.subwindow;
					if (win == None) {
						win = root;
					}
				}
				buttons++;
				break;
			case ButtonRelease:
				if (buttons > 0) {
					buttons--;
				}
				break;
		}
	}

	XUngrabPointer(disp, CurrentTime);

	if (win == None) {
		fprintf(stderr, "Couldn't grab window.\n");
		exit(1);
	}

	Window root_ret, parent_ret, *child_ret;
	unsigned int child_count;
	if (XQueryTree(disp, win, &root_ret, &parent_ret, &child_ret, &child_count)) {
		if (child_count >= 2) {
			win = child_ret[child_count - 1];
		}
	}

	// get attributes
	XWindowAttributes xwa;
	if (!XGetWindowAttributes(disp, win, &xwa)) {
		fprintf(stdout, "Couldn't get window attributes.\n");
		exit(1);
	}

	unsigned int width = xwa.width;
	unsigned int height = xwa.height;

	// allocate memory
	PixelRGB *input = (PixelRGB *) malloc(height * width * sizeof (PixelRGB));

	// allocate memory on device
	PixelRGB *d_input, *d_output;

	cudaMalloc((void **) &d_input, (height * width) * sizeof(PixelRGB));
	cudaMalloc((void **) &d_output, (height * width * scaleFactor * scaleFactor) * sizeof(PixelRGB));

	XImage *image = XGetImage(disp, win, 0, 0, width, height, AllPlanes, ZPixmap);

	if (image == NULL) {
		fprintf(stdout, "Couldn't get image.\n");
		exit(1);
	}

	unsigned char frame_count = 0;
	std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();

	cv::Mat mat_out = cv::Mat(height * scaleFactor, width * scaleFactor, CV_8UC4);

	while (1) {
		// grab image
		image = XGetSubImage(disp, win, 0, 0, width, height, AllPlanes, ZPixmap, image, 0, 0);
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				unsigned long pixel = XGetPixel(image, x, y);
				input[y * width + x].B = (unsigned char) (pixel & image->blue_mask);
				input[y * width + x].G = (unsigned char) ((pixel & image->green_mask) >> 8);
				input[y * width + x].R = (unsigned char) ((pixel & image->red_mask) >> 16);
			}
		}

		// copy data to device
		cudaMemcpy(d_input, input, (height * width) * sizeof(PixelRGB), cudaMemcpyHostToDevice);

		expand_screenshot_frame(height, width, d_input, d_output, scaleFactor);

		// copy data from device
		cudaMemcpy(mat_out.data, d_output, (height * width * scaleFactor * scaleFactor) * sizeof(PixelRGB), cudaMemcpyDeviceToHost);

		cv::imshow("Output", mat_out);
		cv::waitKey(1);

		frame_count++;
		if (frame_count >= 100) {
			std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
			std::cout << "FPS: " << frame_count / std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count() << std::endl;
			frame_count = 0;
			start_time = now;
		}
	}
}
