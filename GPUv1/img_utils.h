#ifndef IMG_UTILS
#define IMG_UTILS

typedef struct __align__(4) {
	unsigned char B;
	unsigned char G;
	unsigned char R;
	unsigned char A;
} PixelRGB;

typedef struct __align__(4) {
	unsigned char Y;
	unsigned char U;
	unsigned char V;
	unsigned char A;
} PixelYUV;

__global__ void prepare_data_kernel(unsigned int length, unsigned char *input, PixelRGB *rgb_data, PixelYUV *yuv_data) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= length)
		return;

	PixelRGB p;
	p.B = input[index * 3];
	p.G = input[index * 3 + 1];
	p.R = input[index * 3 + 2];
	p.A = 255;

	rgb_data[index] = p;

	yuv_data[index].Y =   0.299 * rgb_data[index].R + 0.587 * rgb_data[index].G + 0.114 * rgb_data[index].B;
	yuv_data[index].U = - 0.169 * rgb_data[index].R - 0.331 * rgb_data[index].G + 0.500 * rgb_data[index].B;
	yuv_data[index].V =   0.500 * rgb_data[index].R - 0.419 * rgb_data[index].G - 0.081 * rgb_data[index].B;
}

#endif
