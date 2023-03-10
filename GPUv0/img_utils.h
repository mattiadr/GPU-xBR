#ifndef IMG_UTILS
#define IMG_UTILS

typedef struct {
	unsigned char B;
	unsigned char G;
	unsigned char R;
} PixelRGB;

typedef struct {
	unsigned char Y;
	unsigned char U;
	unsigned char V;
} PixelYUV;

__global__ void rgb_to_yuv_kernel(unsigned int length, PixelRGB *input, PixelYUV *output) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= length)
		return;
	output[index].Y =   0.299 * input[index].R + 0.587 * input[index].G + 0.114 * input[index].B;
	output[index].U = - 0.169 * input[index].R - 0.331 * input[index].G + 0.500 * input[index].B;
	output[index].V =   0.500 * input[index].R - 0.419 * input[index].G - 0.081 * input[index].B;
}

#endif
