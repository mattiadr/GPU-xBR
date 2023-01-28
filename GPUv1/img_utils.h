#ifndef IMG_UTILS
#define IMG_UTILS

typedef struct {
	unsigned char B;
	unsigned char G;
	unsigned char R;
} PixelRGB3;

typedef struct __align__(4) {
	unsigned char B;
	unsigned char G;
	unsigned char R;
} PixelRGB;

typedef struct __align__(4) {
	unsigned char Y;
	unsigned char U;
	unsigned char V;
} PixelYUV;

__device__ PixelYUV rgb_to_yuv(PixelRGB input) {
	PixelYUV output;
	output.Y =   0.299f * input.R + 0.587f * input.G + 0.114f * input.B;
	output.U = - 0.169f * input.R - 0.331f * input.G + 0.500f * input.B;
	output.V =   0.500f * input.R - 0.419f * input.G - 0.081f * input.B;
	return output;
}

__global__ void rgb_to_rgb3_kernel(unsigned int length, PixelRGB *input, PixelRGB3 *output) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= length)
		return;
	PixelRGB p = input[index];
	output[index] = (PixelRGB3) {.R = p.R, .G = p.G, .B = p.B};
}

__global__ void rgb3_to_rgb_kernel(unsigned int length, PixelRGB3 *input, PixelRGB *output) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= length)
		return;
	PixelRGB3 p = input[index];
	output[index] = (PixelRGB) {.R = p.R, .G = p.G, .B = p.B};
}

#endif
